"""Dataset with random 256×256 cropping, rotation, and flip augmentations for CT slice interpolation.

Strategy: Focus on center 384×384 region of 512×512 image, creating 9 crops with stride 64.

Training Augmentations (all 256×256):
- Random 1-of-N selection with configurable weights (default: 9 crops)
- Center crop (position 4) has 2x weight by default
- Optional resize options (512→256, 384→256) with configurable weights
- Rotation: configurable probability, uniform in [-max, +max] degrees
- Random H/V flips

Validation:
- All 9 crops returned per sample (no resizes, no rotation)
- Per-crop metrics tracked alongside aggregated metrics

For test-time reconstruction:
- Run inference on 9 overlapping crops
- Reconstruct center 384×384 region by averaging overlaps
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image

from .two_to_one_slice import STANDARD_TRANSFORM, TwoToOneSliceDataset

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

# Crop configuration
CROP_SIZE = 256
CENTER_REGION_SIZE = 384  # Focus on center 384×384 region
CENTER_OFFSET = (512 - CENTER_REGION_SIZE) // 2  # 64 pixels offset on each side
CROP_STRIDE = 64  # Stride within 384×384 region to get 3×3 grid
NUM_CROPS = 9  # 3x3 grid of crops

# Crop positions in original 512×512 space (y, x) top-left coordinates
# These extract 256×256 crops from the center 384×384 region in a 3x3 grid
CROP_POSITIONS = [
    (64, 64),  # 0: Top-left
    (64, 128),  # 1: Top-center
    (64, 192),  # 2: Top-right
    (128, 64),  # 3: Middle-left
    (128, 128),  # 4: Center
    (128, 192),  # 5: Middle-right
    (192, 64),  # 6: Bottom-left
    (192, 128),  # 7: Bottom-center
    (192, 192),  # 8: Bottom-right
]

# Default crop weights (center has 2x weight)
DEFAULT_CROP_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0)


def reconstruct_from_crops(
    crops: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a 512×512 image from 9 overlapping 256×256 crops.

    The crops cover the center 384×384 region with 64px stride overlap.
    Overlapping pixels are averaged. The outer 64px border is black (zero-padded).

    Args:
        crops: Tensor of shape (9, C, 256, 256) with predictions for each crop position.
               Crops must be ordered according to CROP_POSITIONS (row-major 3×3 grid).

    Returns:
        Reconstructed tensor of shape (C, 512, 512) with:
        - Center 384×384 region filled with averaged predictions
        - Outer 64px border filled with zeros (black)
    """
    if crops.shape[0] != NUM_CROPS:
        raise ValueError(f"Expected {NUM_CROPS} crops, got {crops.shape[0]}")

    channels = crops.shape[1]
    device = crops.device
    dtype = crops.dtype

    # Output canvas (512×512) and count for averaging overlaps
    output = torch.zeros(channels, 512, 512, device=device, dtype=dtype)
    counts = torch.zeros(1, 512, 512, device=device, dtype=dtype)

    # Place each crop into the output canvas at its original position
    for crop_idx, (orig_y, orig_x) in enumerate(CROP_POSITIONS):
        # Add crop to output at original 512×512 coordinates
        output[:, orig_y : orig_y + CROP_SIZE, orig_x : orig_x + CROP_SIZE] += crops[crop_idx]
        counts[:, orig_y : orig_y + CROP_SIZE, orig_x : orig_x + CROP_SIZE] += 1

    # Average overlapping regions (avoid division by zero in border)
    output = output / counts.clamp(min=1)

    return output


class TwoToOneSliceCroppedDataset(TwoToOneSliceDataset):
    """Dataset that applies random 256×256 crops, rotation, and flips during training.

    During training (consistent per triplet):
        - Random 1-of-N selection with configurable weights
        - Center crop (position 4) has 2x weight by default
        - Optional resize options (512→256, 384→256) with configurable weights
        - Rotation: configurable probability, uniform in [-max, +max] degrees
        - Random horizontal flip (flip_prob probability)
        - Random vertical flip (flip_prob probability)

    During validation:
        - If return_all_augmentations=True, returns all 9 crops (no resizes, no rotation)
          as 256×256 versions for per-crop validation metrics
        - No flip augmentations

    All augmentations are applied consistently to all slices in a triplet
    to maintain spatial alignment. All inputs are always 256×256.
    """

    def __init__(
        self,
        root_dir: str,
        stage: str,
        size: int | None = None,
        transform: Callable = STANDARD_TRANSFORM,
        seed: int = 42,
        original_window: tuple[int] = (-20, 107),
        target_window: tuple[int] | None = None,
        log_info: bool = True,
        flip_prob: float = 0.5,
        return_all_augmentations: bool = False,
        crop_weights: tuple[float, ...] = DEFAULT_CROP_WEIGHTS,
        include_resize_512: bool = False,
        resize_512_weight: float = 0.5,
        include_resize_384: bool = False,
        resize_384_weight: float = 0.5,
        rotation_prob: float = 0.5,
        rotation_max_angle: float = 15.0,
    ) -> None:
        """Initialize cropped dataset with augmentations.

        Args:
            root_dir: Root directory containing images
            stage: Dataset stage ("train", "valid", or "test")
            size: Number of samples to use (None for full dataset)
            transform: Transform to apply after augmentations
            seed: Random seed for reproducibility
            original_window: Original window parameters
            target_window: Target window parameters
            log_info: Whether to log dataset info
            flip_prob: Probability of applying each flip (default 0.5)
            return_all_augmentations: If True and stage is "valid", return
                all 9 crops as 256×256 versions for per-crop validation metrics
            crop_weights: Weight per crop position (9 values, center=index 4)
            include_resize_512: Include 512→256 resize option in training
            resize_512_weight: Weight for 512→256 resize option
            include_resize_384: Include 384→256 resize option in training
            resize_384_weight: Weight for 384→256 resize option
            rotation_prob: Probability of applying rotation (0-1)
            rotation_max_angle: Max rotation angle in degrees (uniform in [-max, +max])

        """
        super().__init__(
            root_dir=root_dir,
            stage=stage,
            size=size,
            transform=transform,
            seed=seed,
            original_window=original_window,
            target_window=target_window,
            log_info=log_info,
        )
        self.flip_prob = flip_prob
        self.use_augmentations = stage == "train"
        self.return_all_augmentations = return_all_augmentations and stage == "valid"

        # Store augmentation config
        self.crop_weights = crop_weights
        self.include_resize_512 = include_resize_512 and stage == "train"
        self.include_resize_384 = include_resize_384 and stage == "train"
        self.resize_512_weight = resize_512_weight
        self.resize_384_weight = resize_384_weight
        self.rotation_prob = rotation_prob if stage == "train" else 0.0
        self.rotation_max_angle = rotation_max_angle

        # Build combined weights for random selection
        self._build_selection_weights()

    def _build_selection_weights(self) -> None:
        """Build probability weights for random augmentation selection."""
        weights = list(self.crop_weights)  # 9 crop weights
        self.num_options = NUM_CROPS  # Start with 9

        if self.include_resize_384:
            weights.append(self.resize_384_weight)
            self.num_options += 1

        if self.include_resize_512:
            weights.append(self.resize_512_weight)
            self.num_options += 1

        # Normalize to probabilities
        total = sum(weights)
        self.selection_probs = np.array([w / total for w in weights], dtype=np.float64)

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns base sample count (no expansion). Training uses random 1-of-N
        crop selection per sample.

        Returns:
            int: Total number of base samples

        """
        return sum(len(v) for v in self.triplets.values())

    def _get_augmentation_params(self, crop_idx: int | None = None) -> dict | None:
        """Get augmentation parameters for a triplet.

        Args:
            crop_idx: If provided, use this deterministic crop index (for validation).
                     If None, use weighted random selection (for training).

        Returns:
            dict with 'aug_idx', 'do_h_flip', 'do_v_flip', 'rotation_angle' keys,
            or None if no augmentation

        """
        if not self.use_augmentations:
            return None

        # Deterministic crop for validation, weighted random for training
        if crop_idx is not None:
            aug_idx = crop_idx
        else:
            aug_idx = int(self.rng.choice(self.num_options, p=self.selection_probs))

        # Rotation: rotation_prob chance of applying uniform random in [-max, +max]
        rotation_angle = 0.0
        if self.rotation_prob > 0 and self.rng.random() < self.rotation_prob:
            rotation_angle = float(self.rng.uniform(-self.rotation_max_angle, self.rotation_max_angle))

        return {
            "aug_idx": aug_idx,
            "do_h_flip": self.rng.random() < self.flip_prob,
            "do_v_flip": self.rng.random() < self.flip_prob,
            "rotation_angle": rotation_angle,
        }

    def _apply_augmentation(self, img: Image.Image, params: dict | None) -> torch.Tensor:
        """Apply augmentation to a single PIL image.

        Args:
            img: PIL Image to augment
            params: Augmentation parameters from _get_augmentation_params()

        Returns:
            Augmented image as tensor (C, H, W) in range [0, 1]

        """
        if params is None:
            # No augmentation - just convert to tensor
            return tf.to_tensor(img)

        # Apply spatial transform (crop or resize)
        aug_idx = params["aug_idx"]
        if aug_idx < NUM_CROPS:
            # Options 0-8: Crop at one of the 9 positions
            y, x = CROP_POSITIONS[aug_idx]
            img = tf.crop(img, top=y, left=x, height=CROP_SIZE, width=CROP_SIZE)
        elif aug_idx == NUM_CROPS:
            # First resize option (384→256 if enabled, else 512→256)
            if self.include_resize_384:
                img = tf.center_crop(img, CENTER_REGION_SIZE)
                img = tf.resize(
                    img,
                    size=(CROP_SIZE, CROP_SIZE),
                    interpolation=tf.InterpolationMode.LANCZOS,
                )
            else:
                img = tf.resize(
                    img,
                    size=(CROP_SIZE, CROP_SIZE),
                    interpolation=tf.InterpolationMode.LANCZOS,
                )
        else:
            # Second resize option (512→256)
            img = tf.resize(
                img,
                size=(CROP_SIZE, CROP_SIZE),
                interpolation=tf.InterpolationMode.LANCZOS,
            )

        # Convert to tensor (C, H, W) in range [0, 1]
        img_tensor = tf.to_tensor(img)

        # Apply rotation
        rotation_angle = params.get("rotation_angle", 0.0)
        if rotation_angle != 0.0:
            img_tensor = tf.rotate(
                img_tensor,
                angle=rotation_angle,
                interpolation=tf.InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0,
            )

        # Apply flips
        if params["do_h_flip"]:
            img_tensor = tf.hflip(img_tensor)
        if params["do_v_flip"]:
            img_tensor = tf.vflip(img_tensor)

        return img_tensor

    def _apply_spatial_augmentation(self, img_tensor: torch.Tensor, aug_idx: int) -> torch.Tensor:
        """Apply a specific spatial augmentation to a tensor (no flips, no rotation).

        Used for validation to generate all 9 crops deterministically.

        Args:
            img_tensor: Image tensor (C, H, W) at 512×512 resolution
            aug_idx: Augmentation index (0-8 for crops only in validation)

        Returns:
            Augmented tensor (C, 256, 256)

        """
        if aug_idx < NUM_CROPS:
            # Options 0-8: Crop at one of the 9 positions
            y, x = CROP_POSITIONS[aug_idx]
            return tf.crop(img_tensor, top=y, left=x, height=CROP_SIZE, width=CROP_SIZE)
        # This shouldn't be called with aug_idx >= 9 during validation,
        # but keep the resize logic for backward compatibility
        elif aug_idx == NUM_CROPS:
            # Option 9: Crop center 384×384, then resize to 256×256
            cropped = tf.center_crop(img_tensor, CENTER_REGION_SIZE)
            return tf.resize(
                cropped,
                size=(CROP_SIZE, CROP_SIZE),
                interpolation=tf.InterpolationMode.BILINEAR,
            )
        # Option 10: Resize full 512×512 to 256×256
        return tf.resize(
            img_tensor,
            size=(CROP_SIZE, CROP_SIZE),
            interpolation=tf.InterpolationMode.BILINEAR,
        )

    def _get_all_augmentations(
        self, first_img: torch.Tensor, second_img: torch.Tensor, third_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate all 9 crops for validation (no resizes, no rotation).

        Validation always returns exactly 9 crops for per-crop metric tracking.
        No resize augmentations or rotations are applied during validation.

        Args:
            first_img: First slice tensor (1, 512, 512)
            second_img: Second slice tensor (1, 512, 512)
            third_img: Third slice tensor (1, 512, 512)

        Returns:
            Tuple of (augmented_inputs, augmented_targets) where:
                - augmented_inputs: (9, 2, 256, 256) tensor
                - augmented_targets: (9, 1, 256, 256) tensor

        """
        augmented_inputs = []
        augmented_targets = []

        for aug_idx in range(NUM_CROPS):  # Only 9 crops for validation
            aug_first = self._apply_spatial_augmentation(first_img, aug_idx)
            aug_second = self._apply_spatial_augmentation(second_img, aug_idx)
            aug_third = self._apply_spatial_augmentation(third_img, aug_idx)

            # Stack inputs: (2, 256, 256)
            aug_input = torch.stack([aug_first.squeeze(0), aug_third.squeeze(0)], dim=0)
            augmented_inputs.append(aug_input)
            augmented_targets.append(aug_second)

        return torch.stack(augmented_inputs, dim=0), torch.stack(augmented_targets, dim=0)

    def __getitem__(self, idx: int) -> tuple:
        """Get triplet with random crop selection, rotation, and flips in training mode.

        Training mode uses weighted random 1-of-N crop selection per sample.

        Args:
            idx: Base sample index

        Returns:
            Training mode:
                Tuple of (input_tensor, target_tensor) where:
                    - input_tensor: (2, 256, 256) tensor with first and third slices
                    - target_tensor: (1, 256, 256) tensor with second slice

            Validation mode (return_all_augmentations=True):
                Tuple of (aug_inputs, aug_targets) where:
                    - aug_inputs: (9, 2, 256, 256) all 9 crop positions
                    - aug_targets: (9, 1, 256, 256) all 9 crop positions

            Validation mode (return_all_augmentations=False):
                Tuple of (input_tensor, target_tensor) where:
                    - input_tensor: (2, 512, 512) tensor with first and third slices
                    - target_tensor: (1, 512, 512) tensor with second slice

        """
        # Get triplet IDs
        first_id, second_id, third_id = self._get_triplet_ids(idx=idx)

        # Validation with all augmentations: load raw 512x512, return all 9 crops
        if self.return_all_augmentations:
            with Image.open(self.img_dir / f"{first_id}.png") as img:
                first_img = tf.to_tensor(img)
            with Image.open(self.img_dir / f"{second_id}.png") as img:
                second_img = tf.to_tensor(img)
            with Image.open(self.img_dir / f"{third_id}.png") as img:
                third_img = tf.to_tensor(img)
            return self._get_all_augmentations(first_img, second_img, third_img)

        # Training: random crop + rotation + flips
        aug_params = self._get_augmentation_params(crop_idx=None)

        # Load and augment all three images with same parameters
        with Image.open(self.img_dir / f"{first_id}.png") as img:
            first_img = self._apply_augmentation(img, aug_params)

        with Image.open(self.img_dir / f"{second_id}.png") as img:
            second_img = self._apply_augmentation(img, aug_params)

        with Image.open(self.img_dir / f"{third_id}.png") as img:
            third_img = self._apply_augmentation(img, aug_params)

        # Stack tensors (already in C, H, W format from to_tensor)
        # first_img and third_img are (1, H, W), squeeze to (H, W) then stack to (2, H, W)
        input_tensor = torch.stack([first_img.squeeze(0), third_img.squeeze(0)], dim=0)
        # second_img is (1, H, W), keep as is for target
        target_tensor = second_img

        return input_tensor, target_tensor


class TwoToOneSliceTestCroppedDataset(TwoToOneSliceCroppedDataset):
    """Test dataset that always uses full images (no cropping or augmentations)."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize test dataset with cropping and augmentations disabled.

        All arguments are passed to parent class with stage forced to "test".
        """
        # Force stage to "test" to disable cropping and augmentations
        kwargs["stage"] = "test"
        super().__init__(*args, **kwargs)
