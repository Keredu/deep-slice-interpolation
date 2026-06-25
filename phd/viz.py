from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from torch.amp import autocast

# Load environment variables
load_dotenv()

from phd.datasets.interpolation.two_to_one_slice import TwoToOneSliceTestDataset  # noqa: E402
from phd.datasets.interpolation.two_to_one_slice_cropped import (  # noqa: E402
    CENTER_OFFSET,
    CENTER_REGION_SIZE,
    CROP_POSITIONS,
    CROP_SIZE,
    reconstruct_from_crops,
)

# Corner positions for full 512x512 coverage (y, x) top-left coordinates
# These extract 256×256 crops from the 4 corners of the image
CORNER_POSITIONS = [
    (0, 0),  # Top-left
    (0, 256),  # Top-right
    (256, 0),  # Bottom-left
    (256, 256),  # Bottom-right
]


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array from [0,1] range to [0,255] uint8.

    Background (value=0) stays black (pixel=0). This prevents
    per-image normalization from hiding model prediction errors.

    Args:
        arr: Array with values in [0, 1] range (normalized CT data)

    Returns:
        Array with values in [0, 255] range as uint8 for image saving
    """
    arr = np.clip(arr, 0.0, 1.0)
    arr *= 255.0
    return arr.astype(np.uint8)


def extract_corner_patches(image: torch.Tensor) -> torch.Tensor:
    """Extract 4 corner 256x256 patches from 512x512 image.

    Args:
        image: (C, 512, 512) tensor

    Returns:
        (4, C, 256, 256) tensor of corner patches
    """
    patches = []
    for y, x in CORNER_POSITIONS:
        patch = tf.crop(image, top=y, left=x, height=CROP_SIZE, width=CROP_SIZE)
        patches.append(patch)
    return torch.stack(patches, dim=0)


def combine_corner_predictions(corners: torch.Tensor) -> torch.Tensor:
    """Combine 4 corner 256x256 predictions into 512x512 image.

    The corners don't overlap, so no averaging is needed.

    Args:
        corners: (4, C, 256, 256) tensor

    Returns:
        (C, 512, 512) tensor
    """
    channels = corners.shape[1]
    device = corners.device
    dtype = corners.dtype

    output = torch.zeros(channels, 512, 512, device=device, dtype=dtype)

    for corner_idx, (y, x) in enumerate(CORNER_POSITIONS):
        output[:, y : y + CROP_SIZE, x : x + CROP_SIZE] = corners[corner_idx]

    return output


def predict_via_patch_reconstruction(
    *,
    model: torch.nn.Module,
    batch_inputs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Produce 512x512 predictions via 13-patch reconstruction.

    For each sample, runs the model on 9 overlapping 256x256 center crops
    (top-left corners on a 3x3 grid at 64-pixel stride, spanning the central
    384x384 band) plus 4 non-overlapping 256x256 corner patches. The output
    composite combines averaged center predictions for the inner 384x384
    region and corner predictions for the outer 64-pixel border. This is the
    sole evaluation pipeline: the network was trained on 256x256 patches and
    is never run on a 512x512 input in one shot.

    Args:
        model: Trained model in eval mode.
        batch_inputs: (B, 2, 512, 512) test-set inputs.
        device: CUDA or CPU device (model must already be on it).

    Returns:
        (B, 1, 512, 512) composite predictions.
    """
    center_offset = CENTER_OFFSET
    center_end = CENTER_OFFSET + CENTER_REGION_SIZE
    composites = []
    for sample_idx in range(batch_inputs.shape[0]):
        sample_input = batch_inputs[sample_idx]

        # 9 overlapping center crops -> averaged reconstruction
        center_crops = extract_crops(sample_input).to(device)
        center_out = model(center_crops)
        center_reconstructed = reconstruct_from_crops(center_out)

        # 4 corner patches -> outer 64-pixel border
        corner_patches = extract_corner_patches(sample_input).to(device)
        corner_out = model(corner_patches)
        corner_combined = combine_corner_predictions(corner_out)

        composite = corner_combined.clone()
        composite[:, center_offset:center_end, center_offset:center_end] = center_reconstructed[
            :, center_offset:center_end, center_offset:center_end
        ]
        composites.append(composite)

    return torch.stack(composites, dim=0)


def save_prediction_visualization(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    center_predictions: torch.Tensor,
    corner_predictions: torch.Tensor,
    composite_predictions: torch.Tensor,
    save_dir: Path,
) -> None:
    """Save a visualization as a 3x2 tiled image.

    Layout (3×2 grid, each cell 512x512):
    +------------------+------------------+------------------+
    | 1: First slice   | 2: Target slice  | 3: Third slice   |
    |    (input ch0)   |   (ground truth) |    (input ch1)   |
    +------------------+------------------+------------------+
    | 4: Corner        | 5: Composite     | 6: 384x384 with  |
    |    inferences    |  (4 + 6 merged)  |    gray padding  |
    +------------------+------------------+------------------+

    Uses fixed [0,1] window for grayscale normalization to ensure backgrounds stay black.

    Args:
        inputs: Input tensor of shape (B, 2, 512, 512)
        targets: Target tensor of shape (B, 1, 512, 512)
        center_predictions: Center 384x384 reconstruction with black padding (B, 1, 512, 512)
        corner_predictions: 4 corner inferences combined into 512x512 (B, 1, 512, 512)
        composite_predictions: Corners as base + 384x384 center overlaid (B, 1, 512, 512)
        save_dir: Directory to save visualizations
    """
    # Process one sample at a time
    for idx in range(len(inputs)):
        # Convert tensors to numpy arrays and transpose from CHW to HWC
        input_np = inputs[idx].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)
        target_np = targets[idx].cpu().numpy().transpose(1, 2, 0)  # (H, W, 1)
        center_pred_np = center_predictions[idx].cpu().numpy().transpose(1, 2, 0)  # (H, W, 1)
        corner_pred_np = corner_predictions[idx].cpu().numpy().transpose(1, 2, 0)  # (H, W, 1)
        composite_pred_np = composite_predictions[idx].cpu().numpy().transpose(1, 2, 0)  # (H, W, 1)

        # Rescale data to [0, 255] range using fixed [0,1] window
        input_np = _normalize_to_uint8(input_np)
        target_np = _normalize_to_uint8(target_np)
        center_pred_np = _normalize_to_uint8(center_pred_np)
        corner_pred_np = _normalize_to_uint8(corner_pred_np)
        composite_pred_np = _normalize_to_uint8(composite_pred_np)

        # Get dimensions (should be 512x512)
        h, w, _ = input_np.shape

        # Create gray-padded version of center prediction (position 6)
        # Gray value = 128 for the outer 64px border
        gray_padded = np.full((h, w), 128, dtype=np.uint8)
        y0, y1 = CENTER_OFFSET, CENTER_OFFSET + CENTER_REGION_SIZE
        x0, x1 = CENTER_OFFSET, CENTER_OFFSET + CENTER_REGION_SIZE
        gray_padded[y0:y1, x0:x1] = center_pred_np[y0:y1, x0:x1, 0]

        # Create blank canvas (white background) - 3×2 grid of 512×512 cells
        canvas = Image.new("L", (w * 3, h * 2), 255)

        # Top row: First slice, Target, Third slice
        canvas.paste(Image.fromarray(input_np[:, :, 0]), (0 * w, 0))
        canvas.paste(Image.fromarray(target_np[:, :, 0]), (1 * w, 0))
        canvas.paste(Image.fromarray(input_np[:, :, 1]), (2 * w, 0))

        # Bottom row: Corner inferences, Composite, Gray-padded center
        canvas.paste(Image.fromarray(corner_pred_np[:, :, 0]), (0 * w, h))
        canvas.paste(Image.fromarray(composite_pred_np[:, :, 0]), (1 * w, h))
        canvas.paste(Image.fromarray(gray_padded), (2 * w, h))

        # Save the image
        canvas.save(save_dir / f"{idx}.png")


def extract_crops(tensor: torch.Tensor) -> torch.Tensor:
    """Extract 9 crops from a 512x512 tensor.

    Args:
        tensor: Tensor of shape (C, 512, 512) or (B, C, 512, 512)

    Returns:
        If input is (C, 512, 512): returns (9, C, 256, 256)
        If input is (B, C, 512, 512): returns (B, 9, C, 256, 256)
    """
    if tensor.dim() == 3:
        # Single image: (C, H, W)
        crops = []
        for y, x in CROP_POSITIONS:
            crop = tf.crop(tensor, top=y, left=x, height=CROP_SIZE, width=CROP_SIZE)
            crops.append(crop)
        return torch.stack(crops, dim=0)  # (9, C, 256, 256)
    # Batch: (B, C, H, W)
    batch_crops = []
    for img in tensor:
        crops = []
        for y, x in CROP_POSITIONS:
            crop = tf.crop(img, top=y, left=x, height=CROP_SIZE, width=CROP_SIZE)
            crops.append(crop)
        batch_crops.append(torch.stack(crops, dim=0))
    return torch.stack(batch_crops, dim=0)  # (B, 9, C, 256, 256)


def save_test_visualization(
    test_dataset: TwoToOneSliceTestDataset,
    model: torch.nn.Module,
    device: torch.device,
    save_dir: Path,
    batch_size: int,
) -> None:
    """Save test visualizations for a given model.

    For each 512x512 input, we:
    1. Extract 9 overlapping 256x256 crops for center 384x384 reconstruction
    2. Extract 4 corner 256x256 patches for outer border
    3. Run inference on all crops (with AMP to match training)
    4. Reconstruct center 384x384 and combine with corner predictions for composite
    """
    model.eval()
    total_patients = len(test_dataset)
    logger.debug(f"Saving visualizations for {total_patients} patients")

    with torch.no_grad(), autocast(device_type="cuda"):
        for i, (all_inputs, all_targets) in enumerate(test_dataset):
            # all_inputs: (N, 2, 512, 512), all_targets: (N, 1, 512, 512)
            center_outputs = []
            corner_outputs = []
            composite_outputs = []

            for sample_idx in range(len(all_inputs)):
                input_image = all_inputs[sample_idx]  # (2, 512, 512)

                # Extract crops from this sample's input: (9, 2, 256, 256)
                input_crops = extract_crops(input_image)

                # Extract corner patches: (4, 2, 256, 256)
                input_corners = extract_corner_patches(input_image)

                # Run inference on center crops in batches
                crop_outputs = []
                for j in range(0, len(input_crops), batch_size):
                    crops_batch = input_crops[j : j + batch_size].to(device)
                    batch_output = model(crops_batch).float().cpu()
                    crop_outputs.append(batch_output)
                    del crops_batch

                # Concatenate all crop predictions: (9, 1, 256, 256)
                crop_outputs = torch.cat(crop_outputs, dim=0)

                # Reconstruct center 384x384 (with black padding): (1, 512, 512)
                center_reconstructed = reconstruct_from_crops(crop_outputs)
                center_outputs.append(center_reconstructed)

                # Run inference on corner patches
                corner_preds = []
                for j in range(0, len(input_corners), batch_size):
                    corners_batch = input_corners[j : j + batch_size].to(device)
                    batch_output = model(corners_batch).float().cpu()
                    corner_preds.append(batch_output)
                    del corners_batch

                # Concatenate all corner predictions: (4, 1, 256, 256)
                corner_preds = torch.cat(corner_preds, dim=0)

                # Combine corner predictions into 512x512: (1, 512, 512)
                corner_combined = combine_corner_predictions(corner_preds)
                corner_outputs.append(corner_combined)

                # Create composite: corners as base, center 384x384 overlaid
                composite = corner_combined.clone()
                composite[
                    :,
                    CENTER_OFFSET : CENTER_OFFSET + CENTER_REGION_SIZE,
                    CENTER_OFFSET : CENTER_OFFSET + CENTER_REGION_SIZE,
                ] = center_reconstructed[
                    :,
                    CENTER_OFFSET : CENTER_OFFSET + CENTER_REGION_SIZE,
                    CENTER_OFFSET : CENTER_OFFSET + CENTER_REGION_SIZE,
                ]
                composite_outputs.append(composite)

            # Stack all outputs: (N, 1, 512, 512)
            center_preds = torch.stack(center_outputs, dim=0)
            corner_preds = torch.stack(corner_outputs, dim=0)
            composite_preds = torch.stack(composite_outputs, dim=0)

            test_patient_save_dir = save_dir / test_dataset.get_patient_id_by_index(idx=i)
            test_patient_save_dir.mkdir(parents=True, exist_ok=True)
            save_prediction_visualization(
                inputs=all_inputs,
                targets=all_targets,
                center_predictions=center_preds,
                corner_predictions=corner_preds,
                composite_predictions=composite_preds,
                save_dir=test_patient_save_dir,
            )

            logger.debug(f"Saved visualization {i + 1}/{total_patients}")

            # Clear memory between patients
            del all_inputs, all_targets, center_preds, corner_preds, composite_preds
            del center_outputs, corner_outputs, composite_outputs
            torch.cuda.empty_cache()
