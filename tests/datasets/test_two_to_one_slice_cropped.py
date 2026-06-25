"""Tests for TwoToOneSliceCroppedDataset class."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from phd.datasets.interpolation.two_to_one_slice_cropped import (
    CENTER_OFFSET,
    CROP_POSITIONS,
    CROP_SIZE,
    DEFAULT_CROP_WEIGHTS,
    NUM_CROPS,
    TwoToOneSliceCroppedDataset,
    TwoToOneSliceTestCroppedDataset,
    reconstruct_from_crops,
)


@pytest.fixture
def mock_dataset_init() -> Generator[MagicMock]:
    """Fixture that mocks the parent class init."""
    with patch(
        "phd.datasets.interpolation.two_to_one_slice.TwoToOneSliceDataset.__init__",
    ) as mock_super_init:
        mock_super_init.return_value = None
        yield mock_super_init


@pytest.fixture
def cropped_dataset(mock_dataset_init: MagicMock) -> TwoToOneSliceCroppedDataset:
    """Fixture that creates a TwoToOneSliceCroppedDataset with mocked parent."""
    root_dir = "/tmp/test_dataset"
    dataset = TwoToOneSliceCroppedDataset(
        root_dir=root_dir,
        stage="train",
        size=100,
        log_info=False,
    )
    dataset.root_dir = Path(root_dir)
    dataset.img_dir = Path(root_dir)
    dataset.rng = MagicMock(spec=np.random.Generator)
    dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))
    return dataset


class TestTwoToOneSliceCroppedDataset:
    """Test the TwoToOneSliceCroppedDataset class."""

    def test_augmentation_consistency_across_triplet(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Verify same augmentation applied to all three images in triplet."""
        with patch("PIL.Image.open") as mock_open:
            # Create test images with unique patterns
            img1_arr = np.full((512, 512), 100, dtype=np.uint8)
            img2_arr = np.full((512, 512), 150, dtype=np.uint8)
            img3_arr = np.full((512, 512), 200, dtype=np.uint8)

            mock_open.return_value.__enter__.side_effect = [
                Image.fromarray(img1_arr),
                Image.fromarray(img2_arr),
                Image.fromarray(img3_arr),
            ]

            # Set augmentation: crop position 0, h-flip yes, v-flip no, no rotation
            cropped_dataset.rng.choice.return_value = 0  # Crop position 0
            # random() called for: rotation check (0.9 = no rotation), h_flip (0.2 = yes), v_flip (0.8 = no)
            cropped_dataset.rng.random.side_effect = [0.9, 0.2, 0.8]

            input_tensor, target_tensor = cropped_dataset[0]

            # Verify shapes
            assert input_tensor.shape == (2, 256, 256)
            assert target_tensor.shape == (1, 256, 256)

            # Verify values are consistent (all from same crop position and flipped same way)
            # First slice should have value ~100/255
            assert input_tensor[0].mean().item() == pytest.approx(100 / 255, abs=0.01)
            # Target should have value ~150/255
            assert target_tensor[0].mean().item() == pytest.approx(150 / 255, abs=0.01)
            # Third slice should have value ~200/255
            assert input_tensor[1].mean().item() == pytest.approx(200 / 255, abs=0.01)

    def test_crop_position_correctness(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Verify crops are extracted from correct positions."""
        with patch("PIL.Image.open") as mock_open:
            # Create gradient image where value = x coordinate (mod 256 for uint8)
            # Use values that won't overflow uint8
            gradient = np.tile(np.arange(256, dtype=np.uint8), (512, 2))[:, :512]

            mock_open.return_value.__enter__.return_value = Image.fromarray(gradient)

            # Need triplets for the dataset to work with deterministic indexing
            cropped_dataset.triplets = {
                0: [("a", "b", "c")],
                1: [("d", "e", "f")],
                2: [("g", "h", "i")],
                3: [("j", "k", "l")],
            }

            # Force crop position 4 (center): (y=128, x=128)
            cropped_dataset.rng.choice.return_value = 4  # Force center crop
            cropped_dataset.rng.random.return_value = 0.9  # No flips

            input_tensor, _ = cropped_dataset[0]

            # Verify crop shape is correct
            assert input_tensor.shape == (2, 256, 256)

            # CROP_POSITIONS[4] = (128, 128), so first pixel x-coordinate is 128
            # First pixel should have value = 128/255 (x coordinate of crop start)
            expected_start = 128 / 255.0
            actual_start = input_tensor[0, 0, 0].item()
            assert actual_start == pytest.approx(expected_start, abs=0.001)

            # Last column (x=128+255=383, but 383%256=127 in the tiled pattern)
            expected_end = 127 / 255.0
            actual_end = input_tensor[0, 0, -1].item()
            assert actual_end == pytest.approx(expected_end, abs=0.001)

    def test_resize_option_512(self, mock_dataset_init: MagicMock) -> None:
        """Verify 512→256 resize option works correctly."""
        with patch("PIL.Image.open") as mock_open:
            root_dir = "/tmp/test_dataset"
            dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="train",
                size=100,
                log_info=False,
                include_resize_512=True,
            )
            dataset.root_dir = Path(root_dir)
            dataset.img_dir = Path(root_dir)
            dataset.rng = MagicMock(spec=np.random.Generator)
            dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            # Create image with known pattern
            test_img = np.full((512, 512), 128, dtype=np.uint8)
            mock_open.return_value.__enter__.return_value = Image.fromarray(test_img)

            # Force resize option (index 9 with include_resize_512=True, no include_resize_384)
            dataset.rng.choice.return_value = 9  # First resize option = 512→256
            dataset.rng.random.return_value = 0.9  # No flips, no rotation

            input_tensor, target_tensor = dataset[0]

            # Verify shape is 256x256
            assert input_tensor.shape == (2, 256, 256)
            assert target_tensor.shape == (1, 256, 256)

            # Verify values are preserved (128/255)
            assert input_tensor.mean().item() == pytest.approx(128 / 255, abs=0.01)

    def test_resize_option_384(self, mock_dataset_init: MagicMock) -> None:
        """Verify 384→256 resize option works correctly."""
        with patch("PIL.Image.open") as mock_open:
            root_dir = "/tmp/test_dataset"
            dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="train",
                size=100,
                log_info=False,
                include_resize_384=True,
                include_resize_512=False,
            )
            dataset.root_dir = Path(root_dir)
            dataset.img_dir = Path(root_dir)
            dataset.rng = MagicMock(spec=np.random.Generator)
            dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            # Create image with known pattern
            test_img = np.full((512, 512), 150, dtype=np.uint8)
            mock_open.return_value.__enter__.return_value = Image.fromarray(test_img)

            # Force resize option (index 9 = 384→256 when include_resize_384=True)
            dataset.rng.choice.return_value = 9  # First resize option = 384→256
            dataset.rng.random.return_value = 0.9  # No flips, no rotation

            input_tensor, target_tensor = dataset[0]

            # Verify shape is 256x256
            assert input_tensor.shape == (2, 256, 256)
            assert target_tensor.shape == (1, 256, 256)

            # Verify values are preserved (150/255)
            assert input_tensor.mean().item() == pytest.approx(150 / 255, abs=0.01)

    def test_horizontal_flip(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Verify horizontal flip works correctly."""
        with patch("PIL.Image.open") as mock_open:
            # Create asymmetric image (left bright, right dark)
            img = np.zeros((512, 512), dtype=np.uint8)
            img[:, :256] = 200  # Left half bright
            img[:, 256:] = 50  # Right half dark

            mock_open.return_value.__enter__.return_value = Image.fromarray(img)

            # Crop center + h-flip
            cropped_dataset.rng.choice.return_value = 0  # Crop position 0
            # random() for: rotation (0.9 = no), h_flip (0.1 = yes), v_flip (0.9 = no)
            cropped_dataset.rng.random.side_effect = [0.9, 0.1, 0.9]

            input_tensor, _ = cropped_dataset[0]

            # After h-flip, left should be dark and right should be bright
            left_mean = input_tensor[0, :, :128].mean().item()
            right_mean = input_tensor[0, :, 128:].mean().item()

            # Left should be darker than right after flip
            assert left_mean < right_mean

    def test_vertical_flip(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Verify vertical flip works correctly."""
        with patch("PIL.Image.open") as mock_open:
            # Create asymmetric image (top bright, bottom dark)
            img = np.zeros((512, 512), dtype=np.uint8)
            img[:256, :] = 200  # Top half bright
            img[256:, :] = 50  # Bottom half dark

            mock_open.return_value.__enter__.return_value = Image.fromarray(img)

            # Crop center + v-flip
            cropped_dataset.rng.choice.return_value = 0  # Crop position 0
            # random() for: rotation (0.9 = no), h_flip (0.9 = no), v_flip (0.1 = yes)
            cropped_dataset.rng.random.side_effect = [0.9, 0.9, 0.1]

            input_tensor, _ = cropped_dataset[0]

            # After v-flip, top should be dark and bottom should be bright
            top_mean = input_tensor[0, :128, :].mean().item()
            bottom_mean = input_tensor[0, 128:, :].mean().item()

            # Top should be darker than bottom after flip
            assert top_mean < bottom_mean

    def test_validation_mode_no_augmentation(self, mock_dataset_init: MagicMock) -> None:
        """Verify validation mode uses full images without augmentation."""
        with patch("PIL.Image.open") as mock_open:
            root_dir = "/tmp/test_dataset"
            valid_dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="valid",
                size=100,
                log_info=False,
            )
            valid_dataset.root_dir = Path(root_dir)
            valid_dataset.img_dir = Path(root_dir)
            valid_dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            img = Image.new("L", (512, 512), color=128)
            mock_open.return_value.__enter__.return_value = img

            input_tensor, target_tensor = valid_dataset[0]

            # Should be full 512x512 images
            assert input_tensor.shape == (2, 512, 512)
            assert target_tensor.shape == (1, 512, 512)

    def test_all_crop_positions_valid(self) -> None:
        """Verify all 9 crop positions are within bounds."""
        for y, x in CROP_POSITIONS:
            assert y >= 0
            assert x >= 0
            assert y + CROP_SIZE <= 512
            assert x + CROP_SIZE <= 512

    def test_apply_spatial_augmentation_resize_center(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Test _apply_spatial_augmentation with center resize (aug_idx=9)."""
        # Create a 512x512 tensor
        img_tensor = torch.full((1, 512, 512), 0.5)

        # Apply center resize augmentation (aug_idx=9)
        result = cropped_dataset._apply_spatial_augmentation(img_tensor, aug_idx=9)

        # Should be resized to 256x256
        assert result.shape == (1, 256, 256)
        # Values should be preserved (approximately)
        assert result.mean().item() == pytest.approx(0.5, abs=0.01)

    def test_apply_spatial_augmentation_resize_full(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Test _apply_spatial_augmentation with full resize (aug_idx=10)."""
        # Create a 512x512 tensor
        img_tensor = torch.full((1, 512, 512), 0.6)

        # Apply full resize augmentation (aug_idx=10)
        result = cropped_dataset._apply_spatial_augmentation(img_tensor, aug_idx=10)

        # Should be resized to 256x256
        assert result.shape == (1, 256, 256)
        # Values should be preserved (approximately)
        assert result.mean().item() == pytest.approx(0.6, abs=0.01)

    def test_apply_spatial_augmentation_crop(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Test _apply_spatial_augmentation with crop."""
        # Create a 512x512 tensor with gradient
        img_tensor = torch.zeros((1, 512, 512))
        img_tensor[0, :, :256] = 1.0  # Left half is 1, right half is 0

        # Apply crop at position 0 (center: y=128, x=128)
        result = cropped_dataset._apply_spatial_augmentation(img_tensor, aug_idx=0)

        # Should be cropped to 256x256
        assert result.shape == (1, 256, 256)

    def test_return_all_augmentations(self, mock_dataset_init: MagicMock) -> None:
        """Test validation mode with return_all_augmentations=True."""
        with patch("PIL.Image.open") as mock_open:
            root_dir = "/tmp/test_dataset"
            valid_dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="valid",
                size=100,
                log_info=False,
                return_all_augmentations=True,
            )
            valid_dataset.root_dir = Path(root_dir)
            valid_dataset.img_dir = Path(root_dir)
            valid_dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            img = Image.new("L", (512, 512), color=128)
            mock_open.return_value.__enter__.return_value = img

            result = valid_dataset[0]

            # Should return 2 items: aug_inputs, aug_targets (all 256x256)
            assert len(result) == 2
            aug_inputs, aug_targets = result

            # Augmented tensors: 9 crops at 256x256 (no resizes by default)
            assert aug_inputs.shape == (9, 2, 256, 256)
            assert aug_targets.shape == (9, 1, 256, 256)

    def test_get_all_augmentations(self, cropped_dataset: TwoToOneSliceCroppedDataset) -> None:
        """Test _get_all_augmentations method directly."""
        # Create test tensors
        first_img = torch.full((1, 512, 512), 0.3)
        second_img = torch.full((1, 512, 512), 0.5)
        third_img = torch.full((1, 512, 512), 0.7)

        aug_inputs, aug_targets = cropped_dataset._get_all_augmentations(first_img, second_img, third_img)

        # Should have 9 augmentations (9 crops, no resizes by default)
        assert aug_inputs.shape == (9, 2, 256, 256)
        assert aug_targets.shape == (9, 1, 256, 256)

        # First channel of inputs should have values from first_img (~0.3)
        assert aug_inputs[0, 0].mean().item() == pytest.approx(0.3, abs=0.01)
        # Second channel should have values from third_img (~0.7)
        assert aug_inputs[0, 1].mean().item() == pytest.approx(0.7, abs=0.01)
        # Targets should have values from second_img (~0.5)
        assert aug_targets[0, 0].mean().item() == pytest.approx(0.5, abs=0.01)


class TestTwoToOneSliceTestCroppedDataset:
    """Test the TwoToOneSliceTestCroppedDataset class."""

    def test_test_dataset_forces_test_stage(self, mock_dataset_init: MagicMock) -> None:
        """Test that TwoToOneSliceTestCroppedDataset forces stage to 'test'."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceTestCroppedDataset(
            root_dir=root_dir,
            stage="train",  # Try to use train stage
            size=100,
            log_info=False,
        )

        # use_augmentations should be False because stage is forced to "test"
        assert dataset.use_augmentations is False

    def test_test_dataset_no_augmentations(self, mock_dataset_init: MagicMock) -> None:
        """Test that TwoToOneSliceTestCroppedDataset disables augmentations."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceTestCroppedDataset(
            root_dir=root_dir,
            stage="valid",
            size=100,
            log_info=False,
        )

        # _get_augmentation_params should return None
        assert dataset._get_augmentation_params() is None


class TestReconstructFromCrops:
    """Test the reconstruct_from_crops function."""

    def test_output_shape(self) -> None:
        """Test that output is 512×512."""
        crops = torch.zeros(NUM_CROPS, 1, CROP_SIZE, CROP_SIZE)
        output = reconstruct_from_crops(crops)
        assert output.shape == (1, 512, 512)

    def test_output_shape_multichannel(self) -> None:
        """Test that output preserves channel count."""
        crops = torch.zeros(NUM_CROPS, 3, CROP_SIZE, CROP_SIZE)
        output = reconstruct_from_crops(crops)
        assert output.shape == (3, 512, 512)

    def test_wrong_num_crops_raises(self) -> None:
        """Test that wrong number of crops raises ValueError."""
        crops = torch.zeros(5, 1, CROP_SIZE, CROP_SIZE)  # Wrong: should be 9
        with pytest.raises(ValueError, match="Expected 9 crops"):
            reconstruct_from_crops(crops)

    def test_border_is_zero(self) -> None:
        """Test that outer 64px border is zero (black)."""
        # Fill crops with non-zero values
        crops = torch.ones(NUM_CROPS, 1, CROP_SIZE, CROP_SIZE)
        output = reconstruct_from_crops(crops)

        # Top border (rows 0-63)
        assert output[:, :CENTER_OFFSET, :].sum() == 0
        # Bottom border (rows 448-511)
        assert output[:, 512 - CENTER_OFFSET :, :].sum() == 0
        # Left border (cols 0-63, excluding corners already checked)
        assert output[:, CENTER_OFFSET : 512 - CENTER_OFFSET, :CENTER_OFFSET].sum() == 0
        # Right border (cols 448-511)
        assert output[:, CENTER_OFFSET : 512 - CENTER_OFFSET, 512 - CENTER_OFFSET :].sum() == 0

    def test_center_region_filled(self) -> None:
        """Test that center 384×384 region is filled (non-zero when crops are non-zero)."""
        crops = torch.ones(NUM_CROPS, 1, CROP_SIZE, CROP_SIZE)
        output = reconstruct_from_crops(crops)

        # Center region should be non-zero
        center = output[:, CENTER_OFFSET : 512 - CENTER_OFFSET, CENTER_OFFSET : 512 - CENTER_OFFSET]
        assert center.shape == (1, 384, 384)
        assert (center > 0).all()

    def test_uniform_crops_give_uniform_center(self) -> None:
        """Test that uniform crop values give uniform center region."""
        value = 0.5
        crops = torch.full((NUM_CROPS, 1, CROP_SIZE, CROP_SIZE), value)
        output = reconstruct_from_crops(crops)

        # Center region should all be the same value (averaging identical values)
        center = output[:, CENTER_OFFSET : 512 - CENTER_OFFSET, CENTER_OFFSET : 512 - CENTER_OFFSET]
        assert torch.allclose(center, torch.full_like(center, value))

    def test_overlap_averaging(self) -> None:
        """Test that overlapping regions are averaged correctly."""
        # Create crops where each crop has a unique value
        crops = torch.zeros(NUM_CROPS, 1, CROP_SIZE, CROP_SIZE)
        for i in range(NUM_CROPS):
            crops[i] = float(i + 1)  # Values 1-9

        output = reconstruct_from_crops(crops)

        # The very center pixel (256, 256) should be covered by all 9 crops
        # since all crops are 256×256 and positioned in a 3×3 grid
        # Actually, let's check a pixel that we know the overlap count for
        # Center of the grid at (256, 256) in 512×512 space
        # This is at (192, 192) in the center 384×384 space
        # Crop 4 (center) covers (128, 128) to (384, 384) - includes (256, 256)
        # We need to calculate which crops cover pixel (256, 256)

        # Pixel (256, 256) is covered by crops at positions where:
        # orig_y <= 256 < orig_y + 256 AND orig_x <= 256 < orig_x + 256
        # Positions: (64,64), (64,128), (64,192), (128,64), (128,128), (128,192)
        # That's crops 0,1,2,3,4,5 (6 crops)
        # Wait, let me recalculate based on CROP_POSITIONS

        # CROP_POSITIONS = [(64,64), (64,128), (64,192), (128,64), (128,128), (128,192), (192,64), (192,128), (192,192)]
        # For pixel (256, 256):
        # - Crop 0 (64,64): 64 <= 256 < 320 AND 64 <= 256 < 320 -> YES
        # - Crop 1 (64,128): 64 <= 256 < 320 AND 128 <= 256 < 384 -> YES
        # - Crop 2 (64,192): 64 <= 256 < 320 AND 192 <= 256 < 448 -> YES
        # - Crop 3 (128,64): 128 <= 256 < 384 AND 64 <= 256 < 320 -> YES
        # - Crop 4 (128,128): 128 <= 256 < 384 AND 128 <= 256 < 384 -> YES
        # - Crop 5 (128,192): 128 <= 256 < 384 AND 192 <= 256 < 448 -> YES
        # - Crop 6 (192,64): 192 <= 256 < 448 AND 64 <= 256 < 320 -> YES
        # - Crop 7 (192,128): 192 <= 256 < 448 AND 128 <= 256 < 384 -> YES
        # - Crop 8 (192,192): 192 <= 256 < 448 AND 192 <= 256 < 448 -> YES
        # All 9 crops cover (256, 256)!

        # So pixel (256, 256) should be average of 1+2+3+4+5+6+7+8+9 = 45, / 9 = 5.0
        expected_center_value = sum(range(1, 10)) / 9  # 5.0
        actual_value = output[0, 256, 256].item()
        assert actual_value == pytest.approx(expected_center_value, abs=1e-5)

    def test_preserves_device_and_dtype(self) -> None:
        """Test that output preserves input device and dtype."""
        crops = torch.zeros(NUM_CROPS, 1, CROP_SIZE, CROP_SIZE, dtype=torch.float16)
        output = reconstruct_from_crops(crops)
        assert output.dtype == torch.float16


class TestRandomCropSelection:
    """Test random crop selection for training datasets."""

    def test_training_len_no_expansion(self, mock_dataset_init: MagicMock) -> None:
        """Test that training __len__ returns base length (no expansion)."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            size=100,
            log_info=False,
        )

        # 40 base samples (10 per class * 4 classes)
        dataset.triplets = {
            0: [("a", "b", "c")] * 10,
            1: [("d", "e", "f")] * 10,
            2: [("g", "h", "i")] * 10,
            3: [("j", "k", "l")] * 10,
        }

        # Training mode should return base length (no expansion)
        assert len(dataset) == 40

    def test_validation_len_same_as_training(self, mock_dataset_init: MagicMock) -> None:
        """Test that validation __len__ also returns base length."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="valid",
            size=100,
            log_info=False,
        )

        # 40 base samples
        dataset.triplets = {
            0: [("a", "b", "c")] * 10,
            1: [("d", "e", "f")] * 10,
            2: [("g", "h", "i")] * 10,
            3: [("j", "k", "l")] * 10,
        }

        # Validation mode should also return base length
        assert len(dataset) == 40

    def test_selection_probabilities_match_weights(self, mock_dataset_init: MagicMock) -> None:
        """Verify selection probabilities match configured weights."""
        weights = (1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0)
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            size=100,
            log_info=False,
            crop_weights=weights,
            include_resize_512=False,
        )

        total = sum(weights)
        expected_probs = [w / total for w in weights]
        np.testing.assert_array_almost_equal(dataset.selection_probs, expected_probs, decimal=6)

    def test_selection_probabilities_with_resize_512(self, mock_dataset_init: MagicMock) -> None:
        """Verify selection probabilities include resize_512 weight."""
        weights = (1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0)
        resize_weight = 0.5
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            size=100,
            log_info=False,
            crop_weights=weights,
            include_resize_512=True,
            resize_512_weight=resize_weight,
        )

        total = sum(weights) + resize_weight
        expected_probs = [w / total for w in weights] + [resize_weight / total]
        np.testing.assert_array_almost_equal(dataset.selection_probs, expected_probs, decimal=6)

    def test_num_options_with_resizes(self, mock_dataset_init: MagicMock) -> None:
        """Test num_options increases with resize options."""
        root_dir = "/tmp/test_dataset"

        # No resizes
        dataset1 = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
            include_resize_512=False,
            include_resize_384=False,
        )
        assert dataset1.num_options == 9

        # Only 512 resize
        dataset2 = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
            include_resize_512=True,
            include_resize_384=False,
        )
        assert dataset2.num_options == 10

        # Both resizes
        dataset3 = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
            include_resize_512=True,
            include_resize_384=True,
        )
        assert dataset3.num_options == 11


class TestDatasetSizeLimiting:
    """Test that size parameter correctly limits dataset length."""

    def test_training_dataset_len_base(self, mock_dataset_init: MagicMock) -> None:
        """Test that training __len__ returns base length (no expansion)."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            size=100,
            log_info=False,
        )

        # Manually set triplets to simulate size=40 (10 per class * 4 classes)
        dataset.triplets = {
            0: [("a", "b", "c")] * 10,
            1: [("d", "e", "f")] * 10,
            2: [("g", "h", "i")] * 10,
            3: [("j", "k", "l")] * 10,
        }

        # Training mode: 40 base (no expansion with random selection)
        assert len(dataset) == 40

    def test_validation_dataset_len_base(self, mock_dataset_init: MagicMock) -> None:
        """Test that validation __len__ returns base length."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="valid",
            size=100,
            log_info=False,
        )

        # Manually set triplets to simulate size=40 (10 per class * 4 classes)
        dataset.triplets = {
            0: [("a", "b", "c")] * 10,
            1: [("d", "e", "f")] * 10,
            2: [("g", "h", "i")] * 10,
            3: [("j", "k", "l")] * 10,
        }

        # Validation mode: 40 base
        assert len(dataset) == 40

    def test_dataset_len_with_different_sizes(self, mock_dataset_init: MagicMock) -> None:
        """Test __len__ with various triplet configurations."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            size=100,
            log_info=False,
        )

        # Test with base=80 (20 per class)
        dataset.triplets = {
            0: [("a", "b", "c")] * 20,
            1: [("d", "e", "f")] * 20,
            2: [("g", "h", "i")] * 20,
            3: [("j", "k", "l")] * 20,
        }
        # Training mode: 80 base
        assert len(dataset) == 80

        # Test with base=12 (3 per class)
        dataset.triplets = {
            0: [("a", "b", "c")] * 3,
            1: [("d", "e", "f")] * 3,
            2: [("g", "h", "i")] * 3,
            3: [("j", "k", "l")] * 3,
        }
        # Training mode: 12 base
        assert len(dataset) == 12


class TestDataLoaderBatchShapes:
    """Test that DataLoader produces correct batch shapes."""

    def test_train_dataloader_batch_shapes(self, mock_dataset_init: MagicMock) -> None:
        """Test that training DataLoader produces correct batch shapes."""
        with patch("PIL.Image.open") as mock_open:
            from torch.utils.data import DataLoader

            root_dir = "/tmp/test_dataset"
            dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="train",
                size=100,
                log_info=False,
            )
            dataset.root_dir = Path(root_dir)
            dataset.img_dir = Path(root_dir)
            dataset.rng = MagicMock(spec=np.random.Generator)
            dataset.rng.integers.return_value = 0  # Center crop
            dataset.rng.random.return_value = 0.9  # No flips

            # Set up triplets for 16 samples (4 per class)
            dataset.triplets = {
                0: [("a", "b", "c")] * 4,
                1: [("d", "e", "f")] * 4,
                2: [("g", "h", "i")] * 4,
                3: [("j", "k", "l")] * 4,
            }
            dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            # Mock image loading
            img = Image.new("L", (512, 512), color=128)
            mock_open.return_value.__enter__.return_value = img

            # Create DataLoader with batch_size=8
            loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)

            # Get first batch
            inputs, targets = next(iter(loader))

            # Verify shapes: (batch, channels, height, width)
            assert inputs.shape == (8, 2, 256, 256), f"Expected (8, 2, 256, 256), got {inputs.shape}"
            assert targets.shape == (8, 1, 256, 256), f"Expected (8, 1, 256, 256), got {targets.shape}"

    def test_valid_dataloader_batch_shapes_with_all_augmentations(self, mock_dataset_init: MagicMock) -> None:
        """Test that validation DataLoader with all augmentations produces correct shapes."""
        with patch("PIL.Image.open") as mock_open:
            from torch.utils.data import DataLoader

            root_dir = "/tmp/test_dataset"
            dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="valid",
                size=100,
                log_info=False,
                return_all_augmentations=True,
            )
            dataset.root_dir = Path(root_dir)
            dataset.img_dir = Path(root_dir)

            # Set up triplets for 8 samples (2 per class)
            dataset.triplets = {
                0: [("a", "b", "c")] * 2,
                1: [("d", "e", "f")] * 2,
                2: [("g", "h", "i")] * 2,
                3: [("j", "k", "l")] * 2,
            }
            dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            # Mock image loading
            img = Image.new("L", (512, 512), color=128)
            mock_open.return_value.__enter__.return_value = img

            # Create DataLoader with batch_size=4
            loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)

            # Get first batch
            aug_inputs, aug_targets = next(iter(loader))

            # With return_all_augmentations=True, each sample returns 9 augmentations
            # Shape: (batch, num_augs, channels, height, width)
            assert aug_inputs.shape == (4, 9, 2, 256, 256), f"Expected (4, 9, 2, 256, 256), got {aug_inputs.shape}"
            assert aug_targets.shape == (4, 9, 1, 256, 256), f"Expected (4, 9, 1, 256, 256), got {aug_targets.shape}"

    def test_batch_values_in_valid_range(self, mock_dataset_init: MagicMock) -> None:
        """Test that batch values are normalized to [0, 1] range."""
        with patch("PIL.Image.open") as mock_open:
            from torch.utils.data import DataLoader

            root_dir = "/tmp/test_dataset"
            dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="train",
                size=100,
                log_info=False,
            )
            dataset.root_dir = Path(root_dir)
            dataset.img_dir = Path(root_dir)
            dataset.rng = MagicMock(spec=np.random.Generator)
            dataset.rng.integers.return_value = 0
            dataset.rng.random.return_value = 0.9

            dataset.triplets = {
                0: [("a", "b", "c")] * 4,
                1: [("d", "e", "f")] * 4,
                2: [("g", "h", "i")] * 4,
                3: [("j", "k", "l")] * 4,
            }
            dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            # Create image with known values
            img = Image.new("L", (512, 512), color=128)
            mock_open.return_value.__enter__.return_value = img

            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            inputs, targets = next(iter(loader))

            # Values should be in [0, 1] range
            assert inputs.min() >= 0.0, f"Min value {inputs.min()} is below 0"
            assert inputs.max() <= 1.0, f"Max value {inputs.max()} is above 1"
            assert targets.min() >= 0.0, f"Min value {targets.min()} is below 0"
            assert targets.max() <= 1.0, f"Max value {targets.max()} is above 1"

            # Check expected value (128/255 ≈ 0.502)
            expected_value = 128 / 255.0
            assert abs(inputs.mean().item() - expected_value) < 0.01


class TestWeightedRandomCropSelection:
    """Test weighted random crop selection for training."""

    def test_center_crop_double_probability(self, mock_dataset_init: MagicMock) -> None:
        """Center (position 4) should be selected ~2x more than others."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            size=100,
            log_info=False,
            crop_weights=(1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0),
            include_resize_512=True,
            resize_512_weight=0.5,
        )

        # Set up a real numpy RNG for statistical testing
        dataset.rng = np.random.default_rng(42)

        counts = [0] * 10  # 9 crops + 1 resize
        for _ in range(10000):
            params = dataset._get_augmentation_params()
            counts[params["aug_idx"]] += 1

        # With weights [1,1,1,1,2,1,1,1,1,0.5], total=10.5
        # Center (4): 2/10.5 ≈ 19%
        # Others (0-3,5-8): 1/10.5 ≈ 9.5% each
        # Resize (9): 0.5/10.5 ≈ 4.8%
        other_crops = [counts[i] for i in range(9) if i != 4]
        avg_other = sum(other_crops) / 8
        center_ratio = counts[4] / avg_other
        assert center_ratio == pytest.approx(2.0, abs=0.2)

        resize_ratio = counts[9] / avg_other
        assert resize_ratio == pytest.approx(0.5, abs=0.15)

    def test_default_crop_weights(self, mock_dataset_init: MagicMock) -> None:
        """Test that default crop weights give center 2x weight."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
        )

        # Default weights should be (1,1,1,1,2,1,1,1,1)
        assert dataset.crop_weights == DEFAULT_CROP_WEIGHTS
        assert dataset.crop_weights[4] == 2.0


class TestRotationAugmentation:
    """Test rotation augmentation for training."""

    def test_rotation_probability(self, mock_dataset_init: MagicMock) -> None:
        """Rotation should be applied according to rotation_prob."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
            rotation_prob=0.5,
        )

        # Set up a real numpy RNG for statistical testing
        dataset.rng = np.random.default_rng(42)

        rotated_count = 0
        for _ in range(1000):
            params = dataset._get_augmentation_params()
            if params["rotation_angle"] != 0.0:
                rotated_count += 1

        # Should be ~50% rotated
        assert rotated_count / 1000 == pytest.approx(0.5, abs=0.05)

    def test_rotation_angle_range(self, mock_dataset_init: MagicMock) -> None:
        """Rotation angles should be in [-max, +max]."""
        max_angle = 15.0
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
            rotation_prob=1.0,
            rotation_max_angle=max_angle,
        )

        dataset.rng = np.random.default_rng(42)

        for _ in range(100):
            params = dataset._get_augmentation_params()
            angle = params["rotation_angle"]
            assert angle >= -max_angle
            assert angle <= max_angle

    def test_rotation_uniform_distribution(self, mock_dataset_init: MagicMock) -> None:
        """Non-zero rotations should be uniformly distributed."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
            rotation_prob=1.0,
            rotation_max_angle=15.0,
        )

        dataset.rng = np.random.default_rng(42)

        angles = []
        for _ in range(1000):
            params = dataset._get_augmentation_params()
            angles.append(params["rotation_angle"])

        # Check roughly uniform: mean should be ~0, std should be ~8.66 (15/sqrt(3))
        assert np.mean(angles) == pytest.approx(0.0, abs=1.0)
        assert np.std(angles) == pytest.approx(15 / np.sqrt(3), abs=1.0)

    def test_validation_no_rotation(self, mock_dataset_init: MagicMock) -> None:
        """Validation should never apply rotation."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="valid",
            log_info=False,
            rotation_prob=1.0,  # Would apply 100% in training
        )

        # Rotation prob should be forced to 0 for validation
        assert dataset.rotation_prob == 0.0

    def test_rotation_disabled_when_prob_zero(self, mock_dataset_init: MagicMock) -> None:
        """No rotation when rotation_prob=0."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="train",
            log_info=False,
            rotation_prob=0.0,
        )

        dataset.rng = np.random.default_rng(42)

        for _ in range(100):
            params = dataset._get_augmentation_params()
            assert params["rotation_angle"] == 0.0


class TestValidation9Crops:
    """Test that validation always returns exactly 9 crops."""

    def test_validation_returns_exactly_9_crops(self, mock_dataset_init: MagicMock) -> None:
        """Validation should return 9 crops, never resizes."""
        with patch("PIL.Image.open") as mock_open:
            root_dir = "/tmp/test_dataset"
            dataset = TwoToOneSliceCroppedDataset(
                root_dir=root_dir,
                stage="valid",
                log_info=False,
                return_all_augmentations=True,
                include_resize_512=True,  # Should be ignored for validation
                include_resize_384=True,  # Should be ignored for validation
            )
            dataset.root_dir = Path(root_dir)
            dataset.img_dir = Path(root_dir)
            dataset._get_triplet_ids = MagicMock(return_value=("img1", "img2", "img3"))

            img = Image.new("L", (512, 512), color=128)
            mock_open.return_value.__enter__.return_value = img

            aug_inputs, aug_targets = dataset[0]

            # Should return exactly 9 crops, not 10 or 11
            assert aug_inputs.shape[0] == 9
            assert aug_targets.shape[0] == 9

    def test_resize_options_disabled_for_validation(self, mock_dataset_init: MagicMock) -> None:
        """Resize options should be disabled in validation mode."""
        root_dir = "/tmp/test_dataset"
        dataset = TwoToOneSliceCroppedDataset(
            root_dir=root_dir,
            stage="valid",
            log_info=False,
            include_resize_512=True,  # Should be ignored
            include_resize_384=True,  # Should be ignored
        )

        # Resize should be forced to False for validation
        assert dataset.include_resize_512 is False
        assert dataset.include_resize_384 is False
