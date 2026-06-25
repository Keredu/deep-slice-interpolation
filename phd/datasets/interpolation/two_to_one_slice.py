"""Dataset classes for CT slice interpolation training and testing."""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Standard transforms available for import
STANDARD_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class TwoToOneSliceDataset(Dataset):
    """Dataset for converting 2 consecutive CT slices to 1 slice."""

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
    ) -> None:
        """Dataset for converting 2 consecutive CT slices to 1 slice.

        Args:
            root_dir: Root directory containing the image files
            stage: Either 'train' or 'valid' to specify which stage to use
            size: Total number of triplets to use (it must be divisible by 4)
            transform: Optional transform to be applied on both input and target images
            seed: Random seed for reproducibility
            original_window: Original window for the images
            target_window: Target window for the images
            log_info: Whether to log dataset stats after initialization

        """
        start_time = time.time()
        self.root_dir = Path(root_dir)
        self.stage = stage
        self.size = size
        self.transform = transform
        self.seed = seed
        self.original_window = original_window
        self.target_window = target_window
        self.should_log_info = log_info

        # Validate input parameters
        self._validate_parameters(
            size=size,
            stage=stage,
            original_window=original_window,
            target_window=target_window,
        )

        # Set random seed for reproducibility
        self.rng = np.random.default_rng(seed=seed)

        # Load and filter dataframe based on stage
        pre_dir = Path(os.getenv("DATASETS_DIR"), "pre/rsna-intracranial-hemorrhage-detection")
        self.img_dir = Path(root_dir)
        df = pd.read_csv(Path(pre_dir, "df.csv"))
        # TODO: Remove this when the dataset is updated
        df = df.rename(columns={"split": "stage"})
        df = df[df["stage"] == stage]
        self.df = df

        # Create and balance triplets
        self.triplets = self._create_triplets()
        self.triplets = self._balance_triplets(size=size)

        # Set transform
        # TODO: Add original window to target window transform
        self.transform = transform
        if self.should_log_info:
            self._log_dataset_info(
                start_time=start_time,
                stage=stage,
            )

    def _validate_parameters(
        self,
        size: int | None,
        stage: str,
        original_window: tuple[int],
        target_window: tuple[int] | None,
    ) -> None:
        """Validate input parameters.

        Args:
            size: Number of samples to use (must be divisible by 4)
            stage: Either 'train' or 'valid' to specify which stage to use
            original_window: Original window range for slice indices
            target_window: Target window range for filtering slices

        Raises:
            ValueError:
                - If size is not divisible by 4
                - If stage is not 'train' or 'valid'
                - If window values are invalid

        """
        if size is not None and size % 4 != 0:
            raise ValueError("Size must be divisible by 4 to ensure balanced sampling")
        if stage not in {"train", "valid"}:
            raise ValueError("stage must be either 'train' or 'valid'")
        if target_window and not (
            original_window[0] < original_window[1]
            and target_window[0] < target_window[1]
            and target_window[0] >= original_window[0]
            and target_window[1] <= original_window[1]
        ):
            raise ValueError("Invalid window values: ensure target window is within original window and both are valid")

    def _create_triplets(self) -> dict:
        """Create triplets of consecutive slices with hemorrhage information.

        Returns:
            dict: Dictionary with keys 0, 1, 2, 3 representing the number of IH slices

        """
        groups = self.df.groupby(["PatientID", "StudyInstanceUID"])
        triplets = {i: [] for i in range(4)}

        for _, group in groups:
            if len(group) < 3:
                logger.warning(f"Group {group['PatientID'].values[0]} has less than 3 slices")
                continue
            sorted_group = group.sort_values(by="order")
            ih_windows = np.lib.stride_tricks.sliding_window_view(sorted_group["any"].values, 3)
            sop_windows = np.lib.stride_tricks.sliding_window_view(sorted_group["SOPInstanceUID"].values, 3)
            n_ih_per_window = np.sum(ih_windows, axis=1)

            for n_ih, sop_window in zip(n_ih_per_window, sop_windows, strict=False):
                triplets[n_ih].append(sop_window)
        logger.debug(f"_create_triplets: { {k: len(v) for k, v in triplets.items()} }")
        return triplets

    def _balance_triplets(self, size: int | None) -> dict:
        """Balance triplets according to size parameter.

        Args:
            size: Total number of triplets to use (must be even, will be split equally between IH and non-IH)

        Returns:
            dict: Dictionary with keys 0, 1, 2, 3 representing the number of IH slices

        Raises:
            ValueError: If size is larger than available data

        """
        # Shuffle triplets
        for v in self.triplets.values():
            self.rng.shuffle(v)

        # Verify size is not larger than available data
        available_size = min(len(v) for v in self.triplets.values()) * 3
        if size is not None and size > available_size:
            msg = f"Requested size {size} is larger than available data size {available_size}"
            raise ValueError(msg)

        # Balance the dataset
        if size is None:
            size_per_class = min(len(v) for v in self.triplets.values())
            logger.warning(
                f"No size specified - using balanced dataset with {size_per_class} samples per class",
            )
        else:
            size_per_class = size // len(self.triplets.keys())
        balanced_triplets = {k: v[:size_per_class] for k, v in self.triplets.items()}
        logger.debug(f"_balance_triplets: { {k: len(v) for k, v in balanced_triplets.items()} }")
        return balanced_triplets

    def _log_dataset_info(self, start_time: float, stage: str) -> None:
        """Log dataset initialization information."""
        elapsed_time = time.time() - start_time
        total_samples = sum(len(v) for v in self.triplets.values())
        class_breakdown = ", ".join(f"{k}: {len(v)}" for k, v in self.triplets.items())
        logger.info(
            f"Dataset initialization took {elapsed_time:.2f} seconds\n"
            f"Stage: {stage}\n"
            f"Dataset size: {total_samples} samples\n"
            f"Triplets per class: ({class_breakdown})\n",
        )

    def __len__(self) -> int:
        """Get the total number of triplets in the dataset.

        Returns:
            int: Total number of triplets in the dataset

        """
        return sum(len(v) for v in self.triplets.values())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a triplet of consecutive slices and convert to input-target pair.

        Args:
            idx: Index of the triplet (even: IH, odd: non-IH)

        Returns:
            tuple: (input_tensor, target_tensor) where:
                - input_tensor has 2 channels: [first_slice, third_slice]
                - target_tensor has 1 channel: second_slice
                Both tensors are in range [0, 1] and float32 dtype

        """
        # Get image paths for the triplet
        first_id, second_id, third_id = self._get_triplet_ids(idx=idx)

        # Load images and convert to numpy arrays using context managers
        with Image.open(self.img_dir / f"{first_id}.png") as img:
            first_img = np.array(img, dtype=np.float32) / 255.0
        with Image.open(self.img_dir / f"{second_id}.png") as img:
            second_img = np.array(img, dtype=np.float32) / 255.0
        with Image.open(self.img_dir / f"{third_id}.png") as img:
            third_img = np.array(img, dtype=np.float32) / 255.0

        # Create input tensor with 2 channels (H,W,C format)
        input_arr = np.stack([first_img, third_img], axis=-1)  # Shape: (H, W, 2)

        # Create target array with channel dimension (H,W,C format)
        target_arr = second_img[..., None]  # Shape: (H, W, 1)

        # Apply transforms
        input_tensor = self.transform(input_arr)
        target_tensor = self.transform(target_arr)

        return input_tensor, target_tensor

    def get_in_channels(self) -> int:
        """Get the number of input channels.

        Returns:
            int: Number of input channels

        """
        return 2

    def get_out_channels(self) -> int:
        """Get the number of output channels.

        Returns:
            int: Number of output channels

        """
        return 1

    def _get_triplet_ids(self, idx: int) -> tuple[str, str, str]:
        """Get the triplet IDs for a given index.

        Args:
            idx: Index of the triplet

        """
        return self.triplets[idx % 4][idx // 4]

    def _get_patient_id(self, idx: int) -> tuple[str, str, str]:
        id0, id1, id2 = self._get_triplet_ids(idx=idx)
        patient_id = self.df[self.df["SOPInstanceUID"].isin([id0, id1, id2])]["PatientID"].unique()
        if patient_id.size != 1:
            msg = f"Expected 1 patient ID, got {patient_id.size} for triplet {id0}, {id1}, {id2}"
            raise ValueError(msg)
        return patient_id[0]


class TwoToOneSliceTestDataset(TwoToOneSliceDataset):
    """Dataset for testing the model."""

    def __init__(
        self,
        root_dir: str,
        stage: str,
        mode: Literal["target_is_real", "target_is_interpolated"],
        transform: Callable = STANDARD_TRANSFORM,
        seed: int = 42,
        original_window: tuple[int] = (-20, 107),
        target_window: tuple[int] | None = None,
    ) -> None:
        """Initialize test dataset for model evaluation.

        Args:
            root_dir: Root directory containing the image files
            stage: Dataset stage ('train', 'valid', or 'test')
            mode: Either 'target_is_real' or 'target_is_interpolated'
            transform: Transform to apply to images
            seed: Random seed for reproducibility
            original_window: Original HU window for the images
            target_window: Target HU window (None to keep original)

        """
        start_time = time.time()
        super().__init__(
            root_dir=root_dir,
            stage=stage,
            size=None,
            transform=transform,
            seed=seed,
            original_window=original_window,
            target_window=target_window,
            log_info=False,
        )

        self.mode = mode
        if mode not in {"target_is_real", "target_is_interpolated"}:
            raise ValueError(f"Invalid mode: {mode}")

        self.patients = sorted(self.df["PatientID"].unique())
        for patient_id in self.patients:
            patient_df = self.df[self.df["PatientID"] == patient_id]
            if patient_df["StudyInstanceUID"].nunique() != 1:
                raise ValueError(
                    f"Patient {patient_id} has multiple studies: {patient_df['StudyInstanceUID'].unique()}",
                )
            if len(patient_df) < 2:
                raise ValueError(
                    f"Patient {patient_id} has less than 2 slices: {len(patient_df)}",
                )

        elapsed_time = time.time() - start_time
        self._log_test_dataset_info(stage=stage, elapsed_time=elapsed_time)

    def _validate_parameters(
        self,
        size: int | None,
        stage: str,
        original_window: tuple[int],
        target_window: tuple[int] | None,
    ) -> None:
        """Validate input parameters.

        Args:
            size: Number of samples (must be None for test stage)
            stage: Only 'test' stage allowed
            original_window: Original window range for slice indices
            target_window: Target window range for filtering slices

        Raises:
            ValueError:
                - If stage is not 'test'

        """
        if stage != "test":
            raise ValueError("stage must be 'test'")
        if target_window and not (
            original_window[0] < original_window[1]
            and target_window[0] < target_window[1]
            and target_window[0] >= original_window[0]
            and target_window[1] <= original_window[1]
        ):
            raise ValueError("Invalid window values: ensure target window is within original window and both are valid")

    def _create_triplets(self) -> dict:
        return {}

    def _balance_triplets(self, size: int | None) -> dict:
        return {}

    def __len__(self) -> int:
        """Return number of patients in the dataset."""
        return len(self.patients)

    def get_patient_id_by_index(self, idx: int) -> str:
        """Get patient ID by index."""
        return self.patients[idx]

    def get_item_target_is_real(self, patient_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Get triplets where target is the real middle slice."""
        first_ids = patient_df["SOPInstanceUID"].iloc[:-2].values
        second_ids = patient_df["SOPInstanceUID"].iloc[1:-1].values
        third_ids = patient_df["SOPInstanceUID"].iloc[2:].values
        input_tensors = []
        target_tensors = []
        for first_id, second_id, third_id in zip(first_ids, second_ids, third_ids, strict=False):
            # Load images and convert to numpy arrays using context managers
            with Image.open(self.img_dir / f"{first_id}.png") as img:
                first_img = np.array(img, dtype=np.float32) / 255.0
            with Image.open(self.img_dir / f"{second_id}.png") as img:
                second_img = np.array(img, dtype=np.float32) / 255.0
            with Image.open(self.img_dir / f"{third_id}.png") as img:
                third_img = np.array(img, dtype=np.float32) / 255.0

            # Create input tensor with 2 channels (H,W,C format)
            input_arr = np.stack([first_img, third_img], axis=-1)  # Shape: (H, W, 2)

            # Create target array with channel dimension (H,W,C format)
            target_arr = second_img[..., None]  # Shape: (H, W, 1)

            # Apply transforms
            input_tensor = self.transform(input_arr)
            target_tensor = self.transform(target_arr)

            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
        return torch.stack(input_tensors), torch.stack(target_tensors)

    def get_item_target_is_interpolated(self, patient_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Get pairs where target is the average of input slices (interpolated)."""
        first_ids = patient_df["SOPInstanceUID"].iloc[:-1].values
        third_ids = patient_df["SOPInstanceUID"].iloc[1:].values
        input_tensors = []
        target_tensors = []
        for first_id, third_id in zip(first_ids, third_ids, strict=False):
            # Load images using context managers
            with Image.open(self.img_dir / f"{first_id}.png") as img:
                first_img = np.array(img, dtype=np.float32) / 255.0
            with Image.open(self.img_dir / f"{third_id}.png") as img:
                third_img = np.array(img, dtype=np.float32) / 255.0

            # Create input tensor with 2 channels (H,W,C format)
            input_arr = np.stack([first_img, third_img], axis=-1)  # Shape: (H, W, 2)

            # Create target array by averaging the two input channels (H,W,C format)
            target_arr = np.mean(input_arr[..., :2], axis=-1, keepdims=True)  # Shape: (H, W, 1)

            # Apply transforms
            input_tensor = self.transform(input_arr)
            target_tensor = self.transform(target_arr)

            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)
        return torch.stack(input_tensors), torch.stack(target_tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get all slice pairs for a patient by index."""
        patient_id = self.patients[idx]
        patient_df = self.df[self.df["PatientID"] == patient_id].sort_values(by="order")
        if self.mode == "target_is_real":
            return self.get_item_target_is_real(patient_df)
        if self.mode == "target_is_interpolated":
            return self.get_item_target_is_interpolated(patient_df)
        raise ValueError(f"Invalid mode: {self.mode}")

    def _log_test_dataset_info(self, stage: str, elapsed_time: float) -> None:
        """Log test dataset info for per-patient test datasets."""
        logger.info(
            f"Dataset initialization took {elapsed_time:.2f} seconds\n"
            f"Stage: {stage} (mode={self.mode})\n"
            f"Patients: {len(self.patients)}\n",
        )
