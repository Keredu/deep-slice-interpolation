"""Main Trainer class for CT slice interpolation training."""

import faulthandler
import math
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path
from typing import TextIO

import torch
import torch._inductor.config as inductor_config
from loguru import logger
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from phd.config_io import save_config
from phd.datasets.interpolation.two_to_one_slice import (
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)
from phd.datasets.interpolation.two_to_one_slice_cropped import (
    DEFAULT_CROP_WEIGHTS,
    NUM_CROPS,
    TwoToOneSliceCroppedDataset,
)
from phd.losses.custom import CustomLoss
from phd.metrics import compute_all_metrics
from phd.models.setup_model import setup_model
from phd.plotting import save_metrics_csv
from phd.training.checkpoint import load_checkpoint, save_checkpoint
from phd.training.early_stopping import EarlyStopping
from phd.training.registry import (
    get_experiment_status,
    register_experiment,
    update_experiment_status,
)
from phd.training.scheduler import is_per_batch_scheduler, needs_metric_for_step, setup_scheduler
from phd.training.status import TrainingStatus
from phd.viz import save_test_visualization

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
inductor_config.verbose_progress = False
inductor_config.fx_graph_cache = True  # Cache compiled Triton kernels
inductor_config.autotune_local_cache = True  # Cache autotuning benchmark results


class Trainer:
    """Trainer class for CT slice interpolation models.

    Handles the full training loop including:
    - Device setup and model initialization
    - Dataset and dataloader creation
    - Training and validation epochs
    - Checkpointing and early stopping
    - Metric tracking and visualization
    """

    def __init__(self, config: dict) -> None:
        """Initialize the trainer with configuration.

        Args:
            config: Training configuration dictionary
        """
        self.config = config

        # Core training components (initialized in setup)
        self.device: torch.device | None = None
        self.model: nn.Module | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self.early_stopping: EarlyStopping | None = None
        self.scaler: GradScaler | None = None

        # Datasets and loaders
        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None
        self.test_dataset_target_is_real: TwoToOneSliceTestDataset | None = None
        self.test_dataset_target_is_interpolated: TwoToOneSliceTestDataset | None = None

        # Training state
        self.start_epoch: int = 0
        self.best_valid_loss: float = float("inf")
        self.train_losses: list[float] = []
        self.valid_losses: list[float] = []
        self.best_epochs: list[bool] = []  # Track which epochs had best validation loss

        # Metric histories
        self.metric_histories: dict[str, list[float]] = {
            "ssim": [],
            "ms_ssim": [],
            "mae": [],
            "gradient_mae": [],
            "psnr": [],
            "ncc": [],
        }

        # Timing and learning rate histories (persisted across restarts)
        self.learning_rates: list[float] = []
        self.epoch_times: list[float] = []
        self.train_times: list[float] = []
        self.valid_times: list[float] = []

        # Per-crop metric histories for validation
        self.per_crop_metric_histories: dict[int, dict[str, list[float]]] = {
            i: {"ssim": [], "ms_ssim": [], "mae": [], "gradient_mae": [], "psnr": [], "ncc": []}
            for i in range(NUM_CROPS)
        }

        # Directories
        self.experiment_dir: Path | None = None
        self.epochs_dir: Path | None = None

        # Crash logging
        self._fault_file: TextIO | None = None

        # Track epoch times for current run (not persisted across restarts)
        self._run_epoch_times: list[float] = []

        # Transfer-init tracking (for explicit logging after file logger is attached)
        self._init_from_checkpoint_path: Path | None = None
        self._init_from_checkpoint_epoch: int | None = None

        # Validation DataLoader runtime settings (can be switched to safe mode on SHM errors)
        self._valid_loader_batch_size: int = 0
        self._valid_loader_num_workers: int = 0
        self._valid_loader_pin_memory: bool = False
        self._valid_loader_prefetch_factor: int = 1
        self._validation_safe_mode_enabled: bool = False

    def _log_batch_interval(
        self,
        phase: str,
        batch_idx: int,
        num_batches: int,
        interval_times: list[float],
        interval_losses: list[float],
    ) -> None:
        """Log batch interval statistics and clear interval accumulators.

        Args:
            phase: Phase name ("Train" or "Valid")
            batch_idx: Current batch index (0-based)
            num_batches: Total number of batches
            interval_times: List of batch times in seconds (will be cleared)
            interval_losses: List of batch losses (will be cleared)
        """
        avg_time = sum(interval_times) / len(interval_times) * 1000
        min_time = min(interval_times) * 1000
        max_time = max(interval_times) * 1000
        avg_loss_interval = sum(interval_losses) / len(interval_losses)
        logger.debug(
            f"  {phase} batch {batch_idx + 1}/{num_batches}: "
            f"loss={avg_loss_interval:.4f}, "
            f"time=[min={min_time:.1f}, avg={avg_time:.1f}, max={max_time:.1f}]ms"
        )
        interval_times.clear()
        interval_losses.clear()

    def _handle_resume_status(self, registry_dir: Path) -> tuple[bool, bool]:
        """Check experiment status and handle resume/skip logic.

        Determines whether to resume training, start fresh, or skip based on
        the experiment's status in the registry and existence of checkpoints.

        Args:
            registry_dir: Path to the registry directory

        Returns:
            Tuple of (resume_training, should_skip):
                - resume_training: True if we should load from checkpoint
                - should_skip: True if training is already complete
        """
        status = get_experiment_status(experiments_dir=registry_dir, exp_name=self.config["exp_name"])
        checkpoint_path = self.experiment_dir / "latest_epoch.pth"

        if status is None:
            # No status in registry - new experiment
            if self.experiment_dir.exists():
                logger.warning(
                    f"Experiment directory exists but not in registry. Cleaning up {self.config['exp_name']}"
                )
                shutil.rmtree(self.experiment_dir)
            self.experiment_dir.mkdir(exist_ok=False)
            return False, False

        if status == TrainingStatus.RUNNING:
            logger.info(f"Resuming training for {self.config['experiments_dir']}/{self.config['exp_name']})")
            return True, False

        if status in {TrainingStatus.FINISHED_EPOCHS, TrainingStatus.EARLY_STOPPING, TrainingStatus.NAN_VALUE_DETECTED}:
            logger.info(f"Training already finished for {self.config['exp_name']} with status {status}. Skipping.")
            return False, True

        if status == TrainingStatus.ERROR:
            # Try to resume from checkpoint if one exists
            if checkpoint_path.exists():
                exp_name = self.config["exp_name"]
                logger.info(f"Previous training for {exp_name} failed with ERROR. Resuming...")
                return True, False
            exp_name = self.config["exp_name"]
            logger.warning(f"Previous training for {exp_name} failed with ERROR. Restarting...")
            if self.experiment_dir.exists():
                shutil.rmtree(self.experiment_dir)
            self.experiment_dir.mkdir(exist_ok=False)
            return False, False

        if status == TrainingStatus.NOT_STARTED:
            # Try to resume from checkpoint if one exists (e.g., after reset from ERROR)
            if checkpoint_path.exists():
                logger.info(f"Checkpoint exists for {self.config['exp_name']}. Resuming...")
                return True, False
            if self.experiment_dir.exists():
                # Directory exists but no checkpoint - clean up and start fresh
                shutil.rmtree(self.experiment_dir)
            self.experiment_dir.mkdir(exist_ok=False)
            return False, False

        # Unknown status - start fresh
        self.experiment_dir.mkdir(exist_ok=True)
        return False, False

    def _configure_validation_loader_settings(self) -> None:
        """Configure validation DataLoader settings from config with safe defaults."""
        train_batch_size = int(self.config["batch_size"])
        requested_valid_batch_size = self.config.get("valid_batch_size")
        if requested_valid_batch_size is None:
            # Validation sample is ~9x larger than training sample (all crops),
            # so cap default validation batch size for stability.
            self._valid_loader_batch_size = min(train_batch_size, 64)
        else:
            self._valid_loader_batch_size = int(requested_valid_batch_size)
        self._valid_loader_batch_size = max(1, self._valid_loader_batch_size)

        self._valid_loader_num_workers = max(0, int(self.config.get("valid_num_workers", 0)))
        self._valid_loader_prefetch_factor = max(1, int(self.config.get("valid_prefetch_factor", 1)))

        if "valid_pin_memory" in self.config:
            self._valid_loader_pin_memory = bool(self.config["valid_pin_memory"])
        else:
            # With num_workers=0, pinned-memory thread provides little benefit
            # and increases complexity under heavy validation tensors.
            self._valid_loader_pin_memory = self._valid_loader_num_workers > 0

    def _create_valid_loader(self, valid_dataset: TwoToOneSliceCroppedDataset) -> DataLoader:
        """Create validation DataLoader using current runtime settings."""
        valid_loader_kwargs = {
            "batch_size": self._valid_loader_batch_size,
            "shuffle": False,
            "num_workers": self._valid_loader_num_workers,
            "pin_memory": self._valid_loader_pin_memory,
            "drop_last": True,
            "persistent_workers": False,
        }
        if self._valid_loader_num_workers > 0:
            # Keep prefetched validation batches minimal to reduce SHM pressure.
            valid_loader_kwargs["prefetch_factor"] = self._valid_loader_prefetch_factor

        return DataLoader(valid_dataset, **valid_loader_kwargs)

    def _enable_validation_safe_mode(self) -> None:
        """Switch validation DataLoader to conservative settings and rebuild it."""
        if self.valid_loader is None:
            raise RuntimeError("Validation DataLoader is not initialized")

        if self._validation_safe_mode_enabled:
            return

        valid_dataset = self.valid_loader.dataset

        # Drop references so workers/queues can be collected before recreating.
        self.valid_loader = None
        import gc

        gc.collect()

        self._valid_loader_batch_size = min(self._valid_loader_batch_size, 64)
        self._valid_loader_num_workers = 0
        self._valid_loader_pin_memory = False
        self._validation_safe_mode_enabled = True
        self.valid_loader = self._create_valid_loader(valid_dataset)

        logger.warning(
            "Validation safe mode enabled: "
            f"batch_size={self._valid_loader_batch_size}, "
            f"num_workers={self._valid_loader_num_workers}, "
            f"pin_memory={self._valid_loader_pin_memory}"
        )

    @staticmethod
    def _is_shm_allocation_error(error: RuntimeError) -> bool:
        """Return True if an exception matches known DataLoader SHM allocation failures."""
        message = str(error).lower()
        return (
            "unable to allocate shared memory" in message
            or "shared memory(shm)" in message
            or "unable to mmap" in message
        )

    def setup(self) -> bool:
        """Set up device, model, datasets, optimizer and check for resume.

        Returns:
            True if setup successful, False if training should be skipped
        """
        if not torch.cuda.is_available():
            raise RuntimeError("This script requires a GPU to run")

        self.device = torch.device("cuda")

        # Enable cuDNN auto-tuner for faster convolutions (input sizes are fixed)
        torch.backends.cudnn.benchmark = True

        # Initialize mixed precision scaler (reduces VRAM ~40-50%, speeds up training)
        self.scaler = GradScaler()

        # Create experiments directory and experiment-specific directory
        experiments_dir = Path(self.config["experiments_dir"])
        experiments_dir.mkdir(parents=True, exist_ok=True)
        registry_dir = Path(self.config["registry_dir"])
        registry_dir.mkdir(parents=True, exist_ok=True)

        if not self.config["exp_name"]:
            raise ValueError("Experiment name must be specified in config")

        self.experiment_dir = experiments_dir / self.config["exp_name"]

        # Check if we can resume or should skip (status from registry is source of truth)
        resume_training, should_skip = self._handle_resume_status(registry_dir)
        if should_skip:
            return False

        # Register experiment in the master registry
        register_experiment(
            experiments_dir=registry_dir,
            exp_name=self.config["exp_name"],
            config=self.config,
        )

        self.epochs_dir = self.experiment_dir / "epochs"
        self.epochs_dir.mkdir(parents=True, exist_ok=True)

        save_config(experiment_dir=self.experiment_dir, config=self.config)

        # Initialize datasets and dataloaders
        self._setup_datasets()

        # Initialize model, criterion, optimizer and scheduler
        self._setup_model()

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config["early_stopping_patience"],
            min_delta=self.config["early_stopping_delta"],
        )

        # Load state if resuming
        if resume_training:
            self._load_checkpoint()
        else:
            self._initialize_from_experiment_checkpoint()

        return True

    def _setup_datasets(self) -> None:
        """Initialize datasets and dataloaders."""
        self._validation_safe_mode_enabled = False

        train_dataset = TwoToOneSliceCroppedDataset(
            root_dir=self.config["data_path"],
            transform=STANDARD_TRANSFORM,
            stage="train",
            size=self.config["train_size"],
            flip_prob=self.config["flip_prob"],
            crop_weights=self.config.get("crop_weights", DEFAULT_CROP_WEIGHTS),
            include_resize_512=self.config.get("include_resize_512", True),
            resize_512_weight=self.config.get("resize_512_weight", 0.5),
            include_resize_384=self.config.get("include_resize_384", False),
            resize_384_weight=self.config.get("resize_384_weight", 0.5),
            rotation_prob=self.config.get("rotation_prob", 0.5),
            rotation_max_angle=self.config.get("rotation_max_angle", 15.0),
        )

        valid_dataset = TwoToOneSliceCroppedDataset(
            root_dir=self.config["data_path"],
            transform=STANDARD_TRANSFORM,
            stage="valid",
            size=self.config["valid_size"],
            flip_prob=0.0,
            return_all_augmentations=True,
            # Validation doesn't use augmentation params - always returns 9 crops
        )

        train_loader_kwargs = {
            "batch_size": self.config["batch_size"],
            "shuffle": True,
            "num_workers": self.config["num_workers"],
            "pin_memory": True,
            "drop_last": True,
            # Disabled to avoid SHM race condition (see docs/issues/DATALOADER_SHM_ERROR.md).
            "persistent_workers": False,
        }
        if self.config["num_workers"] > 0 and "train_prefetch_factor" in self.config:
            train_loader_kwargs["prefetch_factor"] = max(1, int(self.config["train_prefetch_factor"]))

        self.train_loader = DataLoader(
            train_dataset,
            **train_loader_kwargs,
        )

        self._configure_validation_loader_settings()
        self.valid_loader = self._create_valid_loader(valid_dataset)
        logger.info(
            "Validation DataLoader configured: "
            f"batch_size={self._valid_loader_batch_size}, "
            f"num_workers={self._valid_loader_num_workers}, "
            f"pin_memory={self._valid_loader_pin_memory}, "
            f"prefetch_factor={self._valid_loader_prefetch_factor if self._valid_loader_num_workers > 0 else 'n/a'}"
        )

        # Only load test datasets if visualizations are enabled
        if self.config.get("generate_test_viz_real", True):
            self.test_dataset_target_is_real = TwoToOneSliceTestDataset(
                root_dir=self.config["data_path"],
                transform=STANDARD_TRANSFORM,
                stage="test",
                mode="target_is_real",
            )

        if self.config.get("generate_test_viz_interpolated", False):
            self.test_dataset_target_is_interpolated = TwoToOneSliceTestDataset(
                root_dir=self.config["data_path"],
                transform=STANDARD_TRANSFORM,
                stage="test",
                mode="target_is_interpolated",
            )

    def _setup_model(self) -> None:
        """Initialize model, criterion, optimizer and scheduler."""
        # Get in/out channels from train_loader dataset
        train_dataset = self.train_loader.dataset

        # Get model config (nested structure)
        model_config = self.config["model"]

        # Suppress warnings from external libraries when loading pretrained weights:
        # - HF Hub unauthenticated requests warning
        # - torchvision deprecation warning (pretrained vs weights API)
        # - timm/smp unexpected keys warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
            warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*Arguments other than.*")
            warnings.filterwarnings("ignore", message=".*Unexpected keys.*")
            self.model = setup_model(
                in_channels=train_dataset.get_in_channels(),
                out_channels=train_dataset.get_out_channels(),
                pretrained=model_config.get("pretrained", True),
                model_type=model_config["type"],
                encoder_name=model_config.get("encoder_name"),
            )
        self.model = self.model.to(self.device)

        # JIT compile for ~22% speedup over eager mode (first iteration is slow due to compilation)
        # Using 'default' mode to avoid CUDA graphs which pre-allocate ~10GB and don't release,
        # causing OOM during validation. 'default' is the only simple mode without CUDA graphs.
        # Benchmark (2026-02-05): eager=218ms, default=179ms, reduce-overhead=176ms, max-autotune=160ms
        logger.info("Compiling model with torch.compile(mode='default')...")
        self.model = torch.compile(self.model, mode="default")

        # Warmup: trigger Triton compilation before training loop
        # Use autocast to match training conditions (different dtypes = different kernels)
        # Use synthetic tensor instead of train_loader to avoid persistent_workers deadlock
        logger.info("Warming up compiled model (Triton kernel compilation)...")
        batch_size = self.config["batch_size"]
        warmup_input = torch.randn(batch_size, 2, 256, 256, device=self.device)
        with torch.no_grad(), autocast(device_type="cuda"):
            self.model(warmup_input)
        torch.cuda.synchronize()
        del warmup_input
        logger.info("Warmup complete")

        self.criterion = CustomLoss(self.config["loss"]).to(self.device)
        logger.info(f"Using loss function: {self.config['loss']['name']}")

        # Setup optimizer from config
        self.optimizer = self._create_optimizer()
        logger.info(f"Using optimizer: {self.config['optimizer']['name']}")

        # Setup scheduler from config
        self.scheduler = setup_scheduler(
            optimizer=self.optimizer,
            scheduler_config=self.config["scheduler"],
            num_epochs=self.config["num_epochs"],
            steps_per_epoch=len(self.train_loader),
        )
        logger.info(f"Using scheduler: {self.config['scheduler']['name']}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config.

        Returns:
            Configured optimizer instance
        """
        optimizer_config = self.config["optimizer"]
        optimizer_name = optimizer_config["name"].lower()
        params = optimizer_config.get("params", {})

        # Extract common parameters
        lr = params.get("lr", 3e-4)
        weight_decay = params.get("weight_decay", 0.0)

        if optimizer_name == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=params.get("betas", (0.9, 0.999)),
                eps=params.get("eps", 1e-8),
            )
        elif optimizer_name == "adam":
            return Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=params.get("betas", (0.9, 0.999)),
                eps=params.get("eps", 1e-8),
            )
        elif optimizer_name == "sgd":
            return SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=params.get("momentum", 0.0),
                nesterov=params.get("nesterov", False),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @staticmethod
    def _adapt_state_dict_for_compiled_model(
        model_state_dict: dict[str, torch.Tensor],
        expected_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Adapt state dict keys between compiled and eager model formats.

        Torch compiled models usually prefix parameter names with ``_orig_mod.``.
        This helper keeps transfer initialization robust across code versions.
        """
        if not model_state_dict or not expected_keys:
            return model_state_dict

        expected_has_prefix = expected_keys[0].startswith("_orig_mod.")
        incoming_keys = list(model_state_dict.keys())
        incoming_has_prefix = incoming_keys[0].startswith("_orig_mod.")

        if expected_has_prefix == incoming_has_prefix:
            return model_state_dict

        prefix = "_orig_mod."
        if expected_has_prefix and not incoming_has_prefix:
            return {f"{prefix}{key}": value for key, value in model_state_dict.items()}
        if not expected_has_prefix and incoming_has_prefix:
            return {
                key[len(prefix) :] if key.startswith(prefix) else key: value
                for key, value in model_state_dict.items()
            }
        return model_state_dict

    def _resolve_init_checkpoint_path(self) -> Path | None:
        """Resolve optional initialization checkpoint path from config."""
        source_experiment = self.config.get("init_from_experiment")
        if not source_experiment:
            return None

        if source_experiment == self.config["exp_name"]:
            raise ValueError("init_from_experiment must be different from exp_name")

        checkpoint_spec = self.config.get("init_from_checkpoint") or "latest_epoch.pth"
        checkpoint_path = Path(checkpoint_spec)
        if checkpoint_path.is_absolute():
            return checkpoint_path
        return Path(self.config["experiments_dir"]) / source_experiment / checkpoint_path

    def _initialize_from_experiment_checkpoint(self) -> None:
        """Initialize model weights from another experiment checkpoint."""
        checkpoint_path = self._resolve_init_checkpoint_path()
        if checkpoint_path is None:
            return
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {checkpoint_path}. "
                "Verify init_from_experiment/init_from_checkpoint in config."
            )

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
            source_epoch = checkpoint.get("epoch")
        else:
            # Support raw torch state_dict files.
            model_state_dict = checkpoint
            source_epoch = None

        if not isinstance(model_state_dict, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain a valid model state_dict")

        model_state_dict = self._adapt_state_dict_for_compiled_model(
            model_state_dict=model_state_dict,
            expected_keys=list(self.model.state_dict().keys()),
        )
        self.model.load_state_dict(model_state_dict)
        self._init_from_checkpoint_path = checkpoint_path
        self._init_from_checkpoint_epoch = source_epoch if isinstance(source_epoch, int) else None

        if isinstance(source_epoch, int):
            logger.info(
                f"Initialized model weights from {checkpoint_path} (source epoch {source_epoch + 1})"
            )
        else:
            logger.info(f"Initialized model weights from {checkpoint_path}")

    def _load_checkpoint(self) -> None:
        """Load checkpoint if resuming training."""
        latest_checkpoint = self.experiment_dir / "latest_epoch.pth"
        if latest_checkpoint.exists():
            state = load_checkpoint(
                path=latest_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
            )
            self.start_epoch = state["epoch"] + 1
            self.train_losses = state["train_losses"]
            self.valid_losses = state["valid_losses"]
            self.best_valid_loss = state["best_valid_loss"]
            if state["metric_histories"]:
                self.metric_histories = state["metric_histories"]
            if state.get("best_epochs"):
                self.best_epochs = state["best_epochs"]
            if state.get("per_crop_metric_histories"):
                self.per_crop_metric_histories = state["per_crop_metric_histories"]
            if state.get("learning_rates"):
                self.learning_rates = state["learning_rates"]
            if state.get("epoch_times"):
                self.epoch_times = state["epoch_times"]
            if state.get("train_times"):
                self.train_times = state["train_times"]
            if state.get("valid_times"):
                self.valid_times = state["valid_times"]
            logger.info(f"Resumed from epoch {self.start_epoch}")
        else:
            logger.warning("Resume flag set but no checkpoint found. Starting from scratch.")

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, epoch_duration_seconds)
        """
        self.model.train()
        train_loss = 0
        num_batches = len(self.train_loader)

        # Check if scheduler should step per batch (e.g., OneCycleLR)
        step_scheduler_per_batch = is_per_batch_scheduler(self.config["scheduler"])

        batch_log_interval = max(1, num_batches // 5)  # Log 5 times per epoch
        logger.debug(
            f"Starting training epoch {epoch + 1} with {num_batches} batches (batch_log_interval={batch_log_interval})"
        )
        epoch_start = time.perf_counter()
        batch_times: list[float] = []
        interval_times: list[float] = []
        interval_losses: list[float] = []

        for batch_idx, (batch_inputs, batch_targets) in enumerate(self.train_loader):
            batch_start = time.perf_counter()

            # non_blocking=True overlaps CPU->GPU transfer with computation
            inputs = batch_inputs.to(self.device, non_blocking=True)
            targets = batch_targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type="cuda"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            train_loss += batch_loss

            batch_time = time.perf_counter() - batch_start
            batch_times.append(batch_time)
            interval_times.append(batch_time)
            interval_losses.append(batch_loss)

            # Step scheduler per batch if needed (e.g., OneCycleLR)
            if step_scheduler_per_batch:
                self.scheduler.step()

            # Debug log at batch_log_interval
            if (batch_idx + 1) % batch_log_interval == 0:
                self._log_batch_interval("Train", batch_idx, num_batches, interval_times, interval_losses)

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = train_loss / num_batches
        avg_batch_time = sum(batch_times) / len(batch_times)

        logger.debug(
            f"Training epoch {epoch + 1} complete: "
            f"avg_loss={avg_loss:.4f}, time={epoch_time:.2f}s, "
            f"avg_batch_time={avg_batch_time * 1000:.1f}ms"
        )

        return avg_loss, epoch_time

    def validate_epoch(self, epoch: int) -> tuple[float, dict[str, float], dict[int, dict[str, float]], float]:
        """Run a single validation epoch on 256×256 crops with per-crop metrics.

        All validation is done at 256×256 resolution using 9 crops from
        center 384×384 region. Computes both aggregated and per-crop metrics.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (valid_loss, epoch_metrics, per_crop_metrics, epoch_duration_seconds)
        """
        try:
            return self._validate_epoch_impl(epoch)
        except RuntimeError as error:
            if not self._is_shm_allocation_error(error):
                raise

            if self._validation_safe_mode_enabled:
                logger.error(
                    "Validation failed with shared-memory error even in safe mode; "
                    "aborting current experiment run."
                )
                raise

            logger.warning(
                "Validation failed with DataLoader shared-memory error. "
                "Rebuilding validation loader in safe mode and retrying epoch once."
            )
            torch.cuda.empty_cache()
            self._enable_validation_safe_mode()
            return self._validate_epoch_impl(epoch)

    def _validate_epoch_impl(self, epoch: int) -> tuple[float, dict[str, float], dict[int, dict[str, float]], float]:
        """Run validation implementation without retry logic."""
        self.model.eval()
        valid_loss = 0.0
        num_batches = len(self.valid_loader)

        epoch_metrics = {
            "ssim": 0.0,
            "ms_ssim": 0.0,
            "mae": 0.0,
            "gradient_mae": 0.0,
            "psnr": 0.0,
            "ncc": 0.0,
        }

        # Per-crop metrics accumulators
        per_crop_metrics: dict[int, dict[str, float]] = {
            i: {"ssim": 0.0, "ms_ssim": 0.0, "mae": 0.0, "gradient_mae": 0.0, "psnr": 0.0, "ncc": 0.0}
            for i in range(NUM_CROPS)
        }
        per_crop_counts: dict[int, int] = dict.fromkeys(range(NUM_CROPS), 0)
        num_batches_for_metrics = 0

        batch_log_interval = max(1, num_batches // 5)  # Log 5 times per epoch
        logger.debug(
            f"Starting validation epoch {epoch + 1} with {num_batches} batches "
            f"(batch_log_interval={batch_log_interval})"
        )
        epoch_start = time.perf_counter()
        batch_times: list[float] = []
        interval_times: list[float] = []
        interval_losses: list[float] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                batch_start = time.perf_counter()

                # Dataset returns (aug_inputs, aug_targets) with shapes:
                # aug_inputs: (batch_size, 9, 2, 256, 256)
                # aug_targets: (batch_size, 9, 1, 256, 256)
                aug_inputs, aug_targets = batch

                aug_inputs = aug_inputs.to(self.device, non_blocking=True)
                aug_targets = aug_targets.to(self.device, non_blocking=True)

                batch_loss = 0.0
                for crop_idx in range(NUM_CROPS):
                    crop_inputs = aug_inputs[:, crop_idx]  # (B, 2, 256, 256)
                    crop_targets = aug_targets[:, crop_idx]  # (B, 1, 256, 256)

                    # Compute loss with mixed precision
                    with autocast(device_type="cuda"):
                        outputs = self.model(crop_inputs)
                        loss = self.criterion(outputs, crop_targets)

                    batch_loss += loss.item()

                    # Compute metrics per crop (float32 for metric compatibility)
                    crop_metrics = compute_all_metrics(
                        pred=outputs.float(),
                        target=crop_targets.float(),
                        data_range=1.0,
                    )
                    for key in epoch_metrics:
                        per_crop_metrics[crop_idx][key] += crop_metrics[key]
                        epoch_metrics[key] += crop_metrics[key]
                    per_crop_counts[crop_idx] += 1

                # Average loss across 9 crops for this batch
                valid_loss += batch_loss / NUM_CROPS
                num_batches_for_metrics += NUM_CROPS

                batch_time = time.perf_counter() - batch_start
                batch_times.append(batch_time)
                interval_times.append(batch_time)
                interval_losses.append(batch_loss / NUM_CROPS)

                # Debug log at batch_log_interval
                if (batch_idx + 1) % batch_log_interval == 0:
                    self._log_batch_interval("Valid", batch_idx, num_batches, interval_times, interval_losses)

        valid_loss /= num_batches

        # Average aggregated metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches_for_metrics
            self.metric_histories[key].append(epoch_metrics[key])

        # Average per-crop metrics and store in history
        for crop_idx in range(NUM_CROPS):
            count = per_crop_counts[crop_idx]
            if count > 0:
                for key in per_crop_metrics[crop_idx]:
                    per_crop_metrics[crop_idx][key] /= count
                    self.per_crop_metric_histories[crop_idx][key].append(per_crop_metrics[crop_idx][key])

        epoch_time = time.perf_counter() - epoch_start
        avg_batch_time = sum(batch_times) / len(batch_times)

        # Log per-crop SSIM summary
        center_ssim = per_crop_metrics[4]["ssim"]
        corner_ssims = [per_crop_metrics[i]["ssim"] for i in [0, 2, 6, 8]]
        edge_ssims = [per_crop_metrics[i]["ssim"] for i in [1, 3, 5, 7]]
        logger.debug(
            f"  Per-crop SSIM: center={center_ssim:.4f}, "
            f"corners={sum(corner_ssims) / 4:.4f}, edges={sum(edge_ssims) / 4:.4f}"
        )

        logger.debug(
            f"Validation epoch {epoch + 1} complete: "
            f"avg_loss={valid_loss:.4f}, time={epoch_time:.2f}s, "
            f"avg_batch_time={avg_batch_time * 1000:.1f}ms"
        )

        return valid_loss, epoch_metrics, per_crop_metrics, epoch_time

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        train_loss: float,
        valid_loss: float,
    ) -> None:
        """Save a checkpoint."""
        save_checkpoint(
            path=path,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loss=train_loss,
            valid_loss=valid_loss,
            best_valid_loss=self.best_valid_loss,
            train_losses=self.train_losses,
            valid_losses=self.valid_losses,
            metric_histories=self.metric_histories,
            config=self.config,
            early_stopping=self.early_stopping,
            best_epochs=self.best_epochs,
            per_crop_metric_histories=self.per_crop_metric_histories,
            learning_rates=self.learning_rates,
            epoch_times=self.epoch_times,
            train_times=self.train_times,
            valid_times=self.valid_times,
        )

    def _save_visualizations(self, epoch: int) -> None:
        """Save test visualizations for a new best model."""
        # Skip if no visualizations are enabled
        generate_real = self.config.get("generate_test_viz_real", True)
        generate_interpolated = self.config.get("generate_test_viz_interpolated", False)

        if not generate_real and not generate_interpolated:
            return

        torch.cuda.empty_cache()

        epoch_dir = self.epochs_dir / str(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = epoch_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        if generate_real and self.test_dataset_target_is_real is not None:
            logger.info("Saving test visualizations for `target is real` mode")
            save_test_visualization(
                test_dataset=self.test_dataset_target_is_real,
                model=self.model,
                device=self.device,
                save_dir=viz_dir / "target_is_real",
                batch_size=self.config["batch_size"],
            )

        if generate_interpolated and self.test_dataset_target_is_interpolated is not None:
            logger.info("Saving test visualizations for `target is interpolated` mode")
            save_test_visualization(
                test_dataset=self.test_dataset_target_is_interpolated,
                model=self.model,
                device=self.device,
                save_dir=viz_dir / "target_is_interpolated",
                batch_size=self.config["batch_size"],
            )

    def _save_error_log(self, error: Exception) -> None:
        """Save full traceback to error.log in experiment directory.

        Args:
            error: The exception that caused the error
        """
        if self.experiment_dir is None:
            return

        error_log = self.experiment_dir / "error.log"
        with error_log.open("w", encoding="utf-8") as f:
            f.write(f"Error: {error}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())

    def run(self) -> None:
        """Run the full training loop."""
        if not self.setup():
            return

        # Add file logging for this experiment
        log_file = self.experiment_dir / "training.log"
        log_handler_id = logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation=None,
        )
        logger.info("=" * 80)
        logger.info("TRAINING SESSION STARTED")
        logger.info("=" * 80)

        if self._init_from_checkpoint_path is not None:
            if self._init_from_checkpoint_epoch is not None:
                logger.info(
                    "Transfer initialization active: "
                    f"loaded weights from {self._init_from_checkpoint_path} "
                    f"(source epoch {self._init_from_checkpoint_epoch + 1})"
                )
            else:
                logger.info(
                    "Transfer initialization active: "
                    f"loaded weights from {self._init_from_checkpoint_path}"
                )

        # Enable faulthandler to capture tracebacks on fatal signals (SIGSEGV, SIGABRT, etc.)
        if self.experiment_dir is not None:
            crash_trace_path = self.experiment_dir / "crash_trace.txt"
            self._fault_file = crash_trace_path.open("w")
            faulthandler.enable(file=self._fault_file)
            logger.debug(f"Faulthandler enabled, writing to {crash_trace_path}")

        try:
            total_epochs = self.config["num_epochs"]
            logger.info(
                f"Starting training: epochs {self.start_epoch + 1}-{total_epochs}, experiment={self.config['exp_name']}"
            )

            exp_name = self.config["exp_name"]
            for epoch in range(self.start_epoch, total_epochs):
                logger.info(f"[{exp_name}] Epoch {epoch + 1}/{total_epochs} starting")

                # Training pass
                train_loss, train_time = self.train_epoch(epoch)

                # Check for NaN/inf in training loss
                if math.isnan(train_loss) or math.isinf(train_loss):
                    logger.error(f"[{exp_name}] Training loss became {train_loss} at epoch {epoch + 1}")
                    update_experiment_status(
                        experiments_dir=Path(self.config["registry_dir"]),
                        exp_name=self.config["exp_name"],
                        status=TrainingStatus.NAN_VALUE_DETECTED,
                        final_epoch=epoch + 1,
                    )
                    return

                # Validation pass (all at 256x256, with per-crop metrics)
                valid_loss, epoch_metrics, _per_crop_metrics, valid_time = self.validate_epoch(epoch)

                # Check for NaN/inf in validation loss
                if math.isnan(valid_loss) or math.isinf(valid_loss):
                    logger.error(f"[{exp_name}] Validation loss became {valid_loss} at epoch {epoch + 1}")
                    update_experiment_status(
                        experiments_dir=Path(self.config["registry_dir"]),
                        exp_name=self.config["exp_name"],
                        status=TrainingStatus.NAN_VALUE_DETECTED,
                        final_epoch=epoch + 1,
                    )
                    return

                self.train_losses.append(train_loss)
                self.valid_losses.append(valid_loss)

                # Check if this is a new best epoch
                is_best = valid_loss < self.best_valid_loss
                self.best_epochs.append(is_best)

                # Track timing and learning rate (persisted across restarts)
                current_lr = self.optimizer.param_groups[0]["lr"]
                epoch_total_time = train_time + valid_time
                self.learning_rates.append(current_lr)
                self.epoch_times.append(epoch_total_time)
                self.train_times.append(train_time)
                self.valid_times.append(valid_time)

                # Track epoch time for current run only (for run_avg calculation)
                self._run_epoch_times.append(epoch_total_time)
                avg_epoch_time = sum(self._run_epoch_times) / len(self._run_epoch_times)

                # Save epoch data to CSV (metrics, timing, learning rate)
                logger.debug("Saving epochs CSV")
                save_metrics_csv(
                    metric_histories=self.metric_histories,
                    save_dir=self.experiment_dir,
                    train_losses=self.train_losses,
                    valid_losses=self.valid_losses,
                    best_epochs=self.best_epochs,
                    learning_rates=self.learning_rates,
                    epoch_times=self.epoch_times,
                    train_times=self.train_times,
                    valid_times=self.valid_times,
                )

                # Log epoch summary
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items() if v > 0)
                logger.info(
                    f"[{exp_name}] Epoch {epoch + 1}/{total_epochs} completed: "
                    f"train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
                    f"best={self.best_valid_loss:.4f}, "
                    f"time={epoch_total_time:.2f}s (train={train_time:.2f}s, valid={valid_time:.2f}s), "
                    f"run_avg={avg_epoch_time:.2f}s (n={len(self._run_epoch_times)})"
                )
                logger.debug(f"Learning rate: {current_lr:.6f}")
                logger.info(f"  Metrics: {metrics_str}")

                # Save visualization if we have a new best model
                if is_best:
                    logger.info(f"New best validation loss: {valid_loss:.4f} (prev: {self.best_valid_loss:.4f})")
                    self._save_visualizations(epoch)
                    self.best_valid_loss = valid_loss

                    # Save best model
                    epoch_dir = self.epochs_dir / str(epoch)
                    epoch_dir.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Saving best model checkpoint to {epoch_dir}")
                    self._save_checkpoint(
                        path=epoch_dir / "weights.pth",
                        epoch=epoch,
                        train_loss=train_loss,
                        valid_loss=valid_loss,
                    )

                # Save latest checkpoint
                logger.debug("Saving latest checkpoint")
                self._save_checkpoint(
                    path=self.experiment_dir / "latest_epoch.pth",
                    epoch=epoch,
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                )

                # Early stopping check
                if self.early_stopping(valid_loss):
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs!")
                    update_experiment_status(
                        experiments_dir=Path(self.config["registry_dir"]),
                        exp_name=self.config["exp_name"],
                        status=TrainingStatus.EARLY_STOPPING,
                        best_valid_loss=self.best_valid_loss,
                        final_epoch=epoch + 1,
                    )
                    return

                # Step scheduler (skip if already stepped per batch)
                if not is_per_batch_scheduler(self.config["scheduler"]):
                    logger.debug("Stepping scheduler")
                    if needs_metric_for_step(self.config["scheduler"]):
                        self.scheduler.step(valid_loss)
                    else:
                        self.scheduler.step()

            logger.info(f"Training completed: {total_epochs} epochs, best_valid_loss={self.best_valid_loss:.4f}")
            update_experiment_status(
                experiments_dir=Path(self.config["registry_dir"]),
                exp_name=self.config["exp_name"],
                status=TrainingStatus.FINISHED_EPOCHS,
                best_valid_loss=self.best_valid_loss,
                final_epoch=self.config["num_epochs"],
            )

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            self._save_error_log(e)
            update_experiment_status(
                experiments_dir=Path(self.config["registry_dir"]),
                exp_name=self.config["exp_name"],
                status=TrainingStatus.ERROR,
            )
            raise

        finally:
            # Clean up DataLoaders to shutdown worker processes
            # (prevents hang on exit with spawn multiprocessing)
            import gc

            self.train_loader = None
            self.valid_loader = None
            gc.collect()

            # Clean up file logger
            logger.remove(log_handler_id)

            # Clean up faulthandler file
            if self._fault_file is not None:
                faulthandler.disable()
                self._fault_file.close()
                self._fault_file = None
