"""Training configuration and dataset path utilities."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Supported scheduler names
SUPPORTED_SCHEDULERS = {
    "cosine",  # CosineAnnealingLR - smooth decay to eta_min
    "cosine_restarts",  # CosineAnnealingWarmRestarts - periodic resets to escape local minima
    "cosine_warmup",  # Linear warmup + cosine decay
    "onecycle",  # OneCycleLR - warmup + aggressive decay (step per batch)
    "plateau",  # ReduceLROnPlateau - reduce on validation plateau
    "step",  # StepLR - reduce by gamma every step_size epochs
    "none",  # No scheduler - constant LR
}

# Supported optimizer names
SUPPORTED_OPTIMIZERS = {
    "adamw",  # AdamW - Adam with decoupled weight decay (recommended)
    "adam",  # Adam - adaptive learning rate
    "sgd",  # SGD - stochastic gradient descent (with optional momentum)
}

# Supported loss names (must match CustomLoss in phd/losses/custom.py)
SUPPORTED_LOSSES = {
    "mse",
    "l1",
    "ssim",
    "msssim",
    "ssim+l1",
    "msssim+l1",
}

# Image Dataset parameters
WINDOW_CENTER = 44
WINDOW_WIDTH = 128
MAX_SIZE = 512
# HU window bounds: [-20, 107] (128 discrete values)
HU_MIN = WINDOW_CENTER - WINDOW_WIDTH // 2  # -20
HU_MAX = WINDOW_CENTER + WINDOW_WIDTH // 2 - 1  # 107

# Batch size constant
BATCH_SIZE = 96


def get_dataset_dir() -> Path:
    """Get the preprocessed dataset directory path.

    Returns:
        Path to the preprocessed image dataset directory

    Raises:
        RuntimeError: If DATASETS_DIR is not set or directory doesn't exist
    """
    datasets_dir = os.getenv("DATASETS_DIR")
    if not datasets_dir:
        raise RuntimeError("DATASETS_DIR environment variable is not set. Please define it in the .env file.")

    pre_dir = Path(datasets_dir, "pre/rsna-intracranial-hemorrhage-detection")
    ds_name = f"1x{MAX_SIZE}x{MAX_SIZE}_{HU_MIN}_{HU_MAX}"
    img_dataset_dir = Path(pre_dir, ds_name)

    if not img_dataset_dir.exists():
        raise RuntimeError(
            f"Image dataset directory {img_dataset_dir} does not exist. "
            "Please check the DATASETS_DIR environment variable."
        )

    return img_dataset_dir


# Supported model types
SUPPORTED_MODELS = {
    "unet",
}


def _validate_scheduler_params(scheduler_name: str, params: dict) -> None:
    """Validate scheduler-specific parameters.

    All scheduler params must be explicitly configured - no defaults allowed.

    Args:
        scheduler_name: Name of the scheduler
        params: Scheduler parameters dictionary

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_params = {
        "cosine": ["eta_min"],
        "cosine_restarts": ["T_0", "T_mult", "eta_min"],
        "cosine_warmup": ["warmup_epochs", "eta_min"],
        "onecycle": ["max_lr", "pct_start", "anneal_strategy"],
        "plateau": ["mode", "factor", "patience", "min_lr"],
        "step": ["step_size", "gamma"],
        "none": [],
    }

    required = required_params.get(scheduler_name, [])
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Scheduler '{scheduler_name}' requires parameters: {missing}")

    # Validate warmup_epochs if present
    if "warmup_epochs" in params and params["warmup_epochs"] < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {params['warmup_epochs']}")

    # Validate T_0 for cosine_restarts
    if "T_0" in params and params["T_0"] <= 0:
        raise ValueError(f"T_0 must be > 0, got {params['T_0']}")

    # Validate T_mult for cosine_restarts
    if "T_mult" in params and params["T_mult"] < 1:
        raise ValueError(f"T_mult must be >= 1, got {params['T_mult']}")


def _validate_loss_weights(loss_name: str, params: dict) -> None:
    """Validate that loss weights sum to 1.0.

    Args:
        loss_name: Name of the loss function
        params: Loss parameters dictionary

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    # Define weight keys for each combined loss
    weight_keys_map = {
        "ssim+l1": ["ssim_weight", "l1_weight"],
        "msssim+l1": ["msssim_weight", "l1_weight"],
    }

    weight_keys = weight_keys_map.get(loss_name)
    if weight_keys is None:
        return  # Not a combined loss, no weight validation needed

    # Check all required weights are present
    missing = [k for k in weight_keys if k not in params]
    if missing:
        raise ValueError(f"Loss '{loss_name}' requires weights: {missing}")

    # Check weights sum to 1.0
    total = sum(params[k] for k in weight_keys)
    if abs(total - 1.0) > 1e-6:
        weights_str = ", ".join(f"{k}={params[k]}" for k in weight_keys)
        raise ValueError(f"Loss weights must sum to 1.0, got {total:.4f} ({weights_str})")


def _validate_augmentation_config(config: dict) -> None:
    """Validate augmentation configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If augmentation parameters are invalid
    """
    crop_weights = config.get("crop_weights", ())
    if len(crop_weights) != 9:
        raise ValueError(f"crop_weights must have 9 values, got {len(crop_weights)}")
    if any(w < 0 for w in crop_weights):
        raise ValueError("crop_weights must be non-negative")

    rotation_prob = config.get("rotation_prob", 0.5)
    if not 0 <= rotation_prob <= 1:
        raise ValueError(f"rotation_prob must be in [0, 1], got {rotation_prob}")

    rotation_max = config.get("rotation_max_angle", 15.0)
    if rotation_max < 0:
        raise ValueError(f"rotation_max_angle must be >= 0, got {rotation_max}")

    resize_512_weight = config.get("resize_512_weight", 0.5)
    if resize_512_weight < 0:
        raise ValueError(f"resize_512_weight must be >= 0, got {resize_512_weight}")

    resize_384_weight = config.get("resize_384_weight", 0.5)
    if resize_384_weight < 0:
        raise ValueError(f"resize_384_weight must be >= 0, got {resize_384_weight}")


def _validate_model_config(model_config: dict) -> None:
    """Validate model configuration.

    Args:
        model_config: Model configuration dictionary

    Raises:
        ValueError: If model configuration is invalid
    """
    if "type" not in model_config:
        raise ValueError("'model.type' is required")

    model_type = model_config["type"].lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: '{model_type}'. Supported models: {sorted(SUPPORTED_MODELS)}")

    # UNet requires encoder_name
    if model_type == "unet" and "encoder_name" not in model_config:
        raise ValueError("'model.encoder_name' is required for unet model")


def _validate_optimizer_config(optimizer_config: dict) -> None:
    """Validate optimizer configuration.

    Args:
        optimizer_config: Optimizer configuration dictionary

    Raises:
        ValueError: If optimizer configuration is invalid
    """
    if "name" not in optimizer_config:
        raise ValueError("'optimizer.name' is required")

    optimizer_name = optimizer_config["name"].lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer name: '{optimizer_name}'. Supported optimizers: {sorted(SUPPORTED_OPTIMIZERS)}"
        )

    params = optimizer_config.get("params", {})

    # Learning rate is required for all optimizers
    if "lr" not in params:
        raise ValueError("'optimizer.params.lr' is required")

    lr = params["lr"]
    if lr <= 0:
        raise ValueError(f"optimizer.params.lr must be > 0, got {lr}")

    # SGD requires momentum if provided, and doesn't use weight_decay the same way
    # but we don't enforce momentum - it has a sensible default of 0


def validate_config(config: dict) -> None:
    """Validate training configuration values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If any configuration value is invalid
    """
    if config.get("batch_size", 0) <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.get('batch_size')}")

    if config.get("num_epochs", 0) <= 0:
        raise ValueError(f"num_epochs must be > 0, got {config.get('num_epochs')}")

    if config.get("num_workers", 0) < 0:
        raise ValueError(f"num_workers must be >= 0, got {config.get('num_workers')}")

    valid_batch_size = config.get("valid_batch_size")
    if valid_batch_size is not None and valid_batch_size <= 0:
        raise ValueError(f"valid_batch_size must be > 0, got {valid_batch_size}")

    if config.get("valid_num_workers", 0) < 0:
        raise ValueError(f"valid_num_workers must be >= 0, got {config.get('valid_num_workers')}")

    if config.get("valid_prefetch_factor", 1) <= 0:
        raise ValueError(f"valid_prefetch_factor must be > 0, got {config.get('valid_prefetch_factor')}")

    init_from_experiment = config.get("init_from_experiment")
    init_from_checkpoint = config.get("init_from_checkpoint")

    if init_from_experiment is not None:
        if not isinstance(init_from_experiment, str) or not init_from_experiment.strip():
            raise ValueError(
                f"init_from_experiment must be a non-empty string when provided, got {init_from_experiment!r}"
            )

    if init_from_checkpoint is not None:
        if not isinstance(init_from_checkpoint, str) or not init_from_checkpoint.strip():
            raise ValueError(
                f"init_from_checkpoint must be a non-empty string when provided, got {init_from_checkpoint!r}"
            )
        if init_from_experiment is None:
            raise ValueError("init_from_checkpoint requires init_from_experiment to be set")

    # Validate model (has default, so always present after create_config)
    if "model" in config:
        _validate_model_config(config["model"])

    # Validate loss (required)
    if "loss" not in config:
        raise ValueError("'loss' configuration is required")
    loss_config = config["loss"]
    if "name" not in loss_config:
        raise ValueError("'loss.name' is required")
    loss_name = loss_config["name"].lower()
    if loss_name not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported loss name: '{loss_name}'. Supported losses: {sorted(SUPPORTED_LOSSES)}")
    _validate_loss_weights(loss_name, loss_config.get("params", {}))

    # Validate scheduler (required)
    if "scheduler" not in config:
        raise ValueError("'scheduler' configuration is required")
    scheduler_config = config["scheduler"]
    if "name" not in scheduler_config:
        raise ValueError("'scheduler.name' is required")
    scheduler_name = scheduler_config["name"].lower()
    if scheduler_name not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler name: '{scheduler_name}'. Supported schedulers: {sorted(SUPPORTED_SCHEDULERS)}"
        )
    _validate_scheduler_params(scheduler_name, scheduler_config.get("params", {}))

    # Validate optimizer (required)
    if "optimizer" not in config:
        raise ValueError("'optimizer' configuration is required")
    _validate_optimizer_config(config["optimizer"])

    # Validate augmentation config if present
    if "crop_weights" in config:
        _validate_augmentation_config(config)


def create_config(
    *,
    # Required - must be provided by experiment config
    exp_name: str,
    model: dict,
    loss: dict,
    scheduler: dict,
    optimizer: dict,
    batch_size: int,
    num_epochs: int,
    # Optional - experiment can override
    train_size: int | None = None,
    valid_size: int | None = None,
    # Infrastructure defaults
    registry_dir: str = "./experiments",
    experiments_dir: str = "./experiments/train_nn1_cropped",
    crop_size: int = 256,
    flip_prob: float = 0.5,
    num_workers: int = 2,
    valid_batch_size: int | None = None,
    valid_num_workers: int = 0,
    valid_pin_memory: bool = False,
    valid_prefetch_factor: int = 1,
    early_stopping_patience: int = 7,
    early_stopping_delta: float = 0.0001,
    # Dataset augmentation config (all configurable)
    crop_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0),
    include_resize_512: bool = True,
    resize_512_weight: float = 0.5,
    include_resize_384: bool = False,
    resize_384_weight: float = 0.5,
    rotation_prob: float = 0.5,
    rotation_max_angle: float = 15.0,
    # Test visualization flags
    generate_test_viz_real: bool = True,
    generate_test_viz_interpolated: bool = False,
    # Optional weight initialization from another experiment
    init_from_experiment: str | None = None,
    init_from_checkpoint: str = "latest_epoch.pth",
) -> dict:
    """Create a configuration dictionary from experiment parameters.

    Args:
        exp_name: Experiment name
        model: Model configuration dict (type, encoder_name, pretrained)
        loss: Loss configuration dict (name, params)
        scheduler: Scheduler configuration dict (name, params)
        optimizer: Optimizer configuration dict (name, params with lr)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        train_size: Training dataset size (None for full dataset)
        valid_size: Validation dataset size (None for full dataset)
        registry_dir: Directory for experiments registry
        experiments_dir: Directory for experiment outputs
        crop_size: Crop size for augmentation
        flip_prob: Flip probability for augmentation
        num_workers: DataLoader workers
        valid_batch_size: Validation batch size (defaults to train batch size if None)
        valid_num_workers: Validation DataLoader workers
        valid_pin_memory: Pin validation batches in host memory
        valid_prefetch_factor: Prefetched batches per validation worker (when workers > 0)
        early_stopping_patience: Early stopping patience
        early_stopping_delta: Early stopping minimum delta
        crop_weights: Weight per crop position (9 values, center=index 4 has 2x default)
        include_resize_512: Include 512→256 resize option in training
        resize_512_weight: Weight for 512→256 resize option
        include_resize_384: Include 384→256 resize option in training
        resize_384_weight: Weight for 384→256 resize option
        rotation_prob: Probability of applying rotation (0-1)
        rotation_max_angle: Max rotation angle in degrees (uniform in [-max, +max])
        generate_test_viz_real: Generate test visualizations for target_is_real mode
        generate_test_viz_interpolated: Generate test visualizations for target_is_interpolated mode
        init_from_experiment: Optional source experiment name for model weight initialization
        init_from_checkpoint: Relative checkpoint path under source experiment directory
            (default: latest_epoch.pth)

    Returns:
        Complete configuration dictionary

    Raises:
        ValueError: If any configuration value is invalid
    """
    config = {
        "exp_name": exp_name,
        "model": model,
        "loss": loss,
        "scheduler": scheduler,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "train_size": train_size,
        "valid_size": valid_size,
        "registry_dir": registry_dir,
        "experiments_dir": experiments_dir,
        "data_path": get_dataset_dir(),
        "crop_size": crop_size,
        "flip_prob": flip_prob,
        "num_workers": num_workers,
        "valid_batch_size": valid_batch_size,
        "valid_num_workers": valid_num_workers,
        "valid_pin_memory": valid_pin_memory,
        "valid_prefetch_factor": valid_prefetch_factor,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_delta": early_stopping_delta,
        # Augmentation config
        "crop_weights": crop_weights,
        "include_resize_512": include_resize_512,
        "resize_512_weight": resize_512_weight,
        "include_resize_384": include_resize_384,
        "resize_384_weight": resize_384_weight,
        "rotation_prob": rotation_prob,
        "rotation_max_angle": rotation_max_angle,
        "generate_test_viz_real": generate_test_viz_real,
        "generate_test_viz_interpolated": generate_test_viz_interpolated,
        "init_from_experiment": init_from_experiment,
        "init_from_checkpoint": init_from_checkpoint if init_from_experiment else None,
    }

    validate_config(config)
    return config
