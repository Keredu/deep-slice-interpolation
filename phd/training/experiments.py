"""Experiment naming utilities."""

import hashlib
import json


def _config_to_canonical_string(config: dict) -> str:
    """Convert config to a canonical string for hashing.

    Only includes fields that affect experiment behavior:
    - loss (name + params)
    - optimizer (name + params including lr)
    - scheduler (name + params)
    - batch_size
    - num_epochs
    - early_stopping_patience
    - init_from_experiment / init_from_checkpoint (when provided)

    Excludes: model (constant), train_size/valid_size (can vary),
    num_workers, generate_test_viz_* (don't affect training).
    """
    # Extract only the fields that define a unique experiment
    key_fields = {
        "loss": config.get("loss"),
        "optimizer": config.get("optimizer"),
        "scheduler": config.get("scheduler"),
        "batch_size": config.get("batch_size"),
        "num_epochs": config.get("num_epochs"),
        "early_stopping_patience": config.get("early_stopping_patience"),
    }

    # Include initialization source only when provided.
    # This preserves existing hashes for historical experiments.
    init_from_experiment = config.get("init_from_experiment")
    if init_from_experiment:
        key_fields["init_from_experiment"] = init_from_experiment
        key_fields["init_from_checkpoint"] = config.get("init_from_checkpoint", "latest_epoch.pth")

    # Sort keys for deterministic ordering
    return json.dumps(key_fields, sort_keys=True)


def generate_experiment_name(config: dict) -> str:
    """Generate a unique, deterministic experiment name from config.

    Creates a name based on loss function, learning rate, and a hash of the config.
    The same config always generates the same name, enabling deduplication.

    Args:
        config: Configuration dict with nested model, loss, optimizer dicts

    Returns:
        Experiment name string (e.g., "ssim+l1_lr8e-4_a1b2c3")
    """
    # Use loss name as the primary identifier
    loss_name = config["loss"]["name"]

    # Extract learning rate for readability
    lr = config.get("optimizer", {}).get("params", {}).get("lr", 8e-4)
    lr_str = f"{lr:.0e}".replace(".", "").replace("+", "").replace("-0", "-")

    # Generate deterministic 6-character hash from config
    config_str = _config_to_canonical_string(config)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:6]

    return f"{loss_name}_lr{lr_str}_{config_hash}"
