"""Training module for CT slice interpolation models.

This module provides:
- Trainer: Main class for running training experiments
- Configuration utilities for setting up experiments
- Registry for tracking all experiments

Heavy imports (torch, etc.) are deferred until actually needed.
"""

# Light imports (no torch dependency)
from phd.training.config import create_config, get_dataset_dir
from phd.training.experiments import generate_experiment_name
from phd.training.registry import (
    get_experiment_status,
    list_experiments,
    register_experiment,
    update_experiment_status,
)
from phd.training.status import TrainingStatus


def __getattr__(name: str) -> type:
    """Lazy import heavy modules only when accessed."""
    if name == "Trainer":
        from phd.training.trainer import Trainer

        return Trainer
    if name == "EarlyStopping":
        from phd.training.early_stopping import EarlyStopping

        return EarlyStopping
    if name == "load_checkpoint":
        from phd.training.checkpoint import load_checkpoint

        return load_checkpoint
    if name == "save_checkpoint":
        from phd.training.checkpoint import save_checkpoint

        return save_checkpoint
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "EarlyStopping",  # noqa: F822
    "Trainer",  # noqa: F822
    "TrainingStatus",
    "create_config",
    "generate_experiment_name",
    "get_dataset_dir",
    "get_experiment_status",
    "list_experiments",
    "load_checkpoint",  # noqa: F822
    "register_experiment",
    "save_checkpoint",  # noqa: F822
    "update_experiment_status",
]
