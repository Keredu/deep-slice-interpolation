"""Shared fixtures for training module tests."""

from pathlib import Path

import pytest
import torch
from torch import nn

from phd.training.trainer import Trainer


class SimpleModel(nn.Module):
    """Simple model for testing without GPU requirements."""

    def __init__(self) -> None:
        """Initialize with a single linear layer."""
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.fc(x)


@pytest.fixture
def simple_model() -> SimpleModel:
    """Return a simple model instance."""
    return SimpleModel()


@pytest.fixture
def minimal_config(tmp_path: Path) -> dict:
    """Return a minimal valid config for Trainer initialization.

    This config has all required fields but uses minimal values
    to avoid needing actual datasets or GPU.
    """
    return {
        "exp_name": "test_experiment",
        "experiments_dir": str(tmp_path / "experiments"),
        "registry_dir": str(tmp_path / "registry"),
        "data_path": str(tmp_path / "data"),
        "num_epochs": 10,
        "batch_size": 4,
        "num_workers": 0,
        "train_size": 100,
        "valid_size": 20,
        "flip_prob": 0.5,
        "early_stopping_patience": 5,
        "early_stopping_delta": 0.001,
        "model": {
            "type": "unet",
            "encoder_name": "resnet18",
            "pretrained": False,
        },
        "optimizer": {
            "name": "adamw",
            "params": {
                "lr": 0.001,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
            },
        },
        "scheduler": {
            "name": "none",
        },
        "loss": {
            "name": "mse",
        },
        "generate_test_viz_real": False,
        "generate_test_viz_interpolated": False,
    }


@pytest.fixture
def trainer(minimal_config: dict) -> Trainer:
    """Return a fresh Trainer instance with minimal config.

    Note: This trainer has NOT called setup(), so device, model, etc. are None.
    Use this for testing __init__ behavior and methods that don't require setup.
    """
    return Trainer(minimal_config)


@pytest.fixture
def trainer_with_model(trainer: Trainer, simple_model: SimpleModel) -> Trainer:
    """Return a Trainer instance with a mock model attached.

    Useful for testing methods like _create_optimizer that need self.model.
    """
    trainer.model = simple_model
    trainer.device = torch.device("cpu")
    return trainer


@pytest.fixture
def mock_trainer_for_resume(minimal_config: dict, tmp_path: Path) -> Trainer:
    """Return a Trainer configured for resume status testing.

    Sets up experiment_dir but doesn't call full setup().
    """
    trainer = Trainer(minimal_config)
    trainer.experiment_dir = tmp_path / "experiments" / minimal_config["exp_name"]
    return trainer
