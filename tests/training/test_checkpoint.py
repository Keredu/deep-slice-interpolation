"""Tests for checkpoint save/load utilities."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from phd.training.checkpoint import load_checkpoint, save_checkpoint
from phd.training.early_stopping import EarlyStopping


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.fc(x)


@pytest.fixture
def model() -> SimpleModel:
    """Return a simple model."""
    return SimpleModel()


@pytest.fixture
def optimizer(model: SimpleModel) -> Adam:
    """Return an optimizer."""
    return Adam(model.parameters(), lr=0.001)


@pytest.fixture
def scheduler(optimizer: Adam) -> StepLR:
    """Return a scheduler."""
    return StepLR(optimizer, step_size=10, gamma=0.1)


@pytest.fixture
def early_stopping() -> EarlyStopping:
    """Return early stopping instance."""
    es = EarlyStopping(patience=5, min_delta=0.001)
    es.counter = 2
    es.best_loss = 0.5
    return es


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_save_checkpoint_creates_file(
        self,
        tmp_path: Path,
        model: SimpleModel,
        optimizer: Adam,
        scheduler: StepLR,
        early_stopping: EarlyStopping,
    ) -> None:
        """Test that save_checkpoint creates a file."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            path=checkpoint_path,
            epoch=10,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=0.1,
            valid_loss=0.2,
            best_valid_loss=0.18,
            train_losses=[0.5, 0.3, 0.1],
            valid_losses=[0.6, 0.4, 0.2],
            metric_histories={"ssim": [0.8, 0.9, 0.95]},
            config={"learning_rate": 0.001},
            early_stopping=early_stopping,
            best_epochs=[False, True, False],
            per_crop_metric_histories={},
            learning_rates=[0.001, 0.001, 0.001],
            epoch_times=[100.0, 95.0, 90.0],
            train_times=[30.0, 28.0, 27.0],
            valid_times=[70.0, 67.0, 63.0],
        )

        assert checkpoint_path.exists()

    def test_save_checkpoint_content(
        self,
        tmp_path: Path,
        model: SimpleModel,
        optimizer: Adam,
        scheduler: StepLR,
        early_stopping: EarlyStopping,
    ) -> None:
        """Test that checkpoint contains expected data."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            path=checkpoint_path,
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=0.1,
            valid_loss=0.2,
            best_valid_loss=0.18,
            train_losses=[0.5, 0.3],
            valid_losses=[0.6, 0.4],
            metric_histories={"psnr": [30.0, 35.0]},
            config={"batch_size": 64},
            early_stopping=early_stopping,
            best_epochs=[True, False],
            per_crop_metric_histories={},
            learning_rates=[0.001, 0.0008],
            epoch_times=[120.5, 115.3],
            train_times=[30.0, 28.0],
            valid_times=[90.5, 87.3],
        )

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["epoch"] == 5
        assert checkpoint["train_loss"] == 0.1
        assert checkpoint["valid_loss"] == 0.2
        assert checkpoint["train_losses"] == [0.5, 0.3]
        assert checkpoint["metric_histories"] == {"psnr": [30.0, 35.0]}
        assert checkpoint["config"] == {"batch_size": 64}
        assert checkpoint["early_stopping_counter"] == 2
        assert checkpoint["early_stopping_best_loss"] == 0.5
        assert checkpoint["best_epochs"] == [True, False]
        assert checkpoint["learning_rates"] == [0.001, 0.0008]
        assert checkpoint["epoch_times"] == [120.5, 115.3]


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_restores_state(
        self,
        tmp_path: Path,
        model: SimpleModel,
        optimizer: Adam,
        scheduler: StepLR,
        early_stopping: EarlyStopping,
    ) -> None:
        """Test that load_checkpoint restores model and optimizer state."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Save checkpoint
        save_checkpoint(
            path=checkpoint_path,
            epoch=7,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=0.1,
            valid_loss=0.2,
            best_valid_loss=0.18,
            train_losses=[0.5, 0.3, 0.1],
            valid_losses=[0.6, 0.4, 0.2],
            metric_histories={"mae": [0.1, 0.05]},
            config={"num_epochs": 100},
            early_stopping=early_stopping,
            best_epochs=[True, False, True],
            per_crop_metric_histories={},
            learning_rates=[0.001, 0.001, 0.001],
            epoch_times=[100.0, 95.0, 90.0],
            train_times=[30.0, 28.0, 27.0],
            valid_times=[70.0, 67.0, 63.0],
        )

        # Create new instances
        new_model = SimpleModel()
        new_optimizer = Adam(new_model.parameters(), lr=0.01)  # Different lr
        new_scheduler = StepLR(new_optimizer, step_size=10, gamma=0.1)
        new_early_stopping = EarlyStopping(patience=5)

        # Load checkpoint
        result = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            early_stopping=new_early_stopping,
        )

        assert result["epoch"] == 7
        assert result["train_losses"] == [0.5, 0.3, 0.1]
        assert result["valid_losses"] == [0.6, 0.4, 0.2]
        assert result["best_valid_loss"] == 0.18
        assert result["metric_histories"] == {"mae": [0.1, 0.05]}
        assert result["best_epochs"] == [True, False, True]

        # Check early stopping was restored
        assert new_early_stopping.counter == 2
        assert new_early_stopping.best_loss == 0.5

    def test_load_checkpoint_without_scheduler(
        self,
        tmp_path: Path,
        model: SimpleModel,
        optimizer: Adam,
        scheduler: StepLR,
        early_stopping: EarlyStopping,
    ) -> None:
        """Test loading checkpoint without providing scheduler."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            path=checkpoint_path,
            epoch=3,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=0.1,
            valid_loss=0.2,
            best_valid_loss=0.18,
            train_losses=[0.5],
            valid_losses=[0.6],
            metric_histories={},
            config={},
            early_stopping=early_stopping,
            best_epochs=[True],
            per_crop_metric_histories={},
            learning_rates=[0.001],
            epoch_times=[100.0],
            train_times=[30.0],
            valid_times=[70.0],
        )

        new_model = SimpleModel()
        new_optimizer = Adam(new_model.parameters())

        # Load without scheduler
        result = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=None,
            early_stopping=None,
        )

        assert result["epoch"] == 3

    def test_load_checkpoint_missing_optional_fields(
        self,
        tmp_path: Path,
        model: SimpleModel,
        optimizer: Adam,
        scheduler: StepLR,
        early_stopping: EarlyStopping,
    ) -> None:
        """Test loading checkpoint that may be missing optional fields."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Create minimal checkpoint manually
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": 0.1,
                "valid_loss": 0.2,
                "best_valid_loss": 0.18,
                "train_losses": [],
                "valid_losses": [],
                "early_stopping_counter": 0,
                "early_stopping_best_loss": float("inf"),
                # Missing: metric_histories, best_epochs
            },
            checkpoint_path,
        )

        new_model = SimpleModel()
        new_optimizer = Adam(new_model.parameters())

        result = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
        )

        # Should use defaults for missing fields
        assert result["metric_histories"] == {}
        assert result["best_epochs"] == []
