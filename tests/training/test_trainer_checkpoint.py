"""Tests for Trainer._load_checkpoint method."""

from pathlib import Path
from unittest.mock import patch

import pytest
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from phd.training.checkpoint import save_checkpoint
from phd.training.early_stopping import EarlyStopping
from phd.training.trainer import Trainer

from .conftest import SimpleModel


class TestTrainerLoadCheckpoint:
    """Tests for Trainer._load_checkpoint method."""

    @pytest.fixture
    def trainer_for_checkpoint(
        self, minimal_config: dict, tmp_path: Path, simple_model: SimpleModel
    ) -> Trainer:
        """Return a trainer configured for checkpoint loading tests."""
        trainer = Trainer(minimal_config)
        trainer.experiment_dir = tmp_path / "experiments" / minimal_config["exp_name"]
        trainer.experiment_dir.mkdir(parents=True)

        # Set up model, optimizer, scheduler, early_stopping
        trainer.model = simple_model
        trainer.optimizer = Adam(simple_model.parameters(), lr=0.001)
        trainer.scheduler = StepLR(trainer.optimizer, step_size=10, gamma=0.1)
        trainer.early_stopping = EarlyStopping(patience=5)

        return trainer

    def _save_test_checkpoint(
        self,
        trainer: Trainer,
        epoch: int = 5,
        train_losses: list | None = None,
        valid_losses: list | None = None,
        metric_histories: dict | None = None,
        per_crop_metric_histories: dict | None = None,
        learning_rates: list | None = None,
        epoch_times: list | None = None,
        train_times: list | None = None,
        valid_times: list | None = None,
        best_epochs: list | None = None,
    ) -> Path:
        """Helper to save a test checkpoint."""
        checkpoint_path = trainer.experiment_dir / "latest_epoch.pth"

        save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            train_loss=0.1,
            valid_loss=0.2,
            best_valid_loss=0.15,
            train_losses=train_losses or [0.5, 0.3, 0.1],
            valid_losses=valid_losses or [0.6, 0.4, 0.2],
            metric_histories=metric_histories or {"ssim": [0.8, 0.85, 0.9]},
            config=trainer.config,
            early_stopping=trainer.early_stopping,
            best_epochs=best_epochs or [False, True, False],
            per_crop_metric_histories=per_crop_metric_histories or {},
            learning_rates=learning_rates or [0.001, 0.001, 0.001],
            epoch_times=epoch_times or [100.0, 95.0, 90.0],
            train_times=train_times or [30.0, 28.0, 27.0],
            valid_times=valid_times or [70.0, 67.0, 63.0],
        )
        return checkpoint_path

    def test_loads_start_epoch_incremented(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that start_epoch is set to checkpoint epoch + 1."""
        self._save_test_checkpoint(trainer_for_checkpoint, epoch=7)

        trainer_for_checkpoint._load_checkpoint()

        assert trainer_for_checkpoint.start_epoch == 8

    def test_loads_train_and_valid_losses(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that train and valid loss histories are loaded."""
        train_losses = [0.8, 0.6, 0.4, 0.2]
        valid_losses = [0.9, 0.7, 0.5, 0.3]
        self._save_test_checkpoint(
            trainer_for_checkpoint,
            train_losses=train_losses,
            valid_losses=valid_losses,
        )

        trainer_for_checkpoint._load_checkpoint()

        assert trainer_for_checkpoint.train_losses == train_losses
        assert trainer_for_checkpoint.valid_losses == valid_losses

    def test_loads_best_valid_loss(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that best_valid_loss is loaded from checkpoint."""
        self._save_test_checkpoint(trainer_for_checkpoint)

        # Verify initial state
        assert trainer_for_checkpoint.best_valid_loss == float("inf")

        trainer_for_checkpoint._load_checkpoint()

        # best_valid_loss was set to 0.15 in _save_test_checkpoint
        assert trainer_for_checkpoint.best_valid_loss == 0.15

    def test_loads_metric_histories(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that metric histories are loaded."""
        metric_histories = {
            "ssim": [0.7, 0.8, 0.9],
            "psnr": [25.0, 28.0, 30.0],
            "mae": [0.1, 0.08, 0.05],
        }
        self._save_test_checkpoint(
            trainer_for_checkpoint, metric_histories=metric_histories
        )

        trainer_for_checkpoint._load_checkpoint()

        assert trainer_for_checkpoint.metric_histories == metric_histories

    def test_loads_best_epochs(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that best_epochs list is loaded."""
        best_epochs = [True, False, False, True, False]
        self._save_test_checkpoint(trainer_for_checkpoint, best_epochs=best_epochs)

        trainer_for_checkpoint._load_checkpoint()

        assert trainer_for_checkpoint.best_epochs == best_epochs

    def test_loads_per_crop_histories_if_present(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that per-crop histories are loaded when present."""
        per_crop = {
            0: {"ssim": [0.8, 0.85], "psnr": [25.0, 27.0]},
            4: {"ssim": [0.9, 0.92], "psnr": [28.0, 30.0]},
        }
        self._save_test_checkpoint(
            trainer_for_checkpoint, per_crop_metric_histories=per_crop
        )

        trainer_for_checkpoint._load_checkpoint()

        assert trainer_for_checkpoint.per_crop_metric_histories == per_crop

    def test_loads_timing_histories(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that timing histories are loaded."""
        learning_rates = [0.001, 0.0008, 0.0006]
        epoch_times = [120.0, 115.0, 110.0]
        train_times = [40.0, 38.0, 36.0]
        valid_times = [80.0, 77.0, 74.0]

        self._save_test_checkpoint(
            trainer_for_checkpoint,
            learning_rates=learning_rates,
            epoch_times=epoch_times,
            train_times=train_times,
            valid_times=valid_times,
        )

        trainer_for_checkpoint._load_checkpoint()

        assert trainer_for_checkpoint.learning_rates == learning_rates
        assert trainer_for_checkpoint.epoch_times == epoch_times
        assert trainer_for_checkpoint.train_times == train_times
        assert trainer_for_checkpoint.valid_times == valid_times

    def test_missing_checkpoint_warns(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that missing checkpoint logs a warning."""
        # Don't create checkpoint file

        with patch("phd.training.trainer.logger") as mock_logger:
            trainer_for_checkpoint._load_checkpoint()

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "no checkpoint found" in call_args.lower()

    def test_missing_checkpoint_keeps_defaults(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that missing checkpoint keeps default values."""
        # Don't create checkpoint file

        with patch("phd.training.trainer.logger"):
            trainer_for_checkpoint._load_checkpoint()

        # Should keep initial values
        assert trainer_for_checkpoint.start_epoch == 0
        assert trainer_for_checkpoint.train_losses == []
        assert trainer_for_checkpoint.valid_losses == []
        assert trainer_for_checkpoint.best_valid_loss == float("inf")

    def test_loads_empty_metric_histories_when_present_but_empty(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test handling of empty metric_histories in checkpoint."""
        self._save_test_checkpoint(
            trainer_for_checkpoint,
            epoch=0,
            train_losses=[],
            valid_losses=[],
            metric_histories={},  # Empty but present
        )

        trainer_for_checkpoint._load_checkpoint()

        # Empty dict from checkpoint should not overwrite (conditional check)
        # The code checks `if state["metric_histories"]:` so empty dict won't overwrite
        # This tests that the trainer keeps its initialized structure
        assert "ssim" in trainer_for_checkpoint.metric_histories

    def test_restores_early_stopping_state(
        self, trainer_for_checkpoint: Trainer
    ) -> None:
        """Test that early stopping state is restored."""
        # Set early stopping state before saving
        trainer_for_checkpoint.early_stopping.counter = 3
        trainer_for_checkpoint.early_stopping.best_loss = 0.25

        self._save_test_checkpoint(trainer_for_checkpoint)

        # Reset early stopping
        trainer_for_checkpoint.early_stopping = EarlyStopping(patience=5)

        trainer_for_checkpoint._load_checkpoint()

        assert trainer_for_checkpoint.early_stopping.counter == 3
        assert trainer_for_checkpoint.early_stopping.best_loss == 0.25
