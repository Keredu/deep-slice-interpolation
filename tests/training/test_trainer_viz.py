"""Tests for Trainer._save_visualizations method."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from phd.training.trainer import Trainer


class TestSaveVisualizations:
    """Tests for _save_visualizations method."""

    @pytest.fixture
    def trainer_for_viz(self, minimal_config: dict, tmp_path: Path) -> Trainer:
        """Return a trainer configured for visualization tests."""
        trainer = Trainer(minimal_config)
        trainer.experiment_dir = tmp_path / "experiments" / minimal_config["exp_name"]
        trainer.experiment_dir.mkdir(parents=True)
        trainer.epochs_dir = trainer.experiment_dir / "epochs"
        trainer.epochs_dir.mkdir(parents=True)
        trainer.device = torch.device("cpu")
        trainer.model = MagicMock()
        return trainer

    def test_creates_epoch_and_viz_dirs(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that epoch and viz directories are created."""
        trainer_for_viz.config["generate_test_viz_real"] = True
        trainer_for_viz.test_dataset_target_is_real = MagicMock()

        with patch("phd.training.trainer.save_test_visualization"):
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=5)

        epoch_dir = trainer_for_viz.epochs_dir / "5"
        viz_dir = epoch_dir / "viz"

        assert epoch_dir.exists()
        assert viz_dir.exists()

    def test_skips_when_both_disabled(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that visualization is skipped when both options disabled."""
        trainer_for_viz.config["generate_test_viz_real"] = False
        trainer_for_viz.config["generate_test_viz_interpolated"] = False

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache") as mock_cache:
                trainer_for_viz._save_visualizations(epoch=5)

        # Should not call visualization or cache clear
        mock_viz.assert_not_called()
        mock_cache.assert_not_called()

        # Should not create directories
        epoch_dir = trainer_for_viz.epochs_dir / "5"
        assert not epoch_dir.exists()

    def test_generates_real_viz_when_enabled(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that real visualization is generated when enabled."""
        trainer_for_viz.config["generate_test_viz_real"] = True
        trainer_for_viz.config["generate_test_viz_interpolated"] = False
        mock_dataset = MagicMock()
        trainer_for_viz.test_dataset_target_is_real = mock_dataset

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=3)

        mock_viz.assert_called_once()
        call_kwargs = mock_viz.call_args[1]
        assert call_kwargs["test_dataset"] == mock_dataset
        assert call_kwargs["model"] == trainer_for_viz.model
        assert call_kwargs["device"] == trainer_for_viz.device
        assert "target_is_real" in str(call_kwargs["save_dir"])

    def test_generates_interpolated_viz_when_enabled(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that interpolated visualization is generated when enabled."""
        trainer_for_viz.config["generate_test_viz_real"] = False
        trainer_for_viz.config["generate_test_viz_interpolated"] = True
        mock_dataset = MagicMock()
        trainer_for_viz.test_dataset_target_is_interpolated = mock_dataset

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=7)

        mock_viz.assert_called_once()
        call_kwargs = mock_viz.call_args[1]
        assert call_kwargs["test_dataset"] == mock_dataset
        assert "target_is_interpolated" in str(call_kwargs["save_dir"])

    def test_generates_both_viz_when_both_enabled(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that both visualizations are generated when both enabled."""
        trainer_for_viz.config["generate_test_viz_real"] = True
        trainer_for_viz.config["generate_test_viz_interpolated"] = True
        mock_real_dataset = MagicMock()
        mock_interp_dataset = MagicMock()
        trainer_for_viz.test_dataset_target_is_real = mock_real_dataset
        trainer_for_viz.test_dataset_target_is_interpolated = mock_interp_dataset

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=10)

        assert mock_viz.call_count == 2

    def test_clears_cuda_cache(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that CUDA cache is cleared before visualization."""
        trainer_for_viz.config["generate_test_viz_real"] = True
        trainer_for_viz.test_dataset_target_is_real = MagicMock()

        with patch("phd.training.trainer.save_test_visualization"):
            with patch("torch.cuda.empty_cache") as mock_cache:
                trainer_for_viz._save_visualizations(epoch=1)

        mock_cache.assert_called_once()

    def test_skips_real_viz_when_dataset_none(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that real viz is skipped when dataset is None."""
        trainer_for_viz.config["generate_test_viz_real"] = True
        trainer_for_viz.config["generate_test_viz_interpolated"] = False
        trainer_for_viz.test_dataset_target_is_real = None

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=1)

        mock_viz.assert_not_called()

    def test_skips_interpolated_viz_when_dataset_none(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that interpolated viz is skipped when dataset is None."""
        trainer_for_viz.config["generate_test_viz_real"] = False
        trainer_for_viz.config["generate_test_viz_interpolated"] = True
        trainer_for_viz.test_dataset_target_is_interpolated = None

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=1)

        mock_viz.assert_not_called()

    def test_passes_batch_size_to_visualization(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test that batch_size is passed to save_test_visualization."""
        trainer_for_viz.config["generate_test_viz_real"] = True
        trainer_for_viz.config["batch_size"] = 16
        trainer_for_viz.test_dataset_target_is_real = MagicMock()

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=1)

        call_kwargs = mock_viz.call_args[1]
        assert call_kwargs["batch_size"] == 16

    def test_uses_default_viz_settings(
        self, trainer_for_viz: Trainer
    ) -> None:
        """Test default visualization config values."""
        # Remove explicit settings to test defaults
        trainer_for_viz.config.pop("generate_test_viz_real", None)
        trainer_for_viz.config.pop("generate_test_viz_interpolated", None)
        trainer_for_viz.test_dataset_target_is_real = MagicMock()

        with patch("phd.training.trainer.save_test_visualization") as mock_viz:
            with patch("torch.cuda.empty_cache"):
                trainer_for_viz._save_visualizations(epoch=1)

        # Default for generate_test_viz_real is True
        mock_viz.assert_called_once()
