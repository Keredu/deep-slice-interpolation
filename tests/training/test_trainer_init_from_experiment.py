"""Tests for transfer initialization from another experiment checkpoint."""

from pathlib import Path

import pytest
import torch

from phd.training.trainer import Trainer

from .conftest import SimpleModel


class TestInitFromExperimentCheckpoint:
    """Tests for Trainer._initialize_from_experiment_checkpoint."""

    @staticmethod
    def _save_source_checkpoint(path: Path, model: SimpleModel, epoch: int = 3) -> None:
        """Save a minimal source checkpoint with model weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            },
            path,
        )

    def test_loads_weights_from_source_experiment(self, minimal_config: dict, tmp_path: Path) -> None:
        """Model parameters should be initialized from source checkpoint."""
        source_exp = "source_exp"
        experiments_dir = tmp_path / "experiments"
        source_ckpt = experiments_dir / source_exp / "latest_epoch.pth"

        source_model = SimpleModel()
        for param in source_model.parameters():
            torch.nn.init.constant_(param, 0.42)
        self._save_source_checkpoint(source_ckpt, source_model, epoch=7)

        config = {
            **minimal_config,
            "experiments_dir": str(experiments_dir),
            "init_from_experiment": source_exp,
            "init_from_checkpoint": "latest_epoch.pth",
        }
        trainer = Trainer(config)
        trainer.model = SimpleModel()

        # Ensure target model starts from different weights.
        for param in trainer.model.parameters():
            torch.nn.init.constant_(param, -0.5)

        trainer._initialize_from_experiment_checkpoint()

        for target_param, source_param in zip(
            trainer.model.state_dict().values(),
            source_model.state_dict().values(),
            strict=True,
        ):
            assert torch.allclose(target_param, source_param)

    def test_raises_when_checkpoint_missing(self, minimal_config: dict, tmp_path: Path) -> None:
        """Missing source checkpoint should raise FileNotFoundError."""
        config = {
            **minimal_config,
            "experiments_dir": str(tmp_path / "experiments"),
            "init_from_experiment": "missing_exp",
            "init_from_checkpoint": "latest_epoch.pth",
        }
        trainer = Trainer(config)
        trainer.model = SimpleModel()

        with pytest.raises(FileNotFoundError, match="Initialization checkpoint not found"):
            trainer._initialize_from_experiment_checkpoint()

    def test_accepts_prefixed_keys_from_compiled_checkpoint(self, minimal_config: dict, tmp_path: Path) -> None:
        """Compiled-model prefixes should be adapted automatically."""
        source_exp = "compiled_source"
        experiments_dir = tmp_path / "experiments"
        source_ckpt = experiments_dir / source_exp / "latest_epoch.pth"

        source_model = SimpleModel()
        for param in source_model.parameters():
            torch.nn.init.constant_(param, 0.13)
        prefixed_state_dict = {f"_orig_mod.{k}": v for k, v in source_model.state_dict().items()}
        source_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": 2, "model_state_dict": prefixed_state_dict}, source_ckpt)

        config = {
            **minimal_config,
            "experiments_dir": str(experiments_dir),
            "init_from_experiment": source_exp,
            "init_from_checkpoint": "latest_epoch.pth",
        }
        trainer = Trainer(config)
        trainer.model = SimpleModel()

        trainer._initialize_from_experiment_checkpoint()

        for target_param, source_param in zip(
            trainer.model.state_dict().values(),
            source_model.state_dict().values(),
            strict=True,
        ):
            assert torch.allclose(target_param, source_param)

