"""Tests for phd.training module initialization and lazy imports."""

import pytest


class TestLazyImports:
    """Tests for lazy import functionality in phd.training."""

    def test_import_trainer(self) -> None:
        """Test lazy import of Trainer class."""
        from phd.training import Trainer
        from phd.training.trainer import Trainer as DirectTrainer

        assert Trainer is DirectTrainer

    def test_import_early_stopping(self) -> None:
        """Test lazy import of EarlyStopping class."""
        from phd.training import EarlyStopping
        from phd.training.early_stopping import EarlyStopping as DirectEarlyStopping

        assert EarlyStopping is DirectEarlyStopping

    def test_import_load_checkpoint(self) -> None:
        """Test lazy import of load_checkpoint function."""
        from phd.training import load_checkpoint
        from phd.training.checkpoint import load_checkpoint as direct_load

        assert load_checkpoint is direct_load

    def test_import_save_checkpoint(self) -> None:
        """Test lazy import of save_checkpoint function."""
        from phd.training import save_checkpoint
        from phd.training.checkpoint import save_checkpoint as direct_save

        assert save_checkpoint is direct_save

    def test_import_unknown_attribute_raises(self) -> None:
        """Test that importing unknown attribute raises AttributeError."""
        import phd.training as training_module

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = training_module.unknown_attribute


class TestDirectImports:
    """Tests for direct (non-lazy) imports from phd.training."""

    def test_import_create_config(self) -> None:
        """Test direct import of create_config."""
        from phd.training import create_config
        from phd.training.config import create_config as direct_create_config

        assert create_config is direct_create_config

    def test_import_get_dataset_dir(self) -> None:
        """Test direct import of get_dataset_dir."""
        from phd.training import get_dataset_dir
        from phd.training.config import get_dataset_dir as direct_get_dataset_dir

        assert get_dataset_dir is direct_get_dataset_dir

    def test_import_generate_experiment_name(self) -> None:
        """Test direct import of generate_experiment_name."""
        from phd.training import generate_experiment_name
        from phd.training.experiments import (
            generate_experiment_name as direct_generate,
        )

        assert generate_experiment_name is direct_generate

    def test_import_training_status(self) -> None:
        """Test direct import of TrainingStatus."""
        from phd.training import TrainingStatus
        from phd.training.status import TrainingStatus as DirectTrainingStatus

        assert TrainingStatus is DirectTrainingStatus

    def test_import_registry_functions(self) -> None:
        """Test direct import of registry functions."""
        from phd.training import (
            get_experiment_status,
            list_experiments,
            register_experiment,
            update_experiment_status,
        )
        from phd.training.registry import (
            get_experiment_status as direct_get,
        )
        from phd.training.registry import (
            list_experiments as direct_list,
        )
        from phd.training.registry import (
            register_experiment as direct_register,
        )
        from phd.training.registry import (
            update_experiment_status as direct_update,
        )

        assert get_experiment_status is direct_get
        assert list_experiments is direct_list
        assert register_experiment is direct_register
        assert update_experiment_status is direct_update


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_contains_expected_names(self) -> None:
        """Test that __all__ contains all expected exports."""
        from phd.training import __all__

        expected = [
            "EarlyStopping",
            "Trainer",
            "TrainingStatus",
            "create_config",
            "generate_experiment_name",
            "get_dataset_dir",
            "get_experiment_status",
            "list_experiments",
            "load_checkpoint",
            "register_experiment",
            "save_checkpoint",
            "update_experiment_status",
        ]
        for name in expected:
            assert name in __all__, f"{name} not in __all__"
