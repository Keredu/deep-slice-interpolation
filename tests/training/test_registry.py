"""Tests for experiment registry functionality."""

import json
from pathlib import Path

import pytest

from phd.training.registry import (
    _get_registry_path,
    _load_registry,
    _save_registry,
    get_experiment_status,
    get_next_experiment,
    get_queue_status,
    list_experiments,
    queue_experiment,
    register_experiment,
    reset_error_experiments,
    update_experiment_status,
)
from phd.training.status import TrainingStatus


@pytest.fixture
def sample_config() -> dict:
    """Return a sample experiment config."""
    return {
        "model": {"type": "unet", "encoder_name": "resnet18"},
        "loss": {"name": "ssim"},
        "scheduler": {"name": "cosine"},
        "optimizer": {"name": "adamw", "params": {"lr": 0.001}},
        "batch_size": 64,
        "num_epochs": 100,
        "train_size": 1000,
        "valid_size": 200,
    }


class TestRegistryPath:
    """Tests for registry path functions."""

    def test_get_registry_path(self, tmp_path: Path) -> None:
        """Test getting registry path."""
        result = _get_registry_path(tmp_path)
        assert result == tmp_path / "experiments_registry.json"


class TestLoadRegistry:
    """Tests for _load_registry function."""

    def test_load_empty_registry_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test loading when no registry file exists."""
        result = _load_registry(tmp_path)
        assert result == {}

    def test_load_existing_registry(self, tmp_path: Path) -> None:
        """Test loading existing registry."""
        registry_path = tmp_path / "experiments_registry.json"
        data = {"exp1": {"status": "RUNNING", "runs": 1}}
        registry_path.write_text(json.dumps(data))

        result = _load_registry(tmp_path)
        assert result == data


class TestSaveRegistry:
    """Tests for _save_registry function."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Test that saving creates the registry file."""
        data = {"exp1": {"status": "RUNNING"}}
        _save_registry(tmp_path, data)

        registry_path = tmp_path / "experiments_registry.json"
        assert registry_path.exists()

        loaded = json.loads(registry_path.read_text())
        assert loaded == data

    def test_save_creates_directory_if_needed(self, tmp_path: Path) -> None:
        """Test that saving creates parent directories."""
        nested_dir = tmp_path / "nested" / "dir"
        data = {"exp1": {"status": "RUNNING"}}
        _save_registry(nested_dir, data)

        assert (nested_dir / "experiments_registry.json").exists()


class TestRegisterExperiment:
    """Tests for register_experiment function."""

    def test_register_new_experiment(self, tmp_path: Path, sample_config: dict) -> None:
        """Test registering a new experiment."""
        register_experiment(tmp_path, "exp1", sample_config)

        registry = _load_registry(tmp_path)
        assert "exp1" in registry
        assert registry["exp1"]["status"] == TrainingStatus.RUNNING.value
        assert registry["exp1"]["runs"] == 1
        assert registry["exp1"]["config"]["loss"] == {"name": "ssim"}

    def test_register_existing_experiment_updates_status(self, tmp_path: Path, sample_config: dict) -> None:
        """Test re-registering updates existing experiment."""
        # First registration
        register_experiment(tmp_path, "exp1", sample_config)

        # Second registration
        register_experiment(tmp_path, "exp1", sample_config)

        registry = _load_registry(tmp_path)
        assert registry["exp1"]["runs"] == 2
        assert registry["exp1"]["status"] == TrainingStatus.RUNNING.value


class TestUpdateExperimentStatus:
    """Tests for update_experiment_status function."""

    def test_update_existing_experiment(self, tmp_path: Path, sample_config: dict) -> None:
        """Test updating status of existing experiment."""
        register_experiment(tmp_path, "exp1", sample_config)
        update_experiment_status(tmp_path, "exp1", TrainingStatus.FINISHED_EPOCHS)

        registry = _load_registry(tmp_path)
        assert registry["exp1"]["status"] == TrainingStatus.FINISHED_EPOCHS.value
        assert "finished" in registry["exp1"]

    def test_update_nonexistent_experiment_creates_entry(self, tmp_path: Path) -> None:
        """Test updating non-existent experiment creates minimal entry."""
        update_experiment_status(tmp_path, "exp_new", TrainingStatus.ERROR)

        registry = _load_registry(tmp_path)
        assert "exp_new" in registry
        assert registry["exp_new"]["status"] == TrainingStatus.ERROR.value

    def test_update_with_best_loss(self, tmp_path: Path, sample_config: dict) -> None:
        """Test updating with best validation loss."""
        register_experiment(tmp_path, "exp1", sample_config)
        update_experiment_status(tmp_path, "exp1", TrainingStatus.FINISHED_EPOCHS, best_valid_loss=0.123)

        registry = _load_registry(tmp_path)
        assert registry["exp1"]["best_valid_loss"] == 0.123

    def test_update_with_final_epoch(self, tmp_path: Path, sample_config: dict) -> None:
        """Test updating with final epoch."""
        register_experiment(tmp_path, "exp1", sample_config)
        update_experiment_status(tmp_path, "exp1", TrainingStatus.EARLY_STOPPING, final_epoch=50)

        registry = _load_registry(tmp_path)
        assert registry["exp1"]["final_epoch"] == 50
        assert "finished" in registry["exp1"]


class TestGetExperimentStatus:
    """Tests for get_experiment_status function."""

    def test_get_existing_status(self, tmp_path: Path, sample_config: dict) -> None:
        """Test getting status of existing experiment."""
        register_experiment(tmp_path, "exp1", sample_config)

        status = get_experiment_status(tmp_path, "exp1")
        assert status == TrainingStatus.RUNNING

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Test getting status of non-existent experiment."""
        status = get_experiment_status(tmp_path, "nonexistent")
        assert status is None


class TestListExperiments:
    """Tests for list_experiments function."""

    def test_list_empty_returns_empty(self, tmp_path: Path) -> None:
        """Test listing when no experiments exist."""
        result = list_experiments(tmp_path)
        assert result == {}

    def test_list_returns_all_experiments(self, tmp_path: Path, sample_config: dict) -> None:
        """Test listing all experiments."""
        register_experiment(tmp_path, "exp1", sample_config)
        register_experiment(tmp_path, "exp2", sample_config)

        result = list_experiments(tmp_path)
        assert "exp1" in result
        assert "exp2" in result


class TestQueueExperiment:
    """Tests for queue_experiment function."""

    def test_queue_new_experiment(self, tmp_path: Path, sample_config: dict) -> None:
        """Test queuing a new experiment."""
        result = queue_experiment(tmp_path, "exp1", sample_config)

        assert result is True
        registry = _load_registry(tmp_path)
        assert registry["exp1"]["status"] == TrainingStatus.NOT_STARTED.value
        assert registry["exp1"]["runs"] == 0

    def test_queue_existing_experiment_returns_false(self, tmp_path: Path, sample_config: dict) -> None:
        """Test queuing existing experiment returns False."""
        queue_experiment(tmp_path, "exp1", sample_config)
        result = queue_experiment(tmp_path, "exp1", sample_config)

        assert result is False


class TestGetNextExperiment:
    """Tests for get_next_experiment function."""

    def test_empty_registry_returns_none(self, tmp_path: Path) -> None:
        """Test that empty registry returns None."""
        result = get_next_experiment(tmp_path)
        assert result is None

    def test_running_experiment_has_priority(self, tmp_path: Path, sample_config: dict) -> None:
        """Test that RUNNING experiments have priority over NOT_STARTED."""
        queue_experiment(tmp_path, "exp_queued", sample_config)
        register_experiment(tmp_path, "exp_running", sample_config)

        result = get_next_experiment(tmp_path)
        assert result == "exp_running"

    def test_not_started_returned_when_no_running(self, tmp_path: Path, sample_config: dict) -> None:
        """Test that NOT_STARTED is returned when no RUNNING."""
        queue_experiment(tmp_path, "exp1", sample_config)
        queue_experiment(tmp_path, "exp2", sample_config)

        result = get_next_experiment(tmp_path)
        # Should return oldest queued (exp1)
        assert result == "exp1"

    def test_finished_experiments_not_returned(self, tmp_path: Path, sample_config: dict) -> None:
        """Test that FINISHED experiments are not returned."""
        register_experiment(tmp_path, "exp1", sample_config)
        update_experiment_status(tmp_path, "exp1", TrainingStatus.FINISHED_EPOCHS)

        result = get_next_experiment(tmp_path)
        assert result is None


class TestGetQueueStatus:
    """Tests for get_queue_status function."""

    def test_empty_registry_all_zeros(self, tmp_path: Path) -> None:
        """Test that empty registry returns all zero counts."""
        result = get_queue_status(tmp_path)

        assert result[TrainingStatus.NOT_STARTED.value] == 0
        assert result[TrainingStatus.RUNNING.value] == 0
        assert result[TrainingStatus.FINISHED_EPOCHS.value] == 0

    def test_counts_correct(self, tmp_path: Path, sample_config: dict) -> None:
        """Test that counts are correct."""
        queue_experiment(tmp_path, "exp1", sample_config)
        queue_experiment(tmp_path, "exp2", sample_config)
        register_experiment(tmp_path, "exp3", sample_config)
        register_experiment(tmp_path, "exp4", sample_config)
        update_experiment_status(tmp_path, "exp4", TrainingStatus.FINISHED_EPOCHS)

        result = get_queue_status(tmp_path)

        assert result[TrainingStatus.NOT_STARTED.value] == 2
        assert result[TrainingStatus.RUNNING.value] == 1
        assert result[TrainingStatus.FINISHED_EPOCHS.value] == 1


class TestResetErrorExperiments:
    """Tests for reset_error_experiments function."""

    def test_reset_error_to_running(self, tmp_path: Path, sample_config: dict) -> None:
        """Test resetting ERROR experiments to RUNNING for priority resumption."""
        register_experiment(tmp_path, "exp1", sample_config)
        update_experiment_status(tmp_path, "exp1", TrainingStatus.ERROR)

        reset = reset_error_experiments(tmp_path)

        assert reset == ["exp1"]
        registry = _load_registry(tmp_path)
        assert registry["exp1"]["status"] == TrainingStatus.RUNNING.value

    def test_no_error_experiments_returns_empty(self, tmp_path: Path, sample_config: dict) -> None:
        """Test that no ERROR experiments returns empty list."""
        queue_experiment(tmp_path, "exp1", sample_config)

        reset = reset_error_experiments(tmp_path)

        assert reset == []

    def test_cleans_up_error_files(self, tmp_path: Path, sample_config: dict) -> None:
        """Test that error files are cleaned up."""
        output_dir = tmp_path / "output"
        exp_dir = output_dir / "exp1"
        exp_dir.mkdir(parents=True)

        # Create error files
        (exp_dir / "error.log").write_text("error")
        (exp_dir / "stderr.log").write_text("stderr")

        register_experiment(tmp_path, "exp1", sample_config)
        update_experiment_status(tmp_path, "exp1", TrainingStatus.ERROR)

        reset_error_experiments(tmp_path, output_dir)

        assert not (exp_dir / "error.log").exists()
        assert not (exp_dir / "stderr.log").exists()
