"""Tests for Trainer._save_error_log method."""

from pathlib import Path

import pytest

from phd.training.trainer import Trainer


class TestSaveErrorLog:
    """Tests for _save_error_log method."""

    def test_creates_error_log_file(
        self, trainer: Trainer, tmp_path: Path
    ) -> None:
        """Test that error log file is created."""
        trainer.experiment_dir = tmp_path / "experiment"
        trainer.experiment_dir.mkdir(parents=True)

        error = ValueError("Test error message")

        trainer._save_error_log(error)

        error_log = trainer.experiment_dir / "error.log"
        assert error_log.exists()

    def test_writes_error_message(
        self, trainer: Trainer, tmp_path: Path
    ) -> None:
        """Test that error message is written to log."""
        trainer.experiment_dir = tmp_path / "experiment"
        trainer.experiment_dir.mkdir(parents=True)

        error = ValueError("Specific test error")

        trainer._save_error_log(error)

        error_log = trainer.experiment_dir / "error.log"
        content = error_log.read_text()

        assert "Error: Specific test error" in content

    def test_writes_traceback(
        self, trainer: Trainer, tmp_path: Path
    ) -> None:
        """Test that traceback is written to log."""
        trainer.experiment_dir = tmp_path / "experiment"
        trainer.experiment_dir.mkdir(parents=True)

        # Create an exception with traceback
        try:
            raise RuntimeError("Error with traceback")
        except RuntimeError as e:
            trainer._save_error_log(e)

        error_log = trainer.experiment_dir / "error.log"
        content = error_log.read_text()

        assert "Full traceback:" in content
        assert "RuntimeError: Error with traceback" in content
        assert "raise RuntimeError" in content  # Part of traceback

    def test_no_experiment_dir_does_nothing(self, trainer: Trainer) -> None:
        """Test that None experiment_dir doesn't cause error."""
        trainer.experiment_dir = None

        error = ValueError("Test error")

        # Should not raise
        trainer._save_error_log(error)

    def test_overwrites_existing_error_log(
        self, trainer: Trainer, tmp_path: Path
    ) -> None:
        """Test that existing error log is overwritten."""
        trainer.experiment_dir = tmp_path / "experiment"
        trainer.experiment_dir.mkdir(parents=True)

        error_log = trainer.experiment_dir / "error.log"
        error_log.write_text("Previous error content")

        new_error = ValueError("New error message")
        trainer._save_error_log(new_error)

        content = error_log.read_text()

        assert "Previous error content" not in content
        assert "New error message" in content

    def test_handles_exception_with_no_args(
        self, trainer: Trainer, tmp_path: Path
    ) -> None:
        """Test handling of exception with no message."""
        trainer.experiment_dir = tmp_path / "experiment"
        trainer.experiment_dir.mkdir(parents=True)

        error = ValueError()

        trainer._save_error_log(error)

        error_log = trainer.experiment_dir / "error.log"
        content = error_log.read_text()

        assert "Error:" in content
        assert "Full traceback:" in content

    def test_handles_nested_exception(
        self, trainer: Trainer, tmp_path: Path
    ) -> None:
        """Test handling of nested/chained exceptions."""
        trainer.experiment_dir = tmp_path / "experiment"
        trainer.experiment_dir.mkdir(parents=True)

        try:
            try:
                raise ValueError("Original error")
            except ValueError as inner:
                raise RuntimeError("Outer error") from inner
        except RuntimeError as e:
            trainer._save_error_log(e)

        error_log = trainer.experiment_dir / "error.log"
        content = error_log.read_text()

        assert "Outer error" in content
        # The traceback may include the chained exception depending on Python version
        assert "RuntimeError" in content


class TestValidateEpochRetry:
    """Tests for validation retry logic on SHM DataLoader errors."""

    def test_retries_once_on_shm_error(self, trainer: Trainer, monkeypatch: pytest.MonkeyPatch) -> None:
        """Validation should retry once in safe mode when SHM allocation fails."""
        calls = {"impl": 0, "safe_mode": 0}
        expected = (0.1, {"ssim": 0.9}, {0: {"ssim": 0.9}}, 1.0)

        def fake_validate_impl(_: int) -> tuple[float, dict[str, float], dict[int, dict[str, float]], float]:
            calls["impl"] += 1
            if calls["impl"] == 1:
                raise RuntimeError("unable to allocate shared memory(shm) for file </torch_1>: Success (0)")
            return expected

        def fake_enable_safe_mode() -> None:
            calls["safe_mode"] += 1
            trainer._validation_safe_mode_enabled = True

        monkeypatch.setattr(trainer, "_validate_epoch_impl", fake_validate_impl)
        monkeypatch.setattr(trainer, "_enable_validation_safe_mode", fake_enable_safe_mode)

        result = trainer.validate_epoch(epoch=0)

        assert result == expected
        assert calls["impl"] == 2
        assert calls["safe_mode"] == 1

    def test_raises_if_shm_error_persists_in_safe_mode(
        self, trainer: Trainer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validation should raise when SHM error persists after safe mode is already enabled."""
        trainer._validation_safe_mode_enabled = True

        def fake_validate_impl(_: int) -> None:
            raise RuntimeError("unable to allocate shared memory(shm) for file </torch_2>: Success (0)")

        monkeypatch.setattr(trainer, "_validate_epoch_impl", fake_validate_impl)

        with pytest.raises(RuntimeError, match="unable to allocate shared memory"):
            trainer.validate_epoch(epoch=0)
