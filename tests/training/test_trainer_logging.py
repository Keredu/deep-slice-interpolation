"""Tests for Trainer._log_batch_interval method."""

from unittest.mock import patch

from phd.training.trainer import Trainer


class TestLogBatchInterval:
    """Tests for _log_batch_interval method."""

    def test_computes_avg_time_correctly(self, trainer: Trainer) -> None:
        """Test that average time is computed correctly."""
        interval_times = [0.1, 0.2, 0.3]  # seconds
        interval_losses = [0.5, 0.5, 0.5]

        with patch("phd.training.trainer.logger") as mock_logger:
            trainer._log_batch_interval(
                phase="Train",
                batch_idx=9,
                num_batches=100,
                interval_times=interval_times,
                interval_losses=interval_losses,
            )

            # Average is 0.2s = 200ms
            call_args = mock_logger.debug.call_args[0][0]
            assert "avg=200.0" in call_args

    def test_computes_min_max_time(self, trainer: Trainer) -> None:
        """Test that min and max times are computed correctly."""
        interval_times = [0.1, 0.3, 0.2]  # seconds
        interval_losses = [0.5, 0.5, 0.5]

        with patch("phd.training.trainer.logger") as mock_logger:
            trainer._log_batch_interval(
                phase="Valid",
                batch_idx=49,
                num_batches=50,
                interval_times=interval_times,
                interval_losses=interval_losses,
            )

            call_args = mock_logger.debug.call_args[0][0]
            # Min is 0.1s = 100ms, Max is 0.3s = 300ms
            assert "min=100.0" in call_args
            assert "max=300.0" in call_args

    def test_computes_avg_loss(self, trainer: Trainer) -> None:
        """Test that average loss is computed correctly."""
        interval_times = [0.1, 0.1, 0.1]
        interval_losses = [0.3, 0.5, 0.7]  # Average = 0.5

        with patch("phd.training.trainer.logger") as mock_logger:
            trainer._log_batch_interval(
                phase="Train",
                batch_idx=19,
                num_batches=100,
                interval_times=interval_times,
                interval_losses=interval_losses,
            )

            call_args = mock_logger.debug.call_args[0][0]
            assert "loss=0.5000" in call_args

    def test_clears_interval_lists(self, trainer: Trainer) -> None:
        """Test that interval lists are cleared after logging."""
        interval_times = [0.1, 0.2, 0.3]
        interval_losses = [0.4, 0.5, 0.6]

        with patch("phd.training.trainer.logger"):
            trainer._log_batch_interval(
                phase="Train",
                batch_idx=9,
                num_batches=100,
                interval_times=interval_times,
                interval_losses=interval_losses,
            )

        assert interval_times == []
        assert interval_losses == []

    def test_logs_correct_batch_numbers(self, trainer: Trainer) -> None:
        """Test that batch numbers are logged correctly (1-indexed)."""
        interval_times = [0.1]
        interval_losses = [0.5]

        with patch("phd.training.trainer.logger") as mock_logger:
            trainer._log_batch_interval(
                phase="Train",
                batch_idx=49,  # 0-indexed
                num_batches=100,
                interval_times=interval_times,
                interval_losses=interval_losses,
            )

            call_args = mock_logger.debug.call_args[0][0]
            assert "batch 50/100" in call_args  # 1-indexed

    def test_logs_phase_name(self, trainer: Trainer) -> None:
        """Test that phase name is included in log."""
        interval_times = [0.1]
        interval_losses = [0.5]

        with patch("phd.training.trainer.logger") as mock_logger:
            trainer._log_batch_interval(
                phase="Valid",
                batch_idx=0,
                num_batches=10,
                interval_times=interval_times,
                interval_losses=interval_losses,
            )

            call_args = mock_logger.debug.call_args[0][0]
            assert "Valid batch" in call_args

    def test_single_batch_interval(self, trainer: Trainer) -> None:
        """Test logging with single batch in interval."""
        interval_times = [0.15]  # 150ms
        interval_losses = [0.42]

        with patch("phd.training.trainer.logger") as mock_logger:
            trainer._log_batch_interval(
                phase="Train",
                batch_idx=0,
                num_batches=1,
                interval_times=interval_times,
                interval_losses=interval_losses,
            )

            call_args = mock_logger.debug.call_args[0][0]
            # With single value, min=avg=max
            assert "min=150.0" in call_args
            assert "avg=150.0" in call_args
            assert "max=150.0" in call_args
            assert "loss=0.4200" in call_args
