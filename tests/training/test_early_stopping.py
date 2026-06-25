"""Tests for EarlyStopping callback."""

from phd.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_counter_increments_without_improvement(self) -> None:
        """Counter goes up when loss doesn't improve."""
        early_stopping = EarlyStopping(patience=5)

        # First call sets best_loss
        early_stopping(1.0)
        assert early_stopping.counter == 0

        # Loss doesn't improve (higher is worse)
        early_stopping(1.5)
        assert early_stopping.counter == 1

        early_stopping(1.2)
        assert early_stopping.counter == 2

        early_stopping(1.1)  # Still worse than best (1.0)
        assert early_stopping.counter == 3

    def test_counter_resets_on_improvement(self) -> None:
        """Counter resets to 0 when loss improves."""
        early_stopping = EarlyStopping(patience=5)

        # Initial loss
        early_stopping(1.0)
        assert early_stopping.counter == 0

        # No improvement
        early_stopping(1.5)
        assert early_stopping.counter == 1

        early_stopping(1.2)
        assert early_stopping.counter == 2

        # Improvement (lower loss)
        early_stopping(0.8)
        assert early_stopping.counter == 0
        assert early_stopping.best_loss == 0.8

    def test_should_stop_at_patience(self) -> None:
        """Returns True when counter reaches patience."""
        patience = 3
        early_stopping = EarlyStopping(patience=patience)

        # Initial loss
        result = early_stopping(1.0)
        assert result is False

        # No improvement for patience epochs
        for _i in range(patience - 1):
            result = early_stopping(2.0)
            assert result is False

        # Reaches patience
        result = early_stopping(2.0)
        assert result is True

    def test_best_loss_updates(self) -> None:
        """best_loss updates when improvement occurs."""
        early_stopping = EarlyStopping(patience=5)

        # First call sets best_loss
        assert early_stopping.best_loss is None
        early_stopping(1.0)
        assert early_stopping.best_loss == 1.0

        # No update when loss is worse
        early_stopping(1.5)
        assert early_stopping.best_loss == 1.0

        # Update when loss improves
        early_stopping(0.5)
        assert early_stopping.best_loss == 0.5

        # Update again on further improvement
        early_stopping(0.3)
        assert early_stopping.best_loss == 0.3

    def test_min_delta(self) -> None:
        """Test that min_delta is respected for improvement threshold."""
        early_stopping = EarlyStopping(patience=5, min_delta=0.1)

        # Initial loss
        early_stopping(1.0)
        assert early_stopping.best_loss == 1.0

        # Small improvement (0.05 < min_delta), not counted
        early_stopping(0.95)
        assert early_stopping.counter == 1
        assert early_stopping.best_loss == 1.0

        # Large enough improvement (0.15 > min_delta)
        early_stopping(0.85)
        assert early_stopping.counter == 0
        assert early_stopping.best_loss == 0.85
