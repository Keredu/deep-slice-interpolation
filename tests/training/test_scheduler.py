"""Tests for scheduler setup utilities."""

import pytest
import torch
from torch.optim import SGD

from phd.training.scheduler import (
    CosineAnnealingWithWarmup,
    NoScheduler,
    is_per_batch_scheduler,
    needs_metric_for_step,
    setup_scheduler,
)


class TestSetupScheduler:
    """Tests for setup_scheduler function."""

    @pytest.fixture
    def optimizer(self) -> SGD:
        """Create a simple optimizer for testing."""
        model = torch.nn.Linear(10, 10)
        return SGD(model.parameters(), lr=0.1)

    def test_cosine_scheduler(self, optimizer: SGD) -> None:
        """Test cosine scheduler creation."""
        config = {"name": "cosine", "params": {"eta_min": 1e-6}}
        scheduler = setup_scheduler(optimizer, config, num_epochs=100)
        assert scheduler is not None
        # Step and verify LR changes
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # LR should decrease (or stay same for first step depending on implementation)
        assert optimizer.param_groups[0]["lr"] <= initial_lr

    def test_cosine_warmup_scheduler(self, optimizer: SGD) -> None:
        """Test cosine warmup scheduler creation."""
        config = {"name": "cosine_warmup", "params": {"warmup_epochs": 5, "eta_min": 0}}
        scheduler = setup_scheduler(optimizer, config, num_epochs=100)
        assert isinstance(scheduler, CosineAnnealingWithWarmup)

    def test_cosine_warmup_lr_increases_during_warmup(self, optimizer: SGD) -> None:
        """Test that LR increases during warmup phase."""
        config = {"name": "cosine_warmup", "params": {"warmup_epochs": 5, "eta_min": 0}}
        scheduler = setup_scheduler(optimizer, config, num_epochs=100)

        # During warmup, LR should increase
        lrs = []
        for _ in range(5):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Each subsequent LR should be >= previous during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1]

    def test_onecycle_scheduler(self, optimizer: SGD) -> None:
        """Test OneCycleLR scheduler creation."""
        config = {
            "name": "onecycle",
            "params": {"max_lr": 0.1, "pct_start": 0.3, "anneal_strategy": "cos"},
        }
        scheduler = setup_scheduler(optimizer, config, num_epochs=10, steps_per_epoch=100)
        assert scheduler is not None

    def test_onecycle_requires_steps_per_epoch(self, optimizer: SGD) -> None:
        """Test that OneCycleLR raises without steps_per_epoch."""
        config = {
            "name": "onecycle",
            "params": {"max_lr": 0.1, "pct_start": 0.3, "anneal_strategy": "cos"},
        }
        with pytest.raises(ValueError, match="steps_per_epoch"):
            setup_scheduler(optimizer, config, num_epochs=10)

    def test_plateau_scheduler(self, optimizer: SGD) -> None:
        """Test ReduceLROnPlateau scheduler creation."""
        config = {
            "name": "plateau",
            "params": {"mode": "min", "factor": 0.5, "patience": 5, "min_lr": 0},
        }
        scheduler = setup_scheduler(optimizer, config, num_epochs=100)
        assert scheduler is not None

    def test_cosine_restarts_scheduler(self, optimizer: SGD) -> None:
        """Test CosineAnnealingWarmRestarts scheduler creation."""
        config = {
            "name": "cosine_restarts",
            "params": {"T_0": 10, "T_mult": 2, "eta_min": 1e-6},
        }
        scheduler = setup_scheduler(optimizer, config, num_epochs=100)
        assert scheduler is not None

        # LR should restart after T_0 epochs
        lrs = []
        for _ in range(25):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should drop then jump back up at restart (epoch 10)
        assert lrs[9] < lrs[0]  # Decayed before restart
        assert lrs[10] > lrs[9]  # Jumped back up after restart

    def test_step_scheduler(self, optimizer: SGD) -> None:
        """Test StepLR scheduler creation."""
        config = {"name": "step", "params": {"step_size": 10, "gamma": 0.1}}
        scheduler = setup_scheduler(optimizer, config, num_epochs=100)
        assert scheduler is not None

    def test_none_scheduler(self, optimizer: SGD) -> None:
        """Test no scheduler (constant LR)."""
        config = {"name": "none", "params": {}}
        scheduler = setup_scheduler(optimizer, config, num_epochs=100)
        assert isinstance(scheduler, NoScheduler)

        # LR should not change
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == initial_lr

    def test_unknown_scheduler_raises(self, optimizer: SGD) -> None:
        """Test that unknown scheduler name raises ValueError."""
        config = {"name": "unknown", "params": {}}
        with pytest.raises(ValueError, match="Unknown scheduler"):
            setup_scheduler(optimizer, config, num_epochs=100)


class TestIsPerBatchScheduler:
    """Tests for is_per_batch_scheduler function."""

    def test_onecycle_is_per_batch(self) -> None:
        """Test that OneCycleLR is identified as per-batch."""
        assert is_per_batch_scheduler({"name": "onecycle"}) is True

    def test_cosine_is_not_per_batch(self) -> None:
        """Test that cosine is not per-batch."""
        assert is_per_batch_scheduler({"name": "cosine"}) is False

    def test_step_is_not_per_batch(self) -> None:
        """Test that step is not per-batch."""
        assert is_per_batch_scheduler({"name": "step"}) is False


class TestNeedsMetricForStep:
    """Tests for needs_metric_for_step function."""

    def test_plateau_needs_metric(self) -> None:
        """Test that plateau scheduler needs metric."""
        assert needs_metric_for_step({"name": "plateau"}) is True

    def test_cosine_does_not_need_metric(self) -> None:
        """Test that cosine does not need metric."""
        assert needs_metric_for_step({"name": "cosine"}) is False

    def test_onecycle_does_not_need_metric(self) -> None:
        """Test that OneCycleLR does not need metric."""
        assert needs_metric_for_step({"name": "onecycle"}) is False


class TestNoScheduler:
    """Tests for NoScheduler class."""

    def test_step_does_nothing(self) -> None:
        """Test that step() is a no-op."""
        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = NoScheduler(optimizer)

        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        scheduler.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == initial_lr

    def test_state_dict_is_empty(self) -> None:
        """Test that state_dict returns empty dict."""
        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = NoScheduler(optimizer)
        assert scheduler.state_dict() == {}

    def test_load_state_dict_is_noop(self) -> None:
        """Test that load_state_dict is a no-op."""
        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = NoScheduler(optimizer)
        # Should not raise
        scheduler.load_state_dict({"some": "state"})


class TestCosineAnnealingWithWarmup:
    """Tests for CosineAnnealingWithWarmup class."""

    def test_warmup_phase(self) -> None:
        """Test LR behavior during warmup phase."""
        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=5,
            total_epochs=100,
            eta_min=0,
        )

        # LR should start at 0.1 * (1/5) = 0.02 and increase to 0.1
        lrs = []
        for _ in range(6):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should increase during warmup
        assert lrs[0] < lrs[4]
        # After warmup, should be close to base_lr
        assert abs(lrs[4] - 0.1) < 0.01

    def test_cosine_phase(self) -> None:
        """Test LR behavior during cosine decay phase."""
        model = torch.nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=5,
            total_epochs=20,
            eta_min=0.001,
        )

        # Skip warmup
        for _ in range(5):
            scheduler.step()

        # LR should decrease during cosine phase
        lrs = []
        for _ in range(15):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should decrease overall
        assert lrs[0] > lrs[-1]
        # Should approach eta_min
        assert lrs[-1] >= 0.001
