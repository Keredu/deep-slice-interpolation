"""Scheduler setup utilities for training."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


class CosineAnnealingWithWarmup(LRScheduler):
    """Cosine annealing scheduler with linear warmup.

    During warmup, LR increases linearly from 0 to base_lr.
    After warmup, LR follows cosine decay to eta_min.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.cosine_epochs = total_epochs - warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            import math

            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epoch / self.cosine_epochs))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]


class NoScheduler:
    """Dummy scheduler that does nothing (constant LR)."""

    def __init__(self, optimizer: Optimizer) -> None:
        """Initialize the no-op scheduler.

        Args:
            optimizer: Wrapped optimizer (stored but not used)
        """
        self.optimizer = optimizer

    def step(self, *args, **kwargs) -> None:
        """No-op step."""
        pass

    def state_dict(self) -> dict:
        """Return empty state dict."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """No-op load."""
        pass


def setup_scheduler(
    optimizer: Optimizer,
    scheduler_config: dict,
    num_epochs: int,
    steps_per_epoch: int | None = None,
) -> LRScheduler | NoScheduler:
    """Create a learning rate scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule
        scheduler_config: Configuration dict with 'name' and 'params' keys
        num_epochs: Total number of training epochs
        steps_per_epoch: Number of batches per epoch (required for OneCycleLR)

    Returns:
        Configured learning rate scheduler

    Raises:
        ValueError: If scheduler name is not supported
    """
    name = scheduler_config.get("name", "cosine").lower()
    params = scheduler_config.get("params", {})

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=params["eta_min"],
        )

    elif name == "cosine_restarts":
        # CosineAnnealingWarmRestarts: periodic LR resets to escape local minima
        # T_0: epochs until first restart
        # T_mult: multiply T_0 by this after each restart
        #   e.g., T_0=20, T_mult=2 → restarts at 20, 60, 140, 300, ...
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params["T_0"],
            T_mult=params["T_mult"],
            eta_min=params["eta_min"],
        )

    elif name == "cosine_warmup":
        return CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=params["warmup_epochs"],
            total_epochs=num_epochs,
            eta_min=params["eta_min"],
        )

    elif name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR requires steps_per_epoch")
        return OneCycleLR(
            optimizer,
            max_lr=params["max_lr"],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=params["pct_start"],
            anneal_strategy=params["anneal_strategy"],
        )

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=params["mode"],
            factor=params["factor"],
            patience=params["patience"],
            min_lr=params["min_lr"],
        )

    elif name == "step":
        return StepLR(
            optimizer,
            step_size=params["step_size"],
            gamma=params["gamma"],
        )

    elif name == "none":
        return NoScheduler(optimizer)

    else:
        raise ValueError(f"Unknown scheduler: {name}")


def is_per_batch_scheduler(scheduler_config: dict) -> bool:
    """Check if scheduler should be stepped per batch (not per epoch).

    Args:
        scheduler_config: Scheduler configuration dict

    Returns:
        True if scheduler should be stepped after each batch
    """
    name = scheduler_config.get("name", "cosine").lower()
    return name == "onecycle"


def needs_metric_for_step(scheduler_config: dict) -> bool:
    """Check if scheduler needs a metric value for step().

    Args:
        scheduler_config: Scheduler configuration dict

    Returns:
        True if scheduler.step() requires a metric argument
    """
    name = scheduler_config.get("name", "cosine").lower()
    return name == "plateau"
