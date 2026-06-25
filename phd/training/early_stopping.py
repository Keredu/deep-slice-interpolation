"""Early stopping callback to prevent overfitting."""


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement
    is observed for a specified number of epochs (patience).
    """

    def __init__(self, patience: int = 7, min_delta: float = 0) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss.

        Args:
            val_loss: Validation loss value

        Returns:
            True if the model should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        self.best_loss = val_loss
        self.counter = 0
        return False
