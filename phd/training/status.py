"""Training status enum."""

from enum import StrEnum


class TrainingStatus(StrEnum):
    """Enum representing the status of a training run."""

    NOT_STARTED = "NOT_STARTED"  # Queued but never run
    RUNNING = "RUNNING"  # Currently training (or interrupted - will resume)
    ERROR = "ERROR"
    FINISHED_EPOCHS = "FINISHED_EPOCHS"
    EARLY_STOPPING = "EARLY_STOPPING"
    NAN_VALUE_DETECTED = "NAN_VALUE_DETECTED"  # Training stopped due to NaN/inf loss
