"""Experiment registry for tracking all training runs."""

import json
from datetime import UTC, datetime
from pathlib import Path

from phd.training.status import TrainingStatus


def _get_registry_path(experiments_dir: Path) -> Path:
    """Get the path to the experiments registry file."""
    return experiments_dir / "experiments_registry.json"


def _load_registry(experiments_dir: Path) -> dict:
    """Load the experiments registry from disk.

    Args:
        experiments_dir: Path to the experiments directory

    Returns:
        Dictionary with experiment names as keys and metadata as values
    """
    registry_path = _get_registry_path(experiments_dir)
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_registry(experiments_dir: Path, registry: dict) -> None:
    """Save the experiments registry to disk.

    Args:
        experiments_dir: Path to the experiments directory
        registry: Dictionary with experiment data
    """
    experiments_dir.mkdir(parents=True, exist_ok=True)
    registry_path = _get_registry_path(experiments_dir)
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _convert_config_for_json(config: dict) -> dict:
    """Convert config dict for JSON storage (Path objects to strings).

    Args:
        config: Configuration dictionary

    Returns:
        Config with Path objects converted to strings
    """
    full_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            full_config[key] = str(value)
        else:
            full_config[key] = value
    return full_config


def register_experiment(
    experiments_dir: Path,
    exp_name: str,
    config: dict,
) -> None:
    """Register a new experiment or update existing one to running status.

    Args:
        experiments_dir: Path to the experiments directory
        exp_name: Name of the experiment
        config: Full experiment configuration dictionary
    """
    registry = _load_registry(experiments_dir)

    now = datetime.now(tz=UTC).isoformat()

    if exp_name in registry:
        # Update existing experiment - preserve queued_at and config
        registry[exp_name]["status"] = TrainingStatus.RUNNING.value
        registry[exp_name]["last_started"] = now
        registry[exp_name]["runs"] = registry[exp_name].get("runs", 0) + 1
        # Clear finished timestamp on resume but keep queued_at and config
        registry[exp_name].pop("finished", None)
        # Don't overwrite config - it was set by queue_experiment with full details
    else:
        # Create new experiment entry (started directly without queuing)
        # Store full config, converting Path objects to strings for JSON
        full_config = _convert_config_for_json(config)

        registry[exp_name] = {
            "status": TrainingStatus.RUNNING.value,
            "queued_at": now,  # Use start time as queue time if not queued
            "created": now,
            "last_started": now,
            "finished": None,
            "runs": 1,
            "config": full_config,
        }

    _save_registry(experiments_dir, registry)


def update_experiment_status(
    experiments_dir: Path,
    exp_name: str,
    status: TrainingStatus,
    best_valid_loss: float | None = None,
    final_epoch: int | None = None,
) -> None:
    """Update the status of an experiment in the registry.

    Args:
        experiments_dir: Path to the experiments directory
        exp_name: Name of the experiment
        status: New status for the experiment
        best_valid_loss: Best validation loss achieved (optional)
        final_epoch: Final epoch number (optional)
    """
    registry = _load_registry(experiments_dir)

    if exp_name not in registry:
        # Experiment wasn't registered, create minimal entry
        registry[exp_name] = {
            "status": status.value,
            "created": datetime.now(tz=UTC).isoformat(),
        }
    else:
        registry[exp_name]["status"] = status.value

    # Record finish time for terminal states
    terminal_states = {
        TrainingStatus.FINISHED_EPOCHS,
        TrainingStatus.EARLY_STOPPING,
        TrainingStatus.ERROR,
        TrainingStatus.NAN_VALUE_DETECTED,
    }
    if status in terminal_states:
        registry[exp_name]["finished"] = datetime.now(tz=UTC).isoformat()

    if best_valid_loss is not None:
        registry[exp_name]["best_valid_loss"] = best_valid_loss

    if final_epoch is not None:
        registry[exp_name]["final_epoch"] = final_epoch

    _save_registry(experiments_dir, registry)


def get_experiment_status(experiments_dir: Path, exp_name: str) -> TrainingStatus | None:
    """Get the status of an experiment from the registry.

    Args:
        experiments_dir: Path to the experiments directory
        exp_name: Name of the experiment

    Returns:
        TrainingStatus if experiment exists, None otherwise
    """
    registry = _load_registry(experiments_dir)
    if exp_name in registry:
        return TrainingStatus(registry[exp_name]["status"])
    return None


def list_experiments(experiments_dir: Path) -> dict:
    """List all experiments in the registry.

    Args:
        experiments_dir: Path to the experiments directory

    Returns:
        Dictionary with experiment names as keys and metadata as values
    """
    return _load_registry(experiments_dir)


def queue_experiment(
    experiments_dir: Path,
    exp_name: str,
    config: dict,
) -> bool:
    """Queue a new experiment without running it.

    Args:
        experiments_dir: Path to the experiments directory
        exp_name: Name of the experiment
        config: Full experiment configuration dictionary (all parameters)

    Returns:
        True if experiment was queued, False if it already exists
    """
    registry = _load_registry(experiments_dir)

    if exp_name in registry:
        return False

    now = datetime.now(tz=UTC).isoformat()

    # Store full config, converting Path objects to strings for JSON
    full_config = _convert_config_for_json(config)

    registry[exp_name] = {
        "status": TrainingStatus.NOT_STARTED.value,
        "queued_at": now,
        "created": now,
        "last_started": None,
        "finished": None,
        "runs": 0,
        "config": full_config,
    }

    experiments_dir.mkdir(parents=True, exist_ok=True)
    _save_registry(experiments_dir, registry)
    return True


def get_next_experiment(experiments_dir: Path) -> str | None:
    """Get the next experiment to run based on priority.

    Priority order:
    1. RUNNING - Resume interrupted experiments first (sorted by last_started, oldest first)
    2. NOT_STARTED - Run new queued experiments (sorted by queued_at, oldest first)

    Args:
        experiments_dir: Path to the experiments directory

    Returns:
        Experiment name or None if no experiments are pending
    """
    registry = _load_registry(experiments_dir)

    # Collect RUNNING experiments sorted by last_started (oldest first)
    # These are experiments that were interrupted (e.g., Ctrl+C)
    running = [
        (name, data.get("last_started", ""))
        for name, data in registry.items()
        if data["status"] == TrainingStatus.RUNNING.value
    ]
    running.sort(key=lambda x: x[1] or "")

    if running:
        return running[0][0]

    # Collect NOT_STARTED experiments sorted by queued_at (oldest first)
    not_started = [
        (name, data.get("queued_at", ""))
        for name, data in registry.items()
        if data["status"] == TrainingStatus.NOT_STARTED.value
    ]
    not_started.sort(key=lambda x: x[1] or "")

    if not_started:
        return not_started[0][0]

    return None


def get_queue_status(experiments_dir: Path) -> dict[str, int]:
    """Get counts of experiments by status.

    Args:
        experiments_dir: Path to the experiments directory

    Returns:
        Dictionary with status names as keys and counts as values
    """
    registry = _load_registry(experiments_dir)

    counts = {
        TrainingStatus.NOT_STARTED.value: 0,
        TrainingStatus.RUNNING.value: 0,
        TrainingStatus.FINISHED_EPOCHS.value: 0,
        TrainingStatus.EARLY_STOPPING.value: 0,
        TrainingStatus.ERROR.value: 0,
        TrainingStatus.NAN_VALUE_DETECTED.value: 0,
    }

    for data in registry.values():
        status = data.get("status", "UNKNOWN")
        if status in counts:
            counts[status] += 1

    return counts


def reset_error_experiments(experiments_dir: Path, output_dir: Path | None = None) -> list[str]:
    """Reset ERROR experiments to RUNNING status for priority resumption.

    Args:
        experiments_dir: Path to the registry directory
        output_dir: Path to experiment output folders (to clean up error files)

    Returns:
        List of experiment names that were reset
    """
    registry = _load_registry(experiments_dir)
    reset = []

    # Error-related files to clean up
    error_files = ["error.log", "stderr.log", "crash_trace.txt"]

    for exp_name, data in registry.items():
        if data["status"] == TrainingStatus.ERROR.value:
            reset.append(exp_name)
            data["status"] = TrainingStatus.RUNNING.value
            data["last_started"] = datetime.now(tz=UTC).isoformat()
            data.pop("finished", None)
            data.pop("error_message", None)
            data.pop("crash_detected", None)

            # Clean up error-related files if output_dir is provided
            if output_dir is not None:
                exp_dir = output_dir / exp_name
                for filename in error_files:
                    filepath = exp_dir / filename
                    if filepath.exists():
                        filepath.unlink()

    if reset:
        _save_registry(experiments_dir, registry)

    return reset
