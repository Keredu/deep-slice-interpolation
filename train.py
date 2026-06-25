"""Training entry point for CT slice interpolation experiments.

This script provides a queue-based interface to run training experiments.

Usage:
    uv run train.py              # Run next experiment from queue
    uv run train.py --run-all    # Run until queue empty
    uv run train.py --show-queue # Show queue status
"""

# Fix persistent shared memory errors in DataLoader.
# The default 'file_descriptor' strategy uses /dev/shm which has known bugs
# (fd leaks, race conditions). The 'file_system' strategy uses regular temp
# files instead, avoiding shm allocation entirely.
# Using 'spawn' start method (instead of 'fork') avoids inheriting file
# descriptors and reduces race conditions in worker processes.
import torch.multiprocessing

torch.multiprocessing.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")

import argparse
import sys
from pathlib import Path
from typing import TextIO

from loguru import logger

from phd.training.registry import (
    get_next_experiment,
    get_queue_status,
    list_experiments,
)


def show_queue_status(registry_dir: Path) -> None:
    """Display the current queue status."""
    counts = get_queue_status(registry_dir)
    registry = list_experiments(registry_dir)

    print("\n=== Experiment Queue Status ===\n")

    # Show counts
    print("Status counts:")
    for status, count in counts.items():
        if count > 0:
            print(f"  {status}: {count}")

    # Show pending experiments in queue order
    running = []
    not_started = []
    for name, data in registry.items():
        status = data.get("status")
        if status == "RUNNING":
            running.append((name, data.get("last_started", "")))
        elif status == "NOT_STARTED":
            not_started.append((name, data.get("queued_at", "")))

    running.sort(key=lambda x: x[1] or "")
    not_started.sort(key=lambda x: x[1] or "")

    if running or not_started:
        print("\nPending experiments (in queue order):")
        for i, (name, _) in enumerate(running, 1):
            print(f"  {i}. [RUNNING - will resume] {name}")
        for i, (name, _) in enumerate(not_started, len(running) + 1):
            print(f"  {i}. [NOT_STARTED] {name}")
    else:
        print("\nNo experiments in queue.")

    # Show completed
    completed = [name for name, data in registry.items() if data.get("status") in {"FINISHED_EPOCHS", "EARLY_STOPPING"}]
    if completed:
        print(f"\nCompleted experiments: {len(completed)}")

    # Show NaN detected
    nan_detected = [name for name, data in registry.items() if data.get("status") == "NAN_VALUE_DETECTED"]
    if nan_detected:
        print("\nExperiments stopped due to NaN/inf loss:")
        for name in nan_detected:
            final_epoch = registry[name].get("final_epoch", "?")
            print(f"  - {name}: stopped at epoch {final_epoch}")

    # Show errors
    errors = [name for name, data in registry.items() if data.get("status") == "ERROR"]
    if errors:
        print("\nFailed experiments (use --reset-errors in register_experiments.py to retry):")
        for name in errors:
            msg = registry[name].get("error_message", "Unknown error")
            print(f"  - {name}: {msg}")

    print()


class TeeStderr:
    """Tee stderr to both console and a log file."""

    def __init__(self, log_file: TextIO) -> None:
        """Initialize with a log file to write to."""
        self.log_file = log_file
        self.original_stderr = sys.__stderr__

    def write(self, data: str) -> int:
        """Write data to both stderr and log file."""
        self.original_stderr.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self) -> None:
        """Flush both stderr and log file."""
        self.original_stderr.flush()
        self.log_file.flush()


def run_next_experiment(registry_dir: Path, experiments_dir: Path) -> bool:
    """Run the next experiment in the queue.

    Returns:
        True if an experiment was run, False if queue is empty
    """
    # Lazy import to avoid loading PyTorch for queue status commands
    from phd.training import Trainer

    exp_name = get_next_experiment(registry_dir)

    if exp_name is None:
        logger.info("No experiments in queue. Use register_experiments.py to add experiments.")
        return False

    logger.info(f"Running next experiment: {exp_name}")

    # Load full config from registry (stored exactly as it will be used)
    registry = list_experiments(registry_dir)
    config = registry[exp_name]["config"]

    trainer = Trainer(config)

    # Tee stderr to capture any crash output
    exp_dir = experiments_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = exp_dir / "stderr.log"
    with stderr_log.open("w", encoding="utf-8") as log_file:
        old_stderr = sys.stderr
        sys.stderr = TeeStderr(log_file)
        try:
            trainer.run()
        finally:
            sys.stderr = old_stderr

    return True


def main() -> None:
    """Run experiments from the queue."""
    # Clean stale PyTorch shm files from previous crashes
    import glob
    import os

    for shm_file in glob.glob("/dev/shm/torch_*"):
        try:
            os.remove(shm_file)
        except OSError:
            pass

    parser = argparse.ArgumentParser(description="Run CT slice interpolation experiments")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all experiments in queue until empty",
    )
    parser.add_argument(
        "--show-queue",
        action="store_true",
        help="Show queue status and exit",
    )
    args = parser.parse_args()

    # Directory paths (running from project root)
    registry_dir = Path("./experiments")
    experiments_dir = Path("./experiments/train_nn1_cropped")

    if args.show_queue:
        show_queue_status(registry_dir)
        return

    if args.run_all:
        logger.info("Running all experiments in queue...")
        count = 0
        while run_next_experiment(registry_dir=registry_dir, experiments_dir=experiments_dir):
            count += 1
        logger.info(f"Finished running {count} experiment(s)")
    elif not run_next_experiment(registry_dir=registry_dir, experiments_dir=experiments_dir):
        show_queue_status(registry_dir)


if __name__ == "__main__":
    main()
    # Force exit: spawn multiprocessing can leave resource tracker threads
    # that prevent clean shutdown after DataLoader workers finish
    sys.exit(0)
