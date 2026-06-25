#!/usr/bin/env python3
"""Cleanup script to keep only the latest epoch checkpoint for each experiment.

Iterates over all experiments in experiments/train_nn1_cropped/, finds the epochs/
directory, and removes all but the highest-numbered epoch to reduce disk usage.
"""

import shutil
from pathlib import Path


def cleanup_experiment_epochs(experiment_dir: Path, dry_run: bool = False) -> None:
    """Remove all but the latest epoch from an experiment's epochs directory."""
    epochs_dir = experiment_dir / "epochs"

    if not epochs_dir.exists():
        print(f"  No epochs/ directory found in {experiment_dir.name}")
        return

    # Get all epoch directories (numeric names only)
    epoch_dirs = []
    for d in epochs_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            epoch_dirs.append((int(d.name), d))

    if len(epoch_dirs) <= 1:
        print(f"  {experiment_dir.name}: only {len(epoch_dirs)} epoch(s), nothing to clean")
        return

    # Sort by epoch number and keep the latest
    epoch_dirs.sort(key=lambda x: x[0])
    latest_epoch_num, _latest_epoch_dir = epoch_dirs[-1]
    epochs_to_remove = epoch_dirs[:-1]

    print(f"  {experiment_dir.name}:")
    print(f"    Keeping epoch {latest_epoch_num}")
    print(f"    Removing {len(epochs_to_remove)} older epoch(s): {[e[0] for e in epochs_to_remove]}")

    if not dry_run:
        for _epoch_num, epoch_dir in epochs_to_remove:
            shutil.rmtree(epoch_dir)
        print(f"    Removed {len(epochs_to_remove)} directories")


def main() -> int:
    """Clean up old epoch checkpoints, keeping only the latest for each experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Keep only the latest epoch checkpoint for each experiment")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).parent.parent / "experiments" / "train_nn1_cropped",
        help="Path to experiments directory (default: experiments/train_nn1_cropped)",
    )
    args = parser.parse_args()

    experiments_dir = args.experiments_dir

    if not experiments_dir.exists():
        print(f"Error: experiments directory not found: {experiments_dir}")
        return 1

    print(f"Cleaning up epochs in: {experiments_dir}")
    if args.dry_run:
        print("DRY RUN - no files will be deleted\n")
    else:
        print()

    # Find all experiment directories
    experiment_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])

    for experiment_dir in experiment_dirs:
        cleanup_experiment_epochs(experiment_dir, dry_run=args.dry_run)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
