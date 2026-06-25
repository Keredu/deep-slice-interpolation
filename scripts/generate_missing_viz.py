#!/usr/bin/env python3
"""Generate missing test visualizations for best experiment epochs.

This script scans experiment directories and generates visualizations for the
best epoch (based on epochs.csv is_best column) if missing. Useful when training
was done with generate_test_viz_real=False but visualizations are needed afterwards.

Usage:
    uv run scripts/generate_missing_viz.py                    # All experiments
    uv run scripts/generate_missing_viz.py exp_name           # Specific experiment
    uv run scripts/generate_missing_viz.py --dry-run          # Show what would be done
"""

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import torch
from loguru import logger

from phd.config_io import resolve_config_path
from phd.datasets.interpolation.two_to_one_slice import (
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)
from phd.models.setup_model import setup_model
from phd.viz import save_test_visualization


def find_best_epoch_by_valid_loss(experiment_dir: Path) -> int | None:
    """Find epoch with lowest validation loss that has weights saved.

    Training writes `epochs.csv` with 1-indexed epoch numbers, while checkpoint
    directories are currently 0-indexed (`epochs/0`, `epochs/1`, ...). This
    function maps rows to checkpoint directories accordingly and falls back to a
    1-indexed directory layout for backward compatibility.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Checkpoint directory epoch number (folder name) for the best model
    """
    csv_path = experiment_dir / "epochs.csv"
    epochs_dir = experiment_dir / "epochs"
    if not csv_path.exists() or not epochs_dir.exists():
        return None

    # Read all epochs with their validation loss and mapped checkpoint folder.
    # Tuple: (checkpoint_dir_epoch, valid_loss, csv_epoch, is_best)
    epoch_losses: list[tuple[int, float, int, bool]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_epoch = int(row["epoch"])
            valid_loss_str = row["valid_loss"]

            # Skip if no valid loss or non-finite value
            if not valid_loss_str:
                continue

            valid_loss = float(valid_loss_str)
            if not math.isfinite(valid_loss):
                continue

            # Preferred mapping for current trainer: csv epoch N -> epochs/(N-1).
            # Fallback mapping supports legacy experiments: csv epoch N -> epochs/N.
            checkpoint_epoch = csv_epoch - 1
            weights_path = epochs_dir / str(checkpoint_epoch) / "weights.pth"
            if not weights_path.exists():
                checkpoint_epoch = csv_epoch
                weights_path = epochs_dir / str(checkpoint_epoch) / "weights.pth"
                if not weights_path.exists():
                    continue

            is_best = row.get("is_best") == "1"
            epoch_losses.append((checkpoint_epoch, valid_loss, csv_epoch, is_best))

    if not epoch_losses:
        return None

    # Prefer epochs explicitly marked as best when available.
    best_marked = [x for x in epoch_losses if x[3]]
    candidates = best_marked if best_marked else epoch_losses

    # Lowest validation loss wins; for ties keep the latest checkpoint epoch.
    best_epoch, *_ = min(candidates, key=lambda x: (x[1], -x[0]))
    return best_epoch


def find_best_epoch_missing_viz(experiment_dir: Path) -> tuple[Path, Path] | None:
    """Find best epoch (by valid_loss) if it's missing target_is_real viz.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Tuple (epoch_dir, weights_path) if best epoch is missing viz, else None
    """
    best_epoch = find_best_epoch_by_valid_loss(experiment_dir)
    if best_epoch is None:
        return None

    epoch_dir = experiment_dir / "epochs" / str(best_epoch)
    weights_path = epoch_dir / "weights.pth"
    viz_dir = epoch_dir / "viz" / "target_is_real"

    if not viz_dir.exists():
        return (epoch_dir, weights_path)

    return None


def load_config_from_experiment(experiment_dir: Path) -> dict:
    """Load config.json from experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Config dictionary
    """
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open() as f:
        return json.load(f)


def generate_viz_for_epoch(
    epoch_dir: Path,
    weights_path: Path,
    config: dict,
    device: torch.device,
) -> None:
    """Generate test visualization for a single epoch.

    Args:
        epoch_dir: Path to epoch directory
        weights_path: Path to weights.pth checkpoint
        config: Experiment config dictionary
        device: Torch device to use
    """
    model_config = config["model"]

    # Set up model (in_channels=2 for two adjacent slices, out_channels=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
        warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*Arguments other than.*")
        warnings.filterwarnings("ignore", message=".*Unexpected keys.*")
        model = setup_model(
            in_channels=2,
            out_channels=1,
            pretrained=False,  # We're loading weights, no need for pretrained
            model_type=model_config["type"],
            encoder_name=model_config.get("encoder_name"),
        )

    model = model.to(device)

    # Load checkpoint weights
    checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Strip _orig_mod. prefix if present (from torch.compile)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("_orig_mod.")
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # Create test dataset
    test_dataset = TwoToOneSliceTestDataset(
        root_dir=resolve_config_path(config["data_path"]),
        transform=STANDARD_TRANSFORM,
        stage="test",
        mode="target_is_real",
    )

    # Generate visualization
    viz_dir = epoch_dir / "viz" / "target_is_real"
    viz_dir.mkdir(parents=True, exist_ok=True)

    save_test_visualization(
        test_dataset=test_dataset,
        model=model,
        device=device,
        save_dir=viz_dir,
        batch_size=config.get("batch_size", 8),
    )

    logger.info(f"Generated visualization: {viz_dir}")


def main() -> None:
    """Generate missing visualizations for the best checkpoint of each experiment."""
    parser = argparse.ArgumentParser(description="Generate missing test visualizations for experiment epochs")
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Experiment name (default: process all experiments)",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without generating",
    )
    args = parser.parse_args()

    experiments_dir = args.experiments_dir
    if not experiments_dir.exists():
        logger.error(f"Experiments directory not found: {experiments_dir}")
        return

    # Find experiments to process (recursively search for directories with config.json)
    if args.experiment:
        # Try exact path first, then search
        exact_path = experiments_dir / args.experiment
        if exact_path.exists() and (exact_path / "config.json").exists():
            experiment_dirs = [exact_path]
        else:
            # Search recursively for matching experiment name
            experiment_dirs = [
                d
                for d in experiments_dir.rglob("*")
                if d.is_dir() and d.name == args.experiment and (d / "config.json").exists()
            ]
            if not experiment_dirs:
                logger.error(f"Experiment not found: {args.experiment}")
                return
    else:
        # Find all directories containing config.json (recursive)
        experiment_dirs = [d.parent for d in experiments_dir.rglob("config.json") if d.parent != experiments_dir]

    if not experiment_dirs:
        logger.info("No experiments found")
        return

    # Collect best epochs missing viz
    all_missing: list[tuple[Path, Path, Path]] = []  # (experiment_dir, epoch_dir, weights_path)
    for exp_dir in sorted(experiment_dirs):
        result = find_best_epoch_missing_viz(exp_dir)
        if result:
            epoch_dir, weights_path = result
            all_missing.append((exp_dir, epoch_dir, weights_path))

    if not all_missing:
        logger.info("All best epochs have visualizations")
        return

    logger.info(f"Found {len(all_missing)} best epochs missing visualizations:")
    for exp_dir, epoch_dir, _ in all_missing:
        logger.info(f"  {exp_dir.name}/epochs/{epoch_dir.name}")

    if args.dry_run:
        logger.info("Dry run - no visualizations generated")
        return

    # Set up device
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (will be slow)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Process each missing epoch
    for exp_dir, epoch_dir, weights_path in all_missing:
        logger.info(f"Processing {exp_dir.name}/epochs/{epoch_dir.name}")
        try:
            config = load_config_from_experiment(exp_dir)
            generate_viz_for_epoch(epoch_dir, weights_path, config, device)
        except Exception as e:
            logger.error(f"Failed to generate viz for {epoch_dir}: {e}")
            continue

        # Clear CUDA cache between experiments to avoid OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Done")


if __name__ == "__main__":
    main()
