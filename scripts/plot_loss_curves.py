#!/usr/bin/env python3
"""Plot loss curves for experiments.

Reads epochs.csv and generates a loss curve plot saved to the experiment directory.

Usage:
    uv run scripts/plot_loss_curves.py exp_name                    # Specific experiment
    uv run scripts/plot_loss_curves.py                             # All experiments
    uv run scripts/plot_loss_curves.py --experiments-dir experiments/train_nn1_cropped
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


def plot_loss_curves(experiment_dir: Path) -> None:
    """Plot and save loss curves for an experiment.

    Args:
        experiment_dir: Path to experiment directory containing epochs.csv
    """
    csv_path = experiment_dir / "epochs.csv"
    if not csv_path.exists():
        logger.warning(f"No epochs.csv found in {experiment_dir.name}, skipping")
        return

    df = pd.read_csv(csv_path)

    if "train_loss" not in df.columns or "valid_loss" not in df.columns:
        logger.warning(f"Missing loss columns in {experiment_dir.name}, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = df["epoch"]
    ax.plot(epochs, df["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, df["valid_loss"], label="Valid Loss", linewidth=1.5)

    # Mark best epochs
    if "is_best" in df.columns:
        best_mask = df["is_best"] == 1
        ax.scatter(
            epochs[best_mask],
            df["valid_loss"][best_mask],
            color="green",
            s=20,
            zorder=5,
            label="Best",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves - {experiment_dir.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Use log scale if loss range is large
    loss_min = min(df["train_loss"].min(), df["valid_loss"].min())
    loss_max = max(df["train_loss"].max(), df["valid_loss"].max())
    if loss_max / loss_min > 10:
        ax.set_yscale("log")

    output_path = experiment_dir / "loss_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved {output_path}")


def find_experiments_with_csv(base_dir: Path) -> list[Path]:
    """Recursively find all directories containing epochs.csv.

    Args:
        base_dir: Base directory to search from

    Returns:
        List of experiment directories containing epochs.csv
    """
    return [csv.parent for csv in base_dir.rglob("epochs.csv")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves for experiments")
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Experiment directory path (default: process all experiments)",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments/)",
    )
    args = parser.parse_args()

    experiments_dir = args.experiments_dir
    if not experiments_dir.exists():
        logger.error(f"Experiments directory not found: {experiments_dir}")
        return

    # Find experiments to process
    if args.experiment:
        exp_path = Path(args.experiment)
        # Handle both absolute path and relative to experiments_dir
        if not exp_path.is_absolute():
            exp_path = experiments_dir / exp_path
        if not exp_path.exists():
            logger.error(f"Experiment not found: {exp_path}")
            return
        experiment_dirs = [exp_path]
    else:
        experiment_dirs = find_experiments_with_csv(experiments_dir)

    if not experiment_dirs:
        logger.info("No experiments with epochs.csv found")
        return

    for exp_dir in sorted(experiment_dirs):
        plot_loss_curves(exp_dir)

    logger.info("Done")


if __name__ == "__main__":
    main()
