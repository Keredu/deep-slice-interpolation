"""Tests for best-epoch selection in generate_missing_viz.py."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from generate_missing_viz import find_best_epoch_by_valid_loss


def _write_epochs_csv(experiment_dir: Path, rows: list[tuple[int, int, float]]) -> None:
    csv_path = experiment_dir / "epochs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "is_best", "valid_loss"])
        for row in rows:
            writer.writerow(row)


def _write_weights(experiment_dir: Path, epoch_dir: int) -> None:
    weights_path = experiment_dir / "epochs" / str(epoch_dir) / "weights.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"test")


def test_maps_csv_epochs_to_zero_based_checkpoint_dirs(tmp_path: Path) -> None:
    """Select the correct best checkpoint when CSV is 1-based and dirs are 0-based."""
    experiment_dir = tmp_path / "exp"
    (experiment_dir / "epochs").mkdir(parents=True)

    # CSV epoch 8 is the true best (valid_loss=0.59), saved as epochs/7/weights.pth.
    # The old implementation could incorrectly pick epochs/4 in this setup.
    _write_epochs_csv(
        experiment_dir,
        rows=[
            (1, 1, 0.90),
            (2, 1, 0.80),
            (3, 1, 0.70),
            (4, 0, 0.75),
            (5, 1, 0.60),
            (6, 0, 0.95),
            (7, 0, 0.90),
            (8, 1, 0.59),
        ],
    )

    for checkpoint_dir in [0, 1, 2, 4, 7]:
        _write_weights(experiment_dir, checkpoint_dir)

    assert find_best_epoch_by_valid_loss(experiment_dir) == 7


def test_falls_back_to_one_based_checkpoint_layout(tmp_path: Path) -> None:
    """Support experiments where checkpoint folders were saved as 1-based."""
    experiment_dir = tmp_path / "exp"
    (experiment_dir / "epochs").mkdir(parents=True)

    _write_epochs_csv(
        experiment_dir,
        rows=[
            (1, 1, 0.50),
            (2, 1, 0.40),
            (3, 1, 0.30),
        ],
    )

    # Only 1-based folder exists for the best CSV epoch.
    _write_weights(experiment_dir, 3)

    assert find_best_epoch_by_valid_loss(experiment_dir) == 3
