#!/usr/bin/env python3
"""Generate training curves figure for the paper.

Panel (a): Validation SSIM over epochs for converged runs — a common metric
           that is directly comparable regardless of which loss was used.
Panel (b): Validation loss (log scale) for divergent SSIM-family runs,
           showing rapid loss explosion.

Usage:
    uv run scripts/generate_paper_training_curves.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENTS_DIR = Path("experiments/train_nn1_cropped")
OUTPUT_PATH = Path("results/figures/training_curves.pdf")

STABLE = {
    "L1": "l1_lr8e-4_b39be9",
    "MSE": "mse_lr8e-4_b558b9",
    "MS-SSIM+L1": "msssim+l1_lr8e-4_bc1d65",
    "SSIM (lr $3{\\times}10^{-3}$)": "ssim_lr3e-3_94f982",
}

DIVERGENT = {
    "SSIM (lr $8{\\times}10^{-4}$)": "ssim_lr8e-4_1b8c15",
    "SSIM+L1": "ssim+l1_lr8e-4_767702",
}

STABLE_COLORS = {
    "L1": "#2196F3",
    "MSE": "#4CAF50",
    "MS-SSIM+L1": "#FF9800",
    "SSIM (lr $3{\\times}10^{-3}$)": "#9C27B0",
}

DIVERGENT_COLORS = {
    "SSIM (lr $8{\\times}10^{-4}$)": "#F44336",
    "SSIM+L1": "#E91E63",
}


def load_epochs(exp_name: str) -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS_DIR / exp_name / "epochs.csv")


def main() -> None:
    fig, (ax_stable, ax_divergent) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Panel (a): Validation SSIM over epochs (comparable across all losses)
    for label, exp_name in STABLE.items():
        df = load_epochs(exp_name)
        ax_stable.plot(
            df["epoch"],
            df["ssim"],
            label=label,
            linewidth=1.8,
            color=STABLE_COLORS[label],
        )
        # Mark best epoch (by validation loss, as used for checkpoint selection)
        best_row = df[df["is_best"] == 1].iloc[-1]
        ax_stable.scatter(
            best_row["epoch"],
            best_row["ssim"],
            color=STABLE_COLORS[label],
            s=40,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )

    ax_stable.set_xlabel("Epoch", fontsize=11)
    ax_stable.set_ylabel("Validation SSIM", fontsize=11)
    ax_stable.set_title("(a) Converged runs", fontsize=12, fontweight="bold")
    ax_stable.legend(fontsize=9, loc="lower right")
    ax_stable.grid(True, alpha=0.3)
    ax_stable.set_xlim(left=0)

    # Panel (b): Validation loss for divergent runs (log scale)
    # Here we DO plot the raw loss because the point is to show the explosion,
    # and both are SSIM-family losses (comparable scale).
    for label, exp_name in DIVERGENT.items():
        df = load_epochs(exp_name)
        ax_divergent.plot(
            df["epoch"],
            df["valid_loss"],
            label=label,
            linewidth=1.8,
            color=DIVERGENT_COLORS[label],
            linestyle="--",
        )

    ax_divergent.set_xlabel("Epoch", fontsize=11)
    ax_divergent.set_ylabel("Validation loss (log scale)", fontsize=11)
    ax_divergent.set_title("(b) Divergent SSIM-family runs", fontsize=12, fontweight="bold")
    ax_divergent.set_yscale("log")
    ax_divergent.legend(fontsize=9, loc="upper left")
    ax_divergent.grid(True, alpha=0.3)
    ax_divergent.set_xlim(left=0)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
