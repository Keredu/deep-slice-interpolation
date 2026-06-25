#!/usr/bin/env python3
"""Generate error map figure for the paper.

For a selected test slice, shows ground truth, predictions from each loss,
and pixel-wise absolute error maps. This visualizes WHERE different losses
produce different errors.

Usage:
    uv run scripts/generate_paper_error_maps.py
    uv run scripts/generate_paper_error_maps.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from phd.config_io import resolve_config_path
from phd.datasets.interpolation.two_to_one_slice import (
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)
from phd.models.setup_model import setup_model
from phd.viz import predict_via_patch_reconstruction

EXPERIMENTS_DIR = Path("experiments/train_nn1_cropped")
OUTPUT_PATH = Path("results/figures/error_maps.pdf")

# Models to compare (label -> experiment dir name)
MODELS = {
    "SSIM": "ssim_lr3e-3_94f982",
    "L1": "l1_lr8e-4_b39be9",
    "MS-SSIM+L1": "msssim+l1_lr8e-4_bc1d65",
    "MSE": "mse_lr8e-4_b558b9",
}

# Patient and slice to visualize (same as paper's Figure 2)
PATIENT_IDX = 6  # ID_615f69e3
SLICE_IDX = 14


def resolve_best_weights(experiment_dir: Path) -> Path:
    """Find best checkpoint (lowest validation loss)."""
    df = pd.read_csv(experiment_dir / "epochs.csv")
    finite = df[np.isfinite(df["valid_loss"])]
    best_epoch_1based = int(finite.loc[finite["valid_loss"].idxmin(), "epoch"])
    weights = experiment_dir / "epochs" / str(best_epoch_1based - 1) / "weights.pth"
    if not weights.exists():
        msg = f"Weights not found: {weights}"
        raise FileNotFoundError(msg)
    return weights


def load_model(experiment_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained model from its experiment directory."""
    config = json.loads((experiment_dir / "config.json").read_text())
    model_cfg = config["model"]

    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,
        model_type=model_cfg["type"],
        encoder_name=model_cfg["encoder_name"],
    )

    weights_path = resolve_best_weights(experiment_dir)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Remove torch.compile wrapper prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--patient", type=int, default=PATIENT_IDX)
    parser.add_argument("--slice", type=int, default=SLICE_IDX)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load test dataset
    config = json.loads(
        (EXPERIMENTS_DIR / "ssim_lr3e-3_94f982" / "config.json").read_text()
    )
    test_dataset = TwoToOneSliceTestDataset(
        root_dir=resolve_config_path(config["data_path"]),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    # Get patient data
    inputs, targets = test_dataset[args.patient]
    patient_id = test_dataset.get_patient_id_by_index(args.patient)

    slice_idx = min(args.slice, inputs.shape[0] - 1)
    input_tensor = inputs[slice_idx : slice_idx + 1].to(device)  # (1, 2, 512, 512)
    target_np = targets[slice_idx, 0].numpy()  # (512, 512)

    print(f"Patient: {patient_id}, slice: {slice_idx}")
    print(f"Input shape: {input_tensor.shape}, target range: [{target_np.min():.3f}, {target_np.max():.3f}]")

    # Run inference for each model
    predictions = {}
    errors = {}
    for label, exp_name in MODELS.items():
        exp_dir = EXPERIMENTS_DIR / exp_name
        model = load_model(exp_dir, device)
        with torch.no_grad():
            pred = (
                predict_via_patch_reconstruction(
                    model=model, batch_inputs=input_tensor, device=device
                )
                .cpu()
                .squeeze()
                .numpy()
            )  # (512, 512)
        predictions[label] = pred
        errors[label] = np.abs(pred - target_np)
        print(f"  {label}: pred range [{pred.min():.3f}, {pred.max():.3f}], MAE={errors[label].mean():.4f}")
        del model
        torch.cuda.empty_cache()

    # Determine global error scale for consistent colormap
    all_errors = np.stack(list(errors.values()))
    vmax = np.percentile(all_errors, 99)  # Clip top 1% for visibility

    # Create figure: 2 rows x 5 columns
    # Row 1: Ground truth + 4 predictions
    # Row 2: blank + 4 error maps
    n_models = len(MODELS)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(3.2 * (n_models + 1), 6.4))

    # Row 1, Col 0: Ground truth
    axes[0, 0].imshow(target_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Ground truth", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # Row 2, Col 0: Input (mean of two input slices for context)
    input_mean = input_tensor[0].cpu().mean(dim=0).numpy()
    axes[1, 0].imshow(input_mean, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Mean input", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    # Columns 1-4: Predictions and error maps
    for i, (label, pred) in enumerate(predictions.items()):
        col = i + 1

        # Row 1: Prediction
        axes[0, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(label, fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        # Row 2: Error map
        im = axes[1, col].imshow(errors[label], cmap="viridis", vmin=0, vmax=vmax)
        mae_val = errors[label].mean()
        axes[1, col].set_title(f"|error| (MAE={mae_val:.4f})", fontsize=9)
        axes[1, col].axis("off")

    # Add colorbar for error maps
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.35])
    fig.colorbar(im, cax=cbar_ax, label="Absolute error")

    fig.tight_layout(rect=[0, 0, 0.91, 1.0])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
