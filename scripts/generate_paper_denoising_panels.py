#!/usr/bin/env python3
"""Generate denoising comparison panels for the paper.

For each patient, shows 10 consecutive slices in 3 columns:
  Column 1: Original (noisy) acquired slice
  Column 2: Denoised version (model output from neighboring slices)
  Column 3: |Original - Denoised| difference map (the removed noise)

The "denoised" version of slice i is the model's prediction when given
slices i-1 and i+1 as input — the model predicts the conditional expectation,
which is the underlying anatomy without acquisition noise.

Usage:
    uv run scripts/generate_paper_denoising_panels.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
OUTPUT_DIR = Path("results/figures")

# Use the reference model (MS-SSIM+L1) for denoising
MODEL_EXP = "msssim+l1_lr8e-4_bc1d65"

# Patients to visualize
PATIENTS = {
    "hemorrhage": {"idx": 28, "id": "ID_fc4fcd34", "start_slice": 5, "n_slices": 10},
    "normal": {"idx": 6, "id": "ID_615f69e3", "start_slice": 5, "n_slices": 10},
}


def load_model(experiment_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained model from its experiment directory."""
    import pandas as pd

    config = json.loads((experiment_dir / "config.json").read_text())
    model_cfg = config["model"]

    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,
        model_type=model_cfg["type"],
        encoder_name=model_cfg["encoder_name"],
    )

    # Find best epoch checkpoint
    df = pd.read_csv(experiment_dir / "epochs.csv")
    finite = df[np.isfinite(df["valid_loss"])]
    best_epoch = int(finite.loc[finite["valid_loss"].idxmin(), "epoch"])
    weights_path = experiment_dir / "epochs" / str(best_epoch - 1) / "weights.pth"

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def denoise_slice(
    model: torch.nn.Module,
    prev_slice: torch.Tensor,
    next_slice: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Denoise the middle slice by predicting it from its neighbors.

    Args:
        model: Trained interpolation model
        prev_slice: Slice i-1, shape (1, H, W)
        next_slice: Slice i+1, shape (1, H, W)
        device: Torch device

    Returns:
        Denoised slice as numpy array (H, W)
    """
    # Stack as 2-channel input: (1, 2, H, W)
    inp = torch.stack([prev_slice[0], next_slice[0]], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = (
            predict_via_patch_reconstruction(model=model, batch_inputs=inp, device=device)
            .cpu()
            .squeeze()
            .numpy()
        )
    return pred


def generate_panel(
    model: torch.nn.Module,
    test_dataset: TwoToOneSliceTestDataset,
    patient_cfg: dict,
    device: torch.device,
    output_path: Path,
    label: str,
) -> None:
    """Generate the Nx3 denoising panel for one patient."""
    inputs, targets = test_dataset[patient_cfg["idx"]]
    # inputs: (N_triplets, 2, 512, 512) — channel 0 = slice i, channel 1 = slice i+2
    # targets: (N_triplets, 1, 512, 512) — slice i+1 (the real middle slice)

    n_slices = patient_cfg["n_slices"]
    start = patient_cfg["start_slice"]

    # For denoising slice i+1, we need:
    #   - triplet i gives us (slice_i, slice_{i+2}) and target slice_{i+1}
    #   - The "original" is targets[i] = slice_{i+1}
    #   - The "denoised" = model(slice_i, slice_{i+2}) = prediction of slice_{i+1}
    # This is exactly what the model already does! The target IS the original noisy
    # slice, and the prediction IS the denoised version.

    fig, axes = plt.subplots(n_slices, 3, figsize=(8, 2.5 * n_slices))

    # Collect all differences first to get consistent colormap scale
    all_diffs = []
    originals = []
    denoised_list = []

    for row, triplet_idx in enumerate(range(start, start + n_slices)):
        if triplet_idx >= inputs.shape[0]:
            break

        original = targets[triplet_idx, 0].numpy()  # (512, 512)
        denoised = denoise_slice(
            model,
            inputs[triplet_idx, 0:1],  # slice i
            inputs[triplet_idx, 1:2],  # slice i+2
            device,
        )
        diff = np.abs(original - denoised)

        originals.append(original)
        denoised_list.append(denoised)
        all_diffs.append(diff)

    vmax_diff = np.percentile(np.stack(all_diffs), 99)

    for row in range(len(originals)):
        # Column 1: Original noisy slice
        axes[row, 0].imshow(originals[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=11, fontweight="bold")

        # Column 2: Denoised slice
        axes[row, 1].imshow(denoised_list[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title("Denoised", fontsize=11, fontweight="bold")

        # Column 3: Difference map
        im = axes[row, 2].imshow(all_diffs[row], cmap="viridis", vmin=0, vmax=vmax_diff)
        axes[row, 2].axis("off")
        if row == 0:
            axes[row, 2].set_title("|Original \u2212 Denoised|", fontsize=11, fontweight="bold")

        # Slice label on the left
        axes[row, 0].set_ylabel(f"Slice {start + row + 1}", fontsize=9, rotation=0, labelpad=40, va="center")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Absolute difference")

    fig.subplots_adjust(left=0.08, right=0.90, top=0.95, bottom=0.02, hspace=0.05, wspace=0.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path} ({label})")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    exp_dir = EXPERIMENTS_DIR / MODEL_EXP
    model = load_model(exp_dir, device)

    # Load dataset
    config = json.loads((exp_dir / "config.json").read_text())
    test_dataset = TwoToOneSliceTestDataset(
        root_dir=resolve_config_path(config["data_path"]),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    for label, patient_cfg in PATIENTS.items():
        output_path = OUTPUT_DIR / f"denoising_{label}.pdf"
        generate_panel(model, test_dataset, patient_cfg, device, output_path, label)


if __name__ == "__main__":
    main()
