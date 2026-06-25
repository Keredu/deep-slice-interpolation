"""Generate qualitative comparison panels including VFI baselines.

Produces side-by-side panels for selected patients/slices showing:
  ground truth | our model | RIFE | FILM | mean baseline

Matches the patients used in existing qualitative figures (Fig 1 and Fig 2).

Usage:
    uv run scripts/generate_vfi_qualitative.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd.datasets.interpolation.two_to_one_slice import (  # noqa: E402
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)

# Reuse VFI model loading from evaluate_vfi_baselines
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_vfi_baselines import (  # noqa: E402
    load_film_model,
    load_rife_model,
    film_predict,
    rife_predict,
)

# Patients and slices matching existing qualitative figures
SAMPLES = [
    # (patient_id, triplet_index, description)
    ("ID_615f69e3", 14, "normal anatomy (ventricular level)"),
    ("ID_fc4fcd34", 16, "hemorrhage case"),
]

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def infer_data_path() -> Path:
    """Infer dataset path from DATASETS_DIR."""
    datasets_dir = os.environ.get("DATASETS_DIR")
    if datasets_dir:
        return Path(datasets_dir) / "pre" / "rsna-intracranial-hemorrhage-detection" / "1x512x512_-20_107"
    raise FileNotFoundError("Cannot find dataset. Set DATASETS_DIR env var.")


def load_our_model(device: torch.device) -> torch.nn.Module:
    """Load our best SSIM model (ssim_lr3e-3_94f982)."""
    from phd.models.setup_model import setup_model

    model = setup_model(
        in_channels=2, out_channels=1, pretrained=False,
        model_type="unet", encoder_name="tu-tf_efficientnetv2_s",
    )
    exp_dir = PROJECT_ROOT / "experiments" / "train_nn1_cropped" / "ssim_lr3e-3_94f982"
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")
    # Find best epoch weights (highest epoch number with weights.pth)
    epoch_dirs = sorted(exp_dir.glob("epochs/*/weights.pth"), key=lambda p: int(p.parent.name))
    if not epoch_dirs:
        raise FileNotFoundError(f"No weights.pth found in {exp_dir}/epochs/")
    weights_path = epoch_dirs[-1]  # Last (best) epoch

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"]
    # Strip _orig_mod. prefix from torch.compile
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def make_comparison_panel(
    *,
    patient_id: str,
    triplet_index: int,
    description: str,
    dataset: TwoToOneSliceTestDataset,
    our_model: torch.nn.Module,
    rife_model,
    film_model,
    device: torch.device,
    output_path: Path,
) -> None:
    """Generate a single comparison panel for one slice."""
    # Find patient index
    patient_idx = None
    for i in range(len(dataset)):
        if dataset.get_patient_id_by_index(i) == patient_id:
            patient_idx = i
            break

    if patient_idx is None:
        print(f"WARNING: Could not find patient {patient_id}, skipping")
        return

    inputs_all, targets_all = dataset[patient_idx]
    # inputs_all: (N_triplets, 2, H, W), targets_all: (N_triplets, 1, H, W)

    if triplet_index >= inputs_all.shape[0]:
        print(f"WARNING: triplet {triplet_index} out of range ({inputs_all.shape[0]} triplets), skipping")
        return

    inputs = inputs_all[triplet_index:triplet_index + 1].to(device)  # (1, 2, H, W)
    target = targets_all[triplet_index:triplet_index + 1].to(device)  # (1, 1, H, W)

    # Our model prediction (13-patch reconstruction; matches training distribution)
    from phd.viz import predict_via_patch_reconstruction
    with torch.no_grad():
        our_pred = predict_via_patch_reconstruction(
            model=our_model, batch_inputs=inputs, device=device
        ).clamp(0, 1)

    # Mean baseline
    mean_pred = inputs[:, 0:1] * 0.5 + inputs[:, 1:2] * 0.5

    # VFI predictions
    rife_pred = rife_predict(rife_model, inputs, device)
    film_pred = film_predict(film_model, inputs, device)

    # Convert to numpy
    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.squeeze().cpu().numpy()

    gt = to_np(target)
    ours = to_np(our_pred)
    rife_out = to_np(rife_pred)
    film_out = to_np(film_pred)
    mean_out = to_np(mean_pred)
    left = to_np(inputs[:, 0:1])
    right = to_np(inputs[:, 1:2])

    # Create figure: 2 rows x 4 cols
    # Row 1: left input | ground truth | right input | (empty)
    # Row 2: our model | RIFE | FILM | mean baseline
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

    # Row 1
    axes[0, 0].imshow(left, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Left input slice", fontsize=12)
    axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground truth", fontsize=12)
    axes[0, 2].imshow(right, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Right input slice", fontsize=12)
    axes[0, 3].axis("off")

    # Row 2
    axes[1, 0].imshow(ours, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Ours (SSIM@3e-3)", fontsize=12)
    axes[1, 1].imshow(rife_out, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("RIFE", fontsize=12)
    axes[1, 2].imshow(film_out, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("FILM", fontsize=12)
    axes[1, 3].imshow(mean_out, cmap="gray", vmin=0, vmax=1)
    axes[1, 3].set_title("Mean baseline", fontsize=12)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{description}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load dataset
    data_path = infer_data_path()
    print(f"Data path: {data_path}")
    dataset = TwoToOneSliceTestDataset(
        root_dir=str(data_path),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    # Load models
    print("Loading models...")
    our_model = load_our_model(device)
    rife_model = load_rife_model(device)
    film_model = load_film_model(device)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for patient_id, triplet_index, description in SAMPLES:
        safe_name = f"vfi_comparison_{patient_id}_{triplet_index}.png"
        output_path = FIGURES_DIR / safe_name
        make_comparison_panel(
            patient_id=patient_id,
            triplet_index=triplet_index,
            description=description,
            dataset=dataset,
            our_model=our_model,
            rife_model=rife_model,
            film_model=film_model,
            device=device,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
