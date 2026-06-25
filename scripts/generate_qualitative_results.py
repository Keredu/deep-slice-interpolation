"""Regenerate the 2x3 qualitative interpolation figures used in the paper.

Produces three PNGs that replace the existing hand-crafted
`qualitative_result_{1,2,3}.png` files referenced by `fig:qual_main` and
`fig:qual_hemorrhage`. The layout preserves the six-cell grid the paper caption
already describes, but with two fixes compared to the legacy figures:

  1. Every panel carries an explicit in-image title (the legacy figures were
     unlabelled, which forced readers to reverse-engineer the caption).
  2. The bottom-right "detail" cell now shows a locator-referenced 2x zoom of
     GT vs. prediction side by side, not a redundant centre crop of the
     prediction alone. A yellow locator rectangle is drawn on the prediction
     panel so the zoom origin is unambiguous.

Model: best-val-SSIM checkpoint of the SSIM-loss run trained at lr 3e-3
(experiment `ssim_lr3e-3_94f982`). The experiment saves only the best epoch;
its on-disk directory is 0-indexed while the training CSV is 1-indexed, so the
checkpoint at `epochs/43/` corresponds to the paper's "epoch 44".

Usage:
    uv run scripts/generate_qualitative_results.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd.datasets.interpolation.two_to_one_slice import (  # noqa: E402
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)
from phd.models.setup_model import setup_model  # noqa: E402
from phd.viz import predict_via_patch_reconstruction  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "train_nn1_cropped" / "ssim_lr3e-3_94f982"

# ROI for the detail zoom: (x, y, size) in 512x512 pixel coords.
# Fixed across all three figures; captures the basal ganglia / ventricular band
# that the paper text highlights as the clinically relevant region.
ROI_XY = (176, 176)
ROI_SIZE = 160

SAMPLES = [
    # (filename, patient_id, triplet_index, title)
    (
        "qualitative_result_1.png",
        "ID_615f69e3",
        14,
        "Normal mid-brain slice (ventricular / basal-ganglia level)",
    ),
    (
        "qualitative_result_2.png",
        "ID_fc4fcd34",
        16,
        "Mid-brain hemorrhage case",
    ),
    (
        "qualitative_result_3.png",
        "ID_615f69e3",
        15,
        "Consecutive normal mid-brain slice (same patient as Fig. 1)",
    ),
]


def infer_data_path() -> Path:
    datasets_dir = os.environ["DATASETS_DIR"] if "DATASETS_DIR" in os.environ else None
    if datasets_dir:
        return Path(datasets_dir) / "pre" / "rsna-intracranial-hemorrhage-detection" / "1x512x512_-20_107"
    raise FileNotFoundError("Cannot find dataset. Set DATASETS_DIR env var.")


def load_model(device: torch.device) -> tuple[torch.nn.Module, Path, int]:
    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,
        model_type="unet",
        encoder_name="tu-tf_efficientnetv2_s",
    )
    epoch_dirs = sorted(EXPERIMENT_DIR.glob("epochs/*/weights.pth"), key=lambda p: int(p.parent.name))
    if not epoch_dirs:
        raise FileNotFoundError(f"No weights.pth found under {EXPERIMENT_DIR}/epochs/")
    weights_path = epoch_dirs[-1]
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval().to(device)
    epoch_on_disk = int(weights_path.parent.name)
    # CSV indexing is 1-based; on-disk dir is 0-based. See module docstring.
    epoch_paper = epoch_on_disk + 1
    return model, weights_path, epoch_paper


def get_patient_triplet(
    dataset: TwoToOneSliceTestDataset,
    patient_id: str,
    triplet_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    for i in range(len(dataset)):
        if dataset.get_patient_id_by_index(i) == patient_id:
            inputs_all, targets_all = dataset[i]
            if triplet_index >= inputs_all.shape[0]:
                raise IndexError(
                    f"triplet {triplet_index} out of range for patient {patient_id} "
                    f"({inputs_all.shape[0]} triplets available)",
                )
            return inputs_all[triplet_index : triplet_index + 1], targets_all[triplet_index : triplet_index + 1]
    raise RuntimeError(f"Patient {patient_id} not found in test split")


def to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().detach().cpu().numpy()


def _strip_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel(ax: plt.Axes, img: np.ndarray, title: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, pad=3)
    _strip_axes(ax)


def _draw_detail(ax: plt.Axes, gt: np.ndarray, pred: np.ndarray) -> None:
    """Bottom-right cell: GT zoom | Prediction zoom, side by side."""
    x, y = ROI_XY
    s = ROI_SIZE
    gt_crop = gt[y : y + s, x : x + s]
    pred_crop = pred[y : y + s, x : x + s]
    separator = np.zeros((s, 4), dtype=gt_crop.dtype)
    combined = np.concatenate([gt_crop, separator, pred_crop], axis=1)
    ax.imshow(combined, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Detail (2$\\times$ zoom): GT  |  Prediction", fontsize=11, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.35")
        spine.set_linewidth(0.8)
    # Sub-labels under each half of the zoom
    ax.text(s / 2, s + 8, "Ground truth", ha="center", va="top", fontsize=9, color="0.2")
    ax.text(s + 4 + s / 2, s + 8, "Prediction", ha="center", va="top", fontsize=9, color="0.2")


def _draw_locator(ax: plt.Axes) -> None:
    x, y = ROI_XY
    s = ROI_SIZE
    rect = Rectangle((x, y), s, s, linewidth=1.0, edgecolor="#FFD60A", facecolor="none")
    ax.add_patch(rect)


def compose_figure(
    *,
    left: np.ndarray,
    gt: np.ndarray,
    right: np.ndarray,
    pred: np.ndarray,
    mean_baseline: np.ndarray,
    out_path: Path,
    title: str,
    model_label: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 8.8),
        gridspec_kw={"wspace": 0.04, "hspace": 0.18},
    )
    _panel(axes[0, 0], left, r"Left input slice $I_{k}$")
    _panel(axes[0, 1], gt, r"Ground truth $I_{k+1}$")
    _panel(axes[0, 2], right, r"Right input slice $I_{k+2}$")
    _panel(axes[1, 0], mean_baseline, r"Mean baseline $\frac{1}{2}(I_{k}+I_{k+2})$")
    _panel(axes[1, 1], pred, f"Prediction — {model_label}")
    _draw_detail(axes[1, 2], gt, pred)
    _draw_locator(axes[1, 1])

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data_path = infer_data_path()
    print(f"Data path: {data_path}")
    dataset = TwoToOneSliceTestDataset(
        root_dir=str(data_path),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    print("Loading model...")
    model, weights_path, epoch_paper = load_model(device)
    print(f"Model weights: {weights_path} (paper epoch {epoch_paper})")
    model_label = f"SSIM loss, lr $3\\!\\times\\!10^{{-3}}$, epoch {epoch_paper}"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, patient_id, triplet_index, title in SAMPLES:
        print(f"  {filename}: {patient_id} triplet={triplet_index}")
        inputs, target = get_patient_triplet(dataset, patient_id, triplet_index)
        inputs = inputs.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred = predict_via_patch_reconstruction(
                model=model,
                batch_inputs=inputs,
                device=device,
            ).clamp(0, 1)
        mean_baseline = inputs[:, 0:1] * 0.5 + inputs[:, 1:2] * 0.5

        out_path = FIGURES_DIR / filename
        compose_figure(
            left=to_np(inputs[:, 0:1]),
            gt=to_np(target),
            right=to_np(inputs[:, 1:2]),
            pred=to_np(pred),
            mean_baseline=to_np(mean_baseline),
            out_path=out_path,
            title=title,
            model_label=model_label,
        )
        print(f"    saved: {out_path}")


if __name__ == "__main__":
    main()
