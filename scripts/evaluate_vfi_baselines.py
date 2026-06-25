"""Evaluate pretrained Video Frame Interpolation (VFI) baselines on CT test set.

Supports RIFE and FILM — two ECCV 2022 VFI methods. Downloads pretrained
weights automatically and evaluates using the same metrics pipeline as the
main experiments.

Usage (on GPU machine):
    # Run both baselines:
    uv run scripts/evaluate_vfi_baselines.py --device cuda --batch-size 8

    # Run only one:
    uv run scripts/evaluate_vfi_baselines.py --device cuda --model rife
    uv run scripts/evaluate_vfi_baselines.py --device cuda --model film

Outputs per model:
    results/tables/{model}_baseline_slice_metrics.csv
    results/tables/{model}_baseline_patient_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from loguru import logger
from pytorch_msssim import MS_SSIM, SSIM
from scipy.stats import wilcoxon

from phd.config_io import resolve_config_path
from phd.datasets.interpolation.two_to_one_slice import (
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VENDOR_DIR = Path("vendor")
METRIC_COLUMNS = ["ssim", "ms_ssim", "mae", "gradient_mae", "psnr", "ncc", "lpips"]
TABLES_DIR = Path("results/tables")

RIFE_REPO_URL = "https://github.com/hzwer/Practical-RIFE.git"
RIFE_DIR = VENDOR_DIR / "RIFE"
RIFE_GDRIVE_ID = "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_"

FILM_DIR = VENDOR_DIR / "FILM"
FILM_WEIGHTS_URL = "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt"


# ---------------------------------------------------------------------------
# RIFE setup & inference
# ---------------------------------------------------------------------------
def ensure_rife_available() -> Path:
    """Clone Practical-RIFE repo and download pretrained HD weights."""
    if not RIFE_DIR.exists():
        logger.info(f"Cloning RIFE repository to {RIFE_DIR}")
        RIFE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", RIFE_REPO_URL, str(RIFE_DIR)],
        )

    weights_dir = RIFE_DIR / "train_log"
    flownet_path = weights_dir / "flownet.pkl"
    if not flownet_path.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        gdrive_url = f"https://drive.google.com/uc?id={RIFE_GDRIVE_ID}"
        download_path = weights_dir / "rife_hd_download"

        logger.info("Downloading RIFE HD pretrained weights from Google Drive...")
        subprocess.check_call([
            sys.executable, "-m", "gdown",
            "--fuzzy", gdrive_url,
            "-O", str(download_path),
        ])

        if not download_path.exists() or download_path.stat().st_size < 1_000_000:
            download_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download failed. Please download manually from:\n"
                f"  https://drive.google.com/file/d/{RIFE_GDRIVE_ID}/view\n"
                f"Extract and place flownet.pkl in: {weights_dir}"
            )

        import shutil
        import zipfile
        if zipfile.is_zipfile(download_path):
            logger.info("Extracting zip archive...")
            with zipfile.ZipFile(download_path, "r") as zf:
                zf.extractall(weights_dir)
            download_path.unlink()
            # The zip may contain a nested train_log/ directory.  Flatten it so
            # that RIFE_HDv3.py, IFNet_HDv3.py and flownet.pkl live directly
            # inside weights_dir (= vendor/RIFE/train_log/).
            nested = weights_dir / "train_log"
            if nested.is_dir():
                for item in nested.iterdir():
                    dest = weights_dir / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))
                shutil.rmtree(nested, ignore_errors=True)
            # Also remove __MACOSX junk if present
            macosx = weights_dir / "__MACOSX"
            if macosx.is_dir():
                shutil.rmtree(macosx, ignore_errors=True)
            if not flownet_path.exists():
                for candidate in weights_dir.rglob("flownet.pkl"):
                    shutil.move(str(candidate), str(flownet_path))
                    break
        else:
            download_path.rename(flownet_path)

        if not flownet_path.exists():
            raise FileNotFoundError(
                f"flownet.pkl not found. Contents: {[p.name for p in weights_dir.iterdir()]}\n"
                f"Download manually: https://drive.google.com/file/d/{RIFE_GDRIVE_ID}/view"
            )
        logger.info(f"RIFE weights ready at {flownet_path}")
    return RIFE_DIR


def load_rife_model(device: torch.device) -> Any:
    """Load RIFE IFNet model with pretrained weights."""
    rife_dir = ensure_rife_available()

    rife_root = str(rife_dir)
    if rife_root not in sys.path:
        sys.path.insert(0, rife_root)

    # Practical-RIFE ships the model definition inside train_log/ alongside weights
    try:
        from train_log.RIFE_HDv3 import Model  # type: ignore[import-not-found]
    except ImportError:
        from model.RIFE_HDv3 import Model  # type: ignore[import-not-found]

    model = Model()
    train_log_dir = str(rife_dir / "train_log")
    flownet_path = f"{train_log_dir}/flownet.pkl"

    def convert(param: dict) -> dict:
        return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

    state_dict = torch.load(flownet_path, map_location="cpu", weights_only=False)
    model.flownet.load_state_dict(convert(state_dict))
    model.eval()
    model.flownet.to(device)
    logger.info("RIFE model loaded successfully")
    return model


def rife_predict(
    model: Any, inputs: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """Run RIFE inference: (B,2,H,W) grayscale → (B,1,H,W) prediction."""
    frame0 = inputs[:, 0:1, :, :].repeat(1, 3, 1, 1)
    frame1 = inputs[:, 1:2, :, :].repeat(1, 3, 1, 1)

    _, _, h, w = frame0.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        frame0 = functional.pad(frame0, (0, pad_w, 0, pad_h), mode="reflect")
        frame1 = functional.pad(frame1, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        result = model.inference(frame0.to(device), frame1.to(device))

    if pad_h > 0 or pad_w > 0:
        result = result[:, :, :h, :w]

    return result.mean(dim=1, keepdim=True).clamp(0, 1)


# ---------------------------------------------------------------------------
# FILM setup & inference
# ---------------------------------------------------------------------------
def ensure_film_available() -> Path:
    """Download FILM TorchScript model if not present."""
    FILM_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = FILM_DIR / "film_net_fp32.pt"
    if not weights_path.exists():
        logger.info("Downloading FILM pretrained weights...")
        subprocess.check_call([
            "wget", "-q", FILM_WEIGHTS_URL, "-O", str(weights_path),
        ])
        if not weights_path.exists() or weights_path.stat().st_size < 1_000_000:
            weights_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"FILM download failed. Download manually from:\n  {FILM_WEIGHTS_URL}\n"
                f"Place in: {weights_path}"
            )
        logger.info(f"FILM weights ready at {weights_path}")
    return weights_path


def load_film_model(device: torch.device) -> torch.jit.ScriptModule:
    """Load FILM TorchScript model."""
    weights_path = ensure_film_available()
    model = torch.jit.load(str(weights_path), map_location="cpu")
    model.eval().to(device=device, dtype=torch.float32)
    logger.info("FILM model loaded successfully")
    return model


def film_predict(
    model: torch.jit.ScriptModule, inputs: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """Run FILM inference: (B,2,H,W) grayscale → (B,1,H,W) prediction."""
    frame0 = inputs[:, 0:1, :, :].repeat(1, 3, 1, 1)
    frame1 = inputs[:, 1:2, :, :].repeat(1, 3, 1, 1)

    # FILM expects inputs in [0, 1], (B, 3, H, W)
    # dt=0.5 for midpoint interpolation
    dt = frame0.new_full((frame0.shape[0], 1), 0.5)

    with torch.no_grad():
        result = model(frame0.to(device), frame1.to(device), dt.to(device))

    return result.mean(dim=1, keepdim=True).clamp(0, 1)


# ---------------------------------------------------------------------------
# Metrics (same as build_revision_tables.py)
# ---------------------------------------------------------------------------
def _sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    grad_x = functional.conv2d(x, sobel_x, padding=0)
    grad_y = functional.conv2d(x, sobel_y, padding=0)
    return torch.sqrt(grad_x.square() + grad_y.square())


def compute_metrics_per_sample(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    ssim_module: SSIM,
    msssim_module: MS_SSIM,
    lpips_module: lpips.LPIPS,
) -> dict[str, np.ndarray]:
    diff = pred - target
    mae_vals = diff.abs().mean(dim=(1, 2, 3))
    mse_vals = diff.square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    psnr_vals = 10.0 * torch.log10(1.0 / mse_vals)

    pred_grad = _sobel_gradient_magnitude(pred)
    target_grad = _sobel_gradient_magnitude(target)
    gradient_mae_vals = (pred_grad - target_grad).abs().mean(dim=(1, 2, 3))

    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    pred_std = pred_centered.std(dim=1) + 1e-8
    target_std = target_centered.std(dim=1) + 1e-8
    n_pixels = pred_flat.shape[1]
    ncc_vals = (pred_centered * target_centered).sum(dim=1) / (
        pred_std * target_std * n_pixels
    )

    ssim_vals = ssim_module(pred, target)
    msssim_vals = msssim_module(pred, target)

    pred_3ch = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
    target_3ch = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
    lpips_vals = lpips_module(pred_3ch, target_3ch).squeeze()
    if lpips_vals.dim() == 0:
        lpips_vals = lpips_vals.unsqueeze(0)

    return {
        "ssim": ssim_vals.detach().cpu().numpy(),
        "ms_ssim": msssim_vals.detach().cpu().numpy(),
        "mae": mae_vals.detach().cpu().numpy(),
        "gradient_mae": gradient_mae_vals.detach().cpu().numpy(),
        "psnr": psnr_vals.detach().cpu().numpy(),
        "ncc": ncc_vals.detach().cpu().numpy(),
        "lpips": lpips_vals.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Generic evaluation loop
# ---------------------------------------------------------------------------
def evaluate_vfi_on_test(
    *,
    model_name: str,
    predict_fn: Any,
    model: Any,
    data_path: Path,
    device: torch.device,
    batch_size: int,
    ssim_module: SSIM,
    msssim_module: MS_SSIM,
    lpips_module: lpips.LPIPS,
) -> pd.DataFrame:
    """Evaluate a VFI model on the full test set."""
    datasets_dir = os.getenv("DATASETS_DIR")
    if not datasets_dir:
        parts = list(data_path.resolve().parts)
        try:
            pre_idx = parts.index("pre")
            os.environ["DATASETS_DIR"] = str(Path(*parts[:pre_idx]))
        except ValueError:
            pass

    test_dataset = TwoToOneSliceTestDataset(
        root_dir=str(data_path),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    experiment_name = f"baseline_{model_name}"
    rows: list[dict[str, Any]] = []
    total_patients = len(test_dataset)
    logger.info(f"[{model_name.upper()}] Evaluating on {total_patients} test patients")

    for patient_idx in range(total_patients):
        patient_id = test_dataset.get_patient_id_by_index(patient_idx)
        inputs, targets = test_dataset[patient_idx]
        n_triplets = inputs.shape[0]

        for start in range(0, n_triplets, batch_size):
            end = min(start + batch_size, n_triplets)
            batch_inputs = inputs[start:end].to(device)
            batch_targets = targets[start:end].to(device)

            batch_pred = predict_fn(model, batch_inputs, device)

            metric_vectors = compute_metrics_per_sample(
                pred=batch_pred,
                target=batch_targets,
                ssim_module=ssim_module,
                msssim_module=msssim_module,
                lpips_module=lpips_module,
            )

            for local_idx in range(end - start):
                row: dict[str, Any] = {
                    "experiment": experiment_name,
                    "patient_id": patient_id,
                    "triplet_index": start + local_idx,
                }
                for metric_name in METRIC_COLUMNS:
                    row[metric_name] = float(metric_vectors[metric_name][local_idx])
                rows.append(row)

        if (patient_idx + 1) % 5 == 0 or (patient_idx + 1) == total_patients:
            logger.info(
                f"[{model_name.upper()}] Processed {patient_idx + 1}/{total_patients} patients"
            )

    df = pd.DataFrame(rows)
    logger.info(f"[{model_name.upper()}] Produced {len(df)} per-slice metric rows")
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    values: np.ndarray, n_samples: int = 2000, seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        values[rng.integers(0, len(values), size=len(values))].mean()
        for _ in range(n_samples)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def compute_patient_summary(
    per_slice_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    patient_means = per_slice_df.groupby("patient_id")[METRIC_COLUMNS].mean()

    summary: dict[str, Any] = {"experiment": per_slice_df["experiment"].iloc[0]}
    summary["n_patients"] = len(patient_means)
    summary["n_slices"] = len(per_slice_df)

    for metric in METRIC_COLUMNS:
        vals = patient_means[metric].values
        summary[f"{metric}_mean"] = float(vals.mean())
        ci_low, ci_high = bootstrap_mean_ci(vals)
        summary[f"{metric}_ci_low"] = ci_low
        summary[f"{metric}_ci_high"] = ci_high

    if reference_df is not None:
        ref_patient_means = reference_df.groupby("patient_id")[METRIC_COLUMNS].mean()
        common_patients = patient_means.index.intersection(ref_patient_means.index)
        for metric in METRIC_COLUMNS:
            cand = patient_means.loc[common_patients, metric].values
            ref = ref_patient_means.loc[common_patients, metric].values
            diff = cand - ref
            summary[f"{metric}_delta"] = float(diff.mean())
            _, p_val = wilcoxon(diff, alternative="two-sided")
            summary[f"{metric}_p"] = float(p_val)

    return pd.DataFrame([summary])


def print_summary(summary_df: pd.DataFrame, model_name: str) -> None:
    logger.info("=" * 60)
    logger.info(f"{model_name.upper()} Baseline Results (patient-level)")
    logger.info("=" * 60)
    for metric in METRIC_COLUMNS:
        mean = summary_df[f"{metric}_mean"].iloc[0]
        ci_low = summary_df[f"{metric}_ci_low"].iloc[0]
        ci_high = summary_df[f"{metric}_ci_high"].iloc[0]
        logger.info(f"  {metric:>12s}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VFI baselines (RIFE, FILM) on CT test set.",
    )
    p.add_argument(
        "--model", type=str, default="all", choices=["rife", "film", "all"],
        help="Which VFI model to evaluate (default: all).",
    )
    p.add_argument(
        "--data-path", type=Path, default=None,
        help="Path to dataset. If not set, inferred from experiment config.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--reference-csv", type=Path, default=None,
        help="Path to reference model per-slice CSV for paired tests.",
    )
    p.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Resolve data path
    if args.data_path is None:
        exp_dir = Path("experiments/train_nn1_cropped")
        configs = list(exp_dir.glob("*/config.json"))
        if not configs:
            logger.error("No --data-path and no experiment configs found.")
            sys.exit(1)
        config = json.loads(configs[0].read_text(encoding="utf-8"))
        args.data_path = resolve_config_path(config["data_path"])
        logger.info(f"Inferred data path: {args.data_path}")

    # Setup metric modules (shared across models)
    ssim_module = SSIM(data_range=1.0, size_average=False, channel=1).to(device)
    msssim_module = MS_SSIM(data_range=1.0, size_average=False, channel=1).to(device)
    lpips_module = lpips.LPIPS(net="alex").to(device)
    lpips_module.eval()

    # Load reference CSV if provided
    reference_df = None
    if args.reference_csv is not None and args.reference_csv.exists():
        reference_df = pd.read_csv(args.reference_csv)
        reference_df = reference_df[
            reference_df["experiment"] == reference_df["experiment"].iloc[0]
        ]
        logger.info(f"Loaded reference CSV: {args.reference_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run: list[str] = []
    if args.model == "all":
        models_to_run = ["rife", "film"]
    else:
        models_to_run = [args.model]

    all_summaries = []

    for model_name in models_to_run:
        logger.info(f"\n{'='*60}\nEvaluating {model_name.upper()}\n{'='*60}")

        if model_name == "rife":
            model = load_rife_model(device)
            predict_fn = rife_predict
        else:
            model = load_film_model(device)
            predict_fn = film_predict

        per_slice_df = evaluate_vfi_on_test(
            model_name=model_name,
            predict_fn=predict_fn,
            model=model,
            data_path=args.data_path,
            device=device,
            batch_size=args.batch_size,
            ssim_module=ssim_module,
            msssim_module=msssim_module,
            lpips_module=lpips_module,
        )

        # Save per-slice
        slice_csv = args.output_dir / f"{model_name}_baseline_slice_metrics.csv"
        per_slice_df.to_csv(slice_csv, index=False)
        logger.info(f"Saved: {slice_csv}")

        # Patient summary
        summary_df = compute_patient_summary(per_slice_df, reference_df)
        summary_csv = args.output_dir / f"{model_name}_baseline_patient_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved: {summary_csv}")

        print_summary(summary_df, model_name)
        all_summaries.append(summary_df)

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Combined summary
    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_csv = args.output_dir / "vfi_baselines_patient_summary.csv"
        combined.to_csv(combined_csv, index=False)
        logger.info(f"\nCombined summary saved: {combined_csv}")


if __name__ == "__main__":
    main()
