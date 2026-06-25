"""Build revision tables for Round 1 reviewer response.

Adds three capabilities beyond the original build_paper_tables.py:
  B1: Cubic z-interpolation baseline (uses 4 neighboring slices when available)
  B2: ancillary LPIPS perceptual metric for raw artefacts
  B4: Hemorrhage-vs-normal test stratification

Outputs are written to `results/tables/` with a revision date tag.

Usage:
    uv run scripts/build_revision_tables.py \
        --device cuda --batch-size 16 \
        --snapshot-date 2026-03-03 \
        --reference-experiment msssim+l1_lr8e-4_bc1d65 \
        --test-experiment baseline_mean \
        --test-experiment baseline_cubic \
        --test-experiment ssim_lr3e-3_94f982 \
        --test-experiment l1_lr8e-4_b39be9 \
        --test-experiment msssim+l1_lr8e-4_bc1d65 \
        --test-experiment mse_lr8e-4_b558b9 \
        --test-experiment l1_lr1e-4_66e5e2 \
        --test-experiment msssim+l1_lr8e-4_e6d845 \
        --test-experiment ssim_lr8e-4_1b8c15
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from loguru import logger
from PIL import Image
from pytorch_msssim import MS_SSIM, SSIM
from scipy.interpolate import interp1d
from scipy.stats import false_discovery_control, wilcoxon

from phd.config_io import resolve_config_path
from phd.datasets.interpolation.two_to_one_slice import STANDARD_TRANSFORM, TwoToOneSliceTestDataset
from phd.models.setup_model import setup_model
from phd.viz import predict_via_patch_reconstruction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIGHER_IS_BETTER = {"ssim", "ms_ssim", "psnr"}
LOWER_IS_BETTER = {"mae", "gradient_mae"}
METRIC_COLUMNS = ["ssim", "ms_ssim", "mae", "gradient_mae", "psnr"]

IH_SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Build revision tables (B1/B2/B4) for Round 1 response.")
    p.add_argument("--experiments-dir", type=Path, default=Path("experiments/train_nn1_cropped"))
    p.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    today = datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    p.add_argument("--snapshot-date", type=str, default=today)
    p.add_argument("--test-experiment", action="append", default=[])
    p.add_argument("--reference-experiment", type=str, default="msssim+l1_lr8e-4_bc1d65")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (reused from build_paper_tables.py)
# ---------------------------------------------------------------------------
def _infer_datasets_dir(data_path: Path) -> Path:
    """Infer DATASETS_DIR from a configured data_path."""
    parts = list(data_path.resolve().parts)
    try:
        pre_idx = parts.index("pre")
    except ValueError as exc:
        raise RuntimeError(f"Cannot infer DATASETS_DIR from data_path={data_path}.") from exc
    return Path(*parts[:pre_idx])


def load_model_from_checkpoint(
    experiment_dir: Path,
    weights_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    """Load model and config for inference."""
    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    model_cfg = config.get("model", {})
    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,
        model_type=model_cfg.get("type", "unet"),
        encoder_name=model_cfg.get("encoder_name"),
    )
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, resolve_config_path(config["data_path"])


def _resolve_best_weights(experiment_dir: Path) -> Path | None:
    """Resolve best checkpoint weights path from epochs.csv."""
    epochs_csv = experiment_dir / "epochs.csv"
    if not epochs_csv.exists():
        return None
    df = pd.read_csv(epochs_csv)
    finite = df[np.isfinite(df["valid_loss"])]
    if finite.empty:
        return None
    best_epoch_1based = int(finite.loc[finite["valid_loss"].idxmin(), "epoch"])
    epoch_dir = experiment_dir / "epochs" / str(best_epoch_1based - 1)
    weights = epoch_dir / "weights.pth"
    if weights.exists():
        return weights
    # Fallback: try latest available
    epochs_dir = experiment_dir / "epochs"
    if not epochs_dir.exists():
        return None
    available = sorted(int(d.name) for d in epochs_dir.iterdir() if d.is_dir() and (d / "weights.pth").exists())
    if not available:
        return None
    return epochs_dir / str(available[-1]) / "weights.pth"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude for a batch of grayscale images."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
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
    """Compute metric vectors, including ancillary NCC/LPIPS raw artefact columns."""
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
    ncc_vals = (pred_centered * target_centered).sum(dim=1) / (pred_std * target_std * n_pixels)

    ssim_vals = ssim_module(pred, target)
    msssim_vals = msssim_module(pred, target)

    # LPIPS expects 3-channel input in [-1, 1]
    pred_3ch = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
    target_3ch = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
    lpips_vals = lpips_module(pred_3ch, target_3ch).squeeze()
    # Handle single-sample batch (squeeze removes all dims)
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
# Test dataset with hemorrhage labels
# ---------------------------------------------------------------------------
def _load_test_dataset(data_path: Path) -> TwoToOneSliceTestDataset:
    """Load the test dataset, ensuring DATASETS_DIR is set."""
    datasets_dir = os.getenv("DATASETS_DIR")
    if not datasets_dir:
        inferred = _infer_datasets_dir(data_path)
        os.environ["DATASETS_DIR"] = str(inferred)
        logger.info(f"DATASETS_DIR inferred as {inferred}")
    return TwoToOneSliceTestDataset(
        root_dir=str(data_path),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )


def _build_triplet_hemorrhage_labels(test_dataset: TwoToOneSliceTestDataset) -> pd.DataFrame:
    """Build a DataFrame mapping (patient_id, triplet_index) → hemorrhage labels.

    For each triplet, reports:
      - target_any: 1 if target (middle) slice has any hemorrhage
      - target_subtype: comma-separated list of positive subtypes, or 'normal'
    """
    rows: list[dict[str, Any]] = []
    for patient_idx in range(len(test_dataset)):
        patient_id = test_dataset.get_patient_id_by_index(patient_idx)
        patient_df = test_dataset.df[test_dataset.df["PatientID"] == patient_id].sort_values("order")
        sop_ids = patient_df["SOPInstanceUID"].values
        n_slices = len(sop_ids)

        for t in range(n_slices - 2):
            target_sop = sop_ids[t + 1]
            target_row = patient_df[patient_df["SOPInstanceUID"] == target_sop].iloc[0]
            target_any = int(target_row["any"])
            subtypes = [s for s in IH_SUBTYPES if target_row.get(s, 0) == 1]
            rows.append({
                "patient_id": patient_id,
                "triplet_index": t,
                "target_any": target_any,
                "target_subtype": ",".join(subtypes) if subtypes else "normal",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B1: Cubic z-interpolation baseline
# ---------------------------------------------------------------------------
def evaluate_cubic_baseline_on_test(
    *,
    experiment_name: str,
    data_path: Path,
    device: torch.device,
    batch_size: int,
    ssim_module: SSIM,
    msssim_module: MS_SSIM,
    lpips_module: lpips.LPIPS,
) -> pd.DataFrame:
    """Evaluate cubic z-interpolation baseline on the test set.

    For each triplet (k, k+1, k+2) where k+1 is the target:
      - If slices k-1 and k+3 exist: cubic interpolation using 4 z-neighbors
      - Otherwise: linear interpolation (= mean of 2 neighbors)
    """
    test_dataset = _load_test_dataset(data_path)

    rows: list[dict[str, Any]] = []
    total_patients = len(test_dataset)
    logger.info(f"[{experiment_name}] Evaluating cubic z-interpolation on {total_patients} patients")

    with torch.no_grad():
        for patient_idx in range(total_patients):
            patient_id = test_dataset.get_patient_id_by_index(patient_idx)
            patient_df = test_dataset.df[test_dataset.df["PatientID"] == patient_id].sort_values("order")
            sop_ids = patient_df["SOPInstanceUID"].values
            n_slices = len(sop_ids)
            img_dir = test_dataset.img_dir

            # Load all patient slices into memory (512x512 grayscale, float32 [0,1])
            all_slices = []
            for sop_id in sop_ids:
                with Image.open(img_dir / f"{sop_id}.png") as img:
                    all_slices.append(np.array(img, dtype=np.float32) / 255.0)
            all_slices = np.stack(all_slices)  # (N, H, W)

            # Also need the original inputs and targets for metric computation
            inputs, targets = test_dataset[patient_idx]
            n_triplets = inputs.shape[0]

            # Compute cubic predictions for all triplets
            predictions = []
            for t in range(n_triplets):
                # Triplet uses slice indices t, t+1, t+2
                # Try to use 4 neighbors: t-1, t, t+2, t+3
                if t >= 1 and (t + 3) < n_slices:
                    # 4-point cubic interpolation
                    z_known = np.array([t - 1, t, t + 2, t + 3], dtype=np.float64)
                    slices_known = all_slices[[t - 1, t, t + 2, t + 3]]  # (4, H, W)
                    flat = slices_known.reshape(4, -1)  # (4, H*W)
                    f = interp1d(z_known, flat, axis=0, kind="cubic")
                    pred_flat = f(float(t + 1))  # (H*W,)
                    pred_img = pred_flat.reshape(all_slices.shape[1], all_slices.shape[2])
                    pred_img = np.clip(pred_img, 0.0, 1.0)
                else:
                    # Fallback: linear (= mean of neighbors)
                    pred_img = (all_slices[t] + all_slices[t + 2]) / 2.0

                predictions.append(pred_img)

            predictions = np.stack(predictions).astype(np.float32)  # (n_triplets, H, W)
            pred_tensor = torch.from_numpy(predictions).unsqueeze(1).to(device)  # (n, 1, H, W)

            # Compute metrics in batches
            for start in range(0, n_triplets, batch_size):
                end = min(start + batch_size, n_triplets)
                batch_pred = pred_tensor[start:end]
                batch_targets = targets[start:end].to(device)

                metric_vectors = compute_metrics_per_sample(
                    pred=batch_pred,
                    target=batch_targets,
                    ssim_module=ssim_module,
                    msssim_module=msssim_module,
                    lpips_module=lpips_module,
                )

                for local_idx in range(end - start):
                    row = {
                        "experiment": experiment_name,
                        "patient_id": patient_id,
                        "triplet_index": start + local_idx,
                    }
                    for metric_name in METRIC_COLUMNS:
                        row[metric_name] = float(metric_vectors[metric_name][local_idx])
                    rows.append(row)

            if (patient_idx + 1) % 5 == 0 or (patient_idx + 1) == total_patients:
                logger.info(f"[{experiment_name}] Processed {patient_idx + 1}/{total_patients} patients")

    df = pd.DataFrame(rows)
    logger.info(f"[{experiment_name}] Produced {len(df)} per-slice metric rows")
    return df


# ---------------------------------------------------------------------------
# Evaluate mean baseline, retaining ancillary LPIPS in raw metrics.
# ---------------------------------------------------------------------------
def evaluate_mean_baseline_on_test(
    *,
    experiment_name: str,
    data_path: Path,
    device: torch.device,
    batch_size: int,
    ssim_module: SSIM,
    msssim_module: MS_SSIM,
    lpips_module: lpips.LPIPS,
) -> pd.DataFrame:
    """Evaluate mean-of-neighbors baseline on test set, retaining ancillary LPIPS."""
    test_dataset = _load_test_dataset(data_path)

    rows: list[dict[str, Any]] = []
    total_patients = len(test_dataset)
    logger.info(f"[{experiment_name}] Evaluating mean baseline on {total_patients} patients")

    with torch.no_grad():
        for patient_idx in range(total_patients):
            patient_id = test_dataset.get_patient_id_by_index(patient_idx)
            inputs, targets = test_dataset[patient_idx]
            n_triplets = inputs.shape[0]
            for start in range(0, n_triplets, batch_size):
                end = min(start + batch_size, n_triplets)
                batch_inputs = inputs[start:end].to(device)
                batch_targets = targets[start:end].to(device)
                batch_pred = batch_inputs[:, 0:1, :, :] * 0.5 + batch_inputs[:, 1:2, :, :] * 0.5

                metric_vectors = compute_metrics_per_sample(
                    pred=batch_pred,
                    target=batch_targets,
                    ssim_module=ssim_module,
                    msssim_module=msssim_module,
                    lpips_module=lpips_module,
                )

                for local_idx in range(end - start):
                    row = {
                        "experiment": experiment_name,
                        "patient_id": patient_id,
                        "triplet_index": start + local_idx,
                    }
                    for metric_name in METRIC_COLUMNS:
                        row[metric_name] = float(metric_vectors[metric_name][local_idx])
                    rows.append(row)

            if (patient_idx + 1) % 5 == 0 or (patient_idx + 1) == total_patients:
                logger.info(f"[{experiment_name}] Processed {patient_idx + 1}/{total_patients} patients")

    df = pd.DataFrame(rows)
    logger.info(f"[{experiment_name}] Produced {len(df)} per-slice metric rows")
    return df


# ---------------------------------------------------------------------------
# Evaluate neural network experiment, retaining ancillary LPIPS in raw metrics.
# ---------------------------------------------------------------------------
def evaluate_experiment_on_test(
    *,
    experiment_name: str,
    experiments_dir: Path,
    weights_path: Path,
    data_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    ssim_module: SSIM,
    msssim_module: MS_SSIM,
    lpips_module: lpips.LPIPS,
) -> pd.DataFrame:
    """Evaluate one experiment checkpoint on the fixed test split, retaining ancillary LPIPS.

    Uses 13-patch reconstruction (9 center crops + 4 corner patches, each
    256×256) to match the model's training distribution; the resulting
    composite is then compared against the 512×512 target.
    """
    test_dataset = _load_test_dataset(data_path)

    model = model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    total_patients = len(test_dataset)
    logger.info(f"[{experiment_name}] Evaluating {total_patients} patients with {weights_path}")

    with torch.no_grad():
        for patient_idx in range(total_patients):
            patient_id = test_dataset.get_patient_id_by_index(patient_idx)
            inputs, targets = test_dataset[patient_idx]
            n_triplets = inputs.shape[0]
            for start in range(0, n_triplets, batch_size):
                end = min(start + batch_size, n_triplets)
                batch_inputs = inputs[start:end].to(device)
                batch_targets = targets[start:end].to(device)
                batch_pred = predict_via_patch_reconstruction(
                    model=model,
                    batch_inputs=batch_inputs,
                    device=device,
                )

                metric_vectors = compute_metrics_per_sample(
                    pred=batch_pred,
                    target=batch_targets,
                    ssim_module=ssim_module,
                    msssim_module=msssim_module,
                    lpips_module=lpips_module,
                )

                for local_idx in range(end - start):
                    row = {
                        "experiment": experiment_name,
                        "patient_id": patient_id,
                        "triplet_index": start + local_idx,
                    }
                    for metric_name in METRIC_COLUMNS:
                        row[metric_name] = float(metric_vectors[metric_name][local_idx])
                    rows.append(row)

            if (patient_idx + 1) % 5 == 0 or (patient_idx + 1) == total_patients:
                logger.info(f"[{experiment_name}] Processed {patient_idx + 1}/{total_patients} patients")

    df = pd.DataFrame(rows)
    logger.info(f"[{experiment_name}] Produced {len(df)} per-slice metric rows")
    return df


# ---------------------------------------------------------------------------
# Bootstrap and statistical helpers
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    n = len(values)
    boot_means = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        boot_means[i] = values[rng.integers(0, n, size=n)].mean()
    low = float(np.percentile(boot_means, 100 * alpha / 2))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (low, high)


def summarize_with_bootstrap(
    *,
    per_slice_df: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Create summary table with mean and bootstrap CI per metric and experiment."""
    rng = np.random.default_rng(bootstrap_seed)
    summary_rows: list[dict[str, Any]] = []
    for experiment_name, group in per_slice_df.groupby("experiment"):
        row: dict[str, Any] = {"experiment": experiment_name, "n_samples": len(group)}
        for metric_name in METRIC_COLUMNS:
            values = group[metric_name].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(values, bootstrap_samples=bootstrap_samples, rng=rng)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_ci_low"] = ci_low
            row[f"{metric_name}_ci_high"] = ci_high
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(by="experiment")


def aggregate_patient_means(per_slice_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-slice metrics to per-patient means per experiment."""
    return (
        per_slice_df.groupby(["experiment", "patient_id"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(by=["experiment", "patient_id"])
    )


def paired_stats_vs_reference(
    *,
    metrics_df: pd.DataFrame,
    reference_experiment: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    unit_index_cols: list[str],
) -> pd.DataFrame:
    """Compute paired improvement stats and Wilcoxon tests versus a reference."""
    if reference_experiment not in metrics_df["experiment"].unique():
        raise ValueError(f"Reference experiment not found: {reference_experiment}")

    rng = np.random.default_rng(bootstrap_seed)
    ref = metrics_df[metrics_df["experiment"] == reference_experiment].set_index(unit_index_cols).sort_index()

    rows: list[dict[str, Any]] = []
    candidates = sorted(set(metrics_df["experiment"].unique()) - {reference_experiment})
    for experiment_name in candidates:
        current = metrics_df[metrics_df["experiment"] == experiment_name].set_index(unit_index_cols).sort_index()
        merged = current.join(ref[METRIC_COLUMNS], how="inner", lsuffix="_candidate", rsuffix="_reference")
        for metric_name in METRIC_COLUMNS:
            candidate_vals = merged[f"{metric_name}_candidate"].to_numpy(dtype=np.float64)
            reference_vals = merged[f"{metric_name}_reference"].to_numpy(dtype=np.float64)
            raw_diff = candidate_vals - reference_vals
            improvement = raw_diff if metric_name in HIGHER_IS_BETTER else -raw_diff

            if np.allclose(improvement, 0.0):
                wilcoxon_stat, pvalue = 0.0, 1.0
            else:
                result = wilcoxon(improvement, zero_method="wilcox", alternative="two-sided")
                wilcoxon_stat, pvalue = float(result.statistic), float(result.pvalue)

            ci_low, ci_high = bootstrap_mean_ci(improvement, bootstrap_samples=bootstrap_samples, rng=rng)
            rows.append({
                "candidate_experiment": experiment_name,
                "reference_experiment": reference_experiment,
                "metric": metric_name,
                "n_pairs": int(improvement.size),
                "analysis_unit": "+".join(unit_index_cols),
                "mean_improvement": float(np.mean(improvement)),
                "improvement_ci_low": ci_low,
                "improvement_ci_high": ci_high,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_pvalue": pvalue,
                "significant_p_lt_0_05": bool(pvalue < 0.05),
            })
    result = pd.DataFrame(rows).sort_values(by=["candidate_experiment", "metric"])
    result["wilcoxon_qvalue_bh"] = false_discovery_control(
        result["wilcoxon_pvalue"].to_numpy(dtype=np.float64),
        method="bh",
    )
    result["significant_q_lt_0_05"] = result["wilcoxon_qvalue_bh"] < 0.05
    return result


# ---------------------------------------------------------------------------
# B4: Hemorrhage stratification
# ---------------------------------------------------------------------------
def stratify_by_hemorrhage(
    *,
    per_slice_df: pd.DataFrame,
    hemorrhage_labels: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Compute per-experiment metrics stratified by hemorrhage presence.

    Returns a DataFrame with columns: experiment, group, n_samples,
    {metric}_mean, {metric}_ci_low, {metric}_ci_high for each metric.
    """
    merged = per_slice_df.merge(
        hemorrhage_labels[["patient_id", "triplet_index", "target_any"]],
        on=["patient_id", "triplet_index"],
        how="left",
    )
    merged["group"] = merged["target_any"].map({0: "normal", 1: "hemorrhage"})

    rng = np.random.default_rng(bootstrap_seed)
    summary_rows: list[dict[str, Any]] = []
    for (experiment_name, group_name), group_df in merged.groupby(["experiment", "group"]):
        row: dict[str, Any] = {
            "experiment": experiment_name,
            "group": group_name,
            "n_samples": len(group_df),
        }
        for metric_name in METRIC_COLUMNS:
            values = group_df[metric_name].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(values, bootstrap_samples=bootstrap_samples, rng=rng)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_ci_low"] = ci_low
            row[f"{metric_name}_ci_high"] = ci_high
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(by=["experiment", "group"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run revision table generation."""
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    if not args.test_experiment:
        logger.error("No --test-experiment specified. Nothing to do.")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")

    # Initialize shared metric modules
    ssim_module = SSIM(data_range=1.0, size_average=False, channel=1).to(device)
    msssim_module = MS_SSIM(data_range=1.0, size_average=False, channel=1).to(device)
    lpips_module = lpips.LPIPS(net="alex").to(device)
    lpips_module.eval()

    # We need a data_path from any experiment to load the test dataset
    data_path: Path | None = None

    per_slice_frames: list[pd.DataFrame] = []
    for experiment_name in args.test_experiment:
        if experiment_name == "baseline_mean":
            if data_path is None:
                data_path = _resolve_data_path(args)
            per_slice_df = evaluate_mean_baseline_on_test(
                experiment_name=experiment_name,
                data_path=data_path,
                device=device,
                batch_size=args.batch_size,
                ssim_module=ssim_module,
                msssim_module=msssim_module,
                lpips_module=lpips_module,
            )
        elif experiment_name == "baseline_cubic":
            if data_path is None:
                data_path = _resolve_data_path(args)
            per_slice_df = evaluate_cubic_baseline_on_test(
                experiment_name=experiment_name,
                data_path=data_path,
                device=device,
                batch_size=args.batch_size,
                ssim_module=ssim_module,
                msssim_module=msssim_module,
                lpips_module=lpips_module,
            )
        else:
            experiment_dir = args.experiments_dir / experiment_name
            weights_path = _resolve_best_weights(experiment_dir)
            if weights_path is None:
                logger.error(f"No weights found for {experiment_name}, skipping.")
                continue
            model, _config, exp_data_path = load_model_from_checkpoint(experiment_dir, weights_path)
            if data_path is None:
                data_path = exp_data_path
            per_slice_df = evaluate_experiment_on_test(
                experiment_name=experiment_name,
                experiments_dir=args.experiments_dir,
                weights_path=weights_path,
                data_path=exp_data_path,
                model=model,
                device=device,
                batch_size=args.batch_size,
                ssim_module=ssim_module,
                msssim_module=msssim_module,
                lpips_module=lpips_module,
            )
        per_slice_frames.append(per_slice_df)

    all_per_slice = pd.concat(per_slice_frames, ignore_index=True)

    # --- Standard summaries over the reported metric family. ---
    summary = summarize_with_bootstrap(
        per_slice_df=all_per_slice,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    patient_level = aggregate_patient_means(all_per_slice)
    patient_summary = summarize_with_bootstrap(
        per_slice_df=patient_level,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    # Paired tests (slice-level + patient-level)
    paired_slice = paired_stats_vs_reference(
        metrics_df=all_per_slice,
        reference_experiment=args.reference_experiment,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        unit_index_cols=["patient_id", "triplet_index"],
    )
    paired_patient = paired_stats_vs_reference(
        metrics_df=patient_level,
        reference_experiment=args.reference_experiment,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        unit_index_cols=["patient_id"],
    )
    paired = pd.concat([paired_slice, paired_patient], ignore_index=True)

    # --- B4: Hemorrhage stratification ---
    if data_path is None:
        raise RuntimeError("No data_path resolved from any experiment. Cannot build hemorrhage labels.")
    test_dataset = _load_test_dataset(data_path)
    hemorrhage_labels = _build_triplet_hemorrhage_labels(test_dataset)
    stratification = stratify_by_hemorrhage(
        per_slice_df=all_per_slice,
        hemorrhage_labels=hemorrhage_labels,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    # --- Write outputs ---
    tag = args.snapshot_date
    outputs = {
        f"revision_slice_metrics_{tag}.csv": all_per_slice,
        f"revision_metrics_summary_{tag}.csv": summary,
        f"revision_patient_metrics_{tag}.csv": patient_level,
        f"revision_patient_summary_{tag}.csv": patient_summary,
        f"revision_paired_stats_{tag}.csv": paired,
        f"revision_hemorrhage_stratification_{tag}.csv": stratification,
        f"revision_hemorrhage_labels_{tag}.csv": hemorrhage_labels,
    }
    for filename, df in outputs.items():
        out_path = args.tables_dir / filename
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {out_path} ({len(df)} rows)")

    logger.info("Done. All revision tables written.")


def _resolve_data_path(args: argparse.Namespace) -> Path:
    """Resolve data_path from the reference experiment config."""
    ref_dir = args.experiments_dir / args.reference_experiment
    config_path = ref_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Reference experiment config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return resolve_config_path(config["data_path"])


if __name__ == "__main__":
    main()
