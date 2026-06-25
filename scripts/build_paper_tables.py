"""Build reproducible experiment tables and statistical summaries for the paper.

This script produces:
1. Snapshot tables from `experiments_registry.json` + per-experiment `epochs.csv`
2. Optional test-set evaluation for selected experiments with:
   - per-slice metrics
   - bootstrap confidence intervals
   - paired Wilcoxon tests vs a reference model

Outputs are written to `results/tables/` by default.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from loguru import logger
from pytorch_msssim import MS_SSIM, SSIM
from scipy.stats import false_discovery_control, wilcoxon

from phd.config_io import resolve_config_path
from phd.datasets.interpolation.two_to_one_slice import STANDARD_TRANSFORM, TwoToOneSliceTestDataset
from phd.models.setup_model import setup_model
from phd.viz import predict_via_patch_reconstruction

HIGHER_IS_BETTER = {"ssim", "ms_ssim", "psnr", "ncc"}
LOWER_IS_BETTER = {"mae", "gradient_mae"}
METRIC_COLUMNS = ["ssim", "ms_ssim", "mae", "gradient_mae", "psnr", "ncc"]


@dataclass(frozen=True)
class BestCheckpointInfo:
    """Best-checkpoint metadata from `epochs.csv`."""

    best_epoch_1based: int | None
    best_valid_loss: float | None
    metrics: dict[str, float]
    total_epochs_recorded: int
    non_finite_valid_epochs: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build reproducible paper tables from experiments.")
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("experiments/experiments_registry.json"),
        help="Path to experiments registry JSON.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("experiments/EXPERIMENT_LOG.md"),
        help="Path to experiment log markdown.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments/train_nn1_cropped"),
        help="Directory containing experiment folders.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory where CSV outputs are written.",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default=date.today().isoformat(),
        help="Date tag used in output filenames (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--test-experiment",
        action="append",
        default=[],
        help="Experiment name to evaluate on test split. Can be passed multiple times.",
    )
    parser.add_argument(
        "--reference-experiment",
        type=str,
        default="msssim+l1_lr8e-4_bc1d65",
        help="Reference experiment for paired statistical tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for test inference.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device for test evaluation (`cpu` or `cuda`).",
    )
    return parser.parse_args()


def parse_flags_from_log(log_path: Path) -> dict[str, dict[str, str]]:
    """Parse success/paper flags from the comprehensive review table in EXPERIMENT_LOG."""
    if not log_path.exists():
        logger.warning(f"Experiment log not found: {log_path}")
        return {}

    content = log_path.read_text(encoding="utf-8")
    header = "## [2026-02-08] Comprehensive Review"
    if header not in content:
        logger.warning("Could not find comprehensive review section in experiment log.")
        return {}

    section = content[content.index(header) :]
    flags: dict[str, dict[str, str]] = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) < 6:
            continue
        exp_name = cols[0].strip("`")
        flags[exp_name] = {
            "registry_status_log": cols[1],
            "success_flag": cols[3],
            "paper_flag": cols[4],
            "review_notes": cols[5],
        }
    logger.info(f"Parsed flags for {len(flags)} experiments from {log_path}")
    return flags


def load_registry(registry_path: Path) -> dict[str, Any]:
    """Load registry JSON."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def get_best_checkpoint_info(epochs_csv_path: Path) -> BestCheckpointInfo:
    """Read `epochs.csv` and return best-checkpoint metadata."""
    if not epochs_csv_path.exists():
        return BestCheckpointInfo(
            best_epoch_1based=None,
            best_valid_loss=None,
            metrics={},
            total_epochs_recorded=0,
            non_finite_valid_epochs=0,
        )

    df = pd.read_csv(epochs_csv_path)
    if df.empty:
        return BestCheckpointInfo(
            best_epoch_1based=None,
            best_valid_loss=None,
            metrics={},
            total_epochs_recorded=0,
            non_finite_valid_epochs=0,
        )

    valid_loss_numeric = pd.to_numeric(df["valid_loss"], errors="coerce")
    finite_mask = valid_loss_numeric.notna() & np.isfinite(valid_loss_numeric)
    non_finite_valid_epochs = int((~finite_mask).sum())
    finite_df = df.loc[finite_mask].copy()
    if finite_df.empty:
        return BestCheckpointInfo(
            best_epoch_1based=None,
            best_valid_loss=None,
            metrics={},
            total_epochs_recorded=len(df),
            non_finite_valid_epochs=non_finite_valid_epochs,
        )

    best_idx = pd.to_numeric(finite_df["valid_loss"], errors="coerce").idxmin()
    best_row = finite_df.loc[best_idx]
    metrics: dict[str, float] = {}
    for metric_name in METRIC_COLUMNS:
        if metric_name in best_row and pd.notna(best_row[metric_name]):
            metrics[metric_name] = float(best_row[metric_name])

    return BestCheckpointInfo(
        best_epoch_1based=int(best_row["epoch"]),
        best_valid_loss=float(best_row["valid_loss"]),
        metrics=metrics,
        total_epochs_recorded=len(df),
        non_finite_valid_epochs=non_finite_valid_epochs,
    )


def infer_datasets_dir_from_data_path(data_path: Path) -> Path:
    """Infer DATASETS_DIR from a configured `data_path`."""
    parts = list(data_path.resolve().parts)
    try:
        pre_idx = parts.index("pre")
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot infer DATASETS_DIR from data_path={data_path}. Expected '/.../pre/...'.",
        ) from exc
    return Path(*parts[:pre_idx])


def resolve_best_weights_path(experiment_dir: Path, best_epoch_1based: int | None) -> Path | None:
    """Resolve the best-checkpoint weights path from best epoch metadata."""
    if best_epoch_1based is None:
        return None

    direct = experiment_dir / "epochs" / f"{best_epoch_1based - 1}" / "weights.pth"
    if direct.exists():
        return direct

    # Fallback: choose latest available checkpoint if direct mapping is missing.
    epoch_root = experiment_dir / "epochs"
    if not epoch_root.exists():
        return None
    candidates = sorted(
        [p for p in epoch_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    for candidate in reversed(candidates):
        candidate_weights = candidate / "weights.pth"
        if candidate_weights.exists():
            return candidate_weights
    return None


def make_snapshot_dataframe(
    *,
    registry: dict[str, Any],
    experiments_dir: Path,
    log_flags: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Build a normalized snapshot dataframe from registry + epochs + log flags."""
    rows: list[dict[str, Any]] = []
    for exp_name in sorted(registry.keys()):
        reg_entry = registry[exp_name]
        config = reg_entry.get("config", {})
        loss_cfg = config.get("loss", {})
        optimizer_cfg = config.get("optimizer", {})
        experiment_dir = experiments_dir / exp_name
        epochs_csv_path = experiment_dir / "epochs.csv"
        checkpoint_info = get_best_checkpoint_info(epochs_csv_path)
        weights_path = resolve_best_weights_path(experiment_dir, checkpoint_info.best_epoch_1based)

        metrics_payload = {f"{metric_name}": checkpoint_info.metrics.get(metric_name) for metric_name in METRIC_COLUMNS}
        log_payload = log_flags.get(exp_name, {})

        status = reg_entry.get("status")
        has_partial_progress = checkpoint_info.total_epochs_recorded > 0
        is_pending = status in {"NOT_STARTED", "RUNNING"} and not has_partial_progress
        is_partial = status in {"NOT_STARTED", "NAN_VALUE_DETECTED", "ERROR", "RUNNING"} and has_partial_progress
        is_success = status in {"EARLY_STOPPING", "FINISHED_EPOCHS"}

        auto_success_flag = "FAILED"
        if is_pending:
            auto_success_flag = "PENDING"
        elif is_success:
            auto_success_flag = "SUCCESS"
        elif is_partial:
            auto_success_flag = "PARTIAL"

        promising_partial = bool(
            is_partial
            and checkpoint_info.best_epoch_1based is not None
            and checkpoint_info.best_epoch_1based >= 5
            and (checkpoint_info.metrics.get("ssim") or 0.0) >= 0.70
            and (checkpoint_info.metrics.get("mae") or 1.0) <= 0.045
        )

        rows.append(
            {
                "experiment": exp_name,
                "registry_status": status,
                "queued_at": reg_entry.get("queued_at"),
                "last_started": reg_entry.get("last_started"),
                "finished": reg_entry.get("finished"),
                "runs": reg_entry.get("runs"),
                "loss_name": loss_cfg.get("name"),
                "loss_params_json": json.dumps(loss_cfg.get("params", {}), sort_keys=True),
                "lr": optimizer_cfg.get("params", {}).get("lr"),
                "batch_size": config.get("batch_size"),
                "num_workers": config.get("num_workers"),
                "early_stopping_patience": config.get("early_stopping_patience"),
                "best_epoch": checkpoint_info.best_epoch_1based,
                "best_valid_loss": checkpoint_info.best_valid_loss,
                "total_epochs_recorded": checkpoint_info.total_epochs_recorded,
                "non_finite_valid_epochs": checkpoint_info.non_finite_valid_epochs,
                "weights_path": str(weights_path) if weights_path is not None else None,
                "has_weights": bool(weights_path is not None),
                "auto_success_flag": auto_success_flag,
                "promising_partial": promising_partial,
                "success_flag": log_payload.get("success_flag", auto_success_flag),
                "paper_flag": log_payload.get("paper_flag", "UNFLAGGED"),
                "review_notes": log_payload.get("review_notes", ""),
                **metrics_payload,
            },
        )

    df = pd.DataFrame(rows)
    return df


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the sample mean."""
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        v = float(values[0])
        return (v, v)

    indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
    boot_means = values[indices].mean(axis=1)
    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return (low, high)


def load_model_from_checkpoint(experiment_dir: Path, weights_path: Path) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    """Load model and config for inference."""
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config for experiment: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

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

    data_path = resolve_config_path(config["data_path"])
    return model, config, data_path


def _sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude for a batch of grayscale images."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
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
) -> dict[str, np.ndarray]:
    """Compute metric vectors (one value per sample)."""
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

    return {
        "ssim": ssim_vals.detach().cpu().numpy(),
        "ms_ssim": msssim_vals.detach().cpu().numpy(),
        "mae": mae_vals.detach().cpu().numpy(),
        "gradient_mae": gradient_mae_vals.detach().cpu().numpy(),
        "psnr": psnr_vals.detach().cpu().numpy(),
        "ncc": ncc_vals.detach().cpu().numpy(),
    }


def evaluate_experiment_on_test(
    *,
    experiment_name: str,
    experiments_dir: Path,
    weights_path: Path,
    data_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """Evaluate one experiment checkpoint on the fixed test split."""
    datasets_dir = os.getenv("DATASETS_DIR")
    if not datasets_dir:
        inferred = infer_datasets_dir_from_data_path(data_path)
        os.environ["DATASETS_DIR"] = str(inferred)
        logger.info(f"DATASETS_DIR was not set. Inferred DATASETS_DIR={inferred}")

    test_dataset = TwoToOneSliceTestDataset(
        root_dir=str(data_path),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    model = model.to(device)
    model.eval()
    ssim_module = SSIM(data_range=1.0, size_average=False, channel=1).to(device)
    msssim_module = MS_SSIM(data_range=1.0, size_average=False, channel=1).to(device)

    rows: list[dict[str, Any]] = []
    total_patients = len(test_dataset)
    logger.info(f"[{experiment_name}] Evaluating {total_patients} test patients with checkpoint {weights_path}")
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
                )

                batch_size_actual = end - start
                for local_idx in range(batch_size_actual):
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


def evaluate_mean_baseline_on_test(
    *,
    experiment_name: str,
    data_path: Path,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """Evaluate the classical mean-of-neighbors baseline on the fixed test split."""
    datasets_dir = os.getenv("DATASETS_DIR")
    if not datasets_dir:
        inferred = infer_datasets_dir_from_data_path(data_path)
        os.environ["DATASETS_DIR"] = str(inferred)
        logger.info(f"DATASETS_DIR was not set. Inferred DATASETS_DIR={inferred}")

    test_dataset = TwoToOneSliceTestDataset(
        root_dir=str(data_path),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    ssim_module = SSIM(data_range=1.0, size_average=False, channel=1).to(device)
    msssim_module = MS_SSIM(data_range=1.0, size_average=False, channel=1).to(device)

    rows: list[dict[str, Any]] = []
    total_patients = len(test_dataset)
    logger.info(f"[{experiment_name}] Evaluating classical baseline on {total_patients} test patients")
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
                )

                batch_size_actual = end - start
                for local_idx in range(batch_size_actual):
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
        row: dict[str, Any] = {"experiment": experiment_name, "n_samples": int(len(group))}
        for metric_name in METRIC_COLUMNS:
            values = group[metric_name].to_numpy(dtype=np.float64)
            mean_value = float(values.mean())
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
            row[f"{metric_name}_mean"] = mean_value
            row[f"{metric_name}_ci_low"] = ci_low
            row[f"{metric_name}_ci_high"] = ci_high
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(by="experiment")


def aggregate_patient_means(per_slice_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-slice metrics to per-patient means per experiment."""
    grouped = (
        per_slice_df.groupby(["experiment", "patient_id"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(by=["experiment", "patient_id"])
    )
    return grouped


def paired_stats_vs_reference(
    *,
    metrics_df: pd.DataFrame,
    reference_experiment: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    unit_index_cols: list[str],
) -> pd.DataFrame:
    """Compute paired improvement stats and Wilcoxon tests versus a reference experiment."""
    if reference_experiment not in metrics_df["experiment"].unique():
        raise ValueError(f"Reference experiment not found in test metrics: {reference_experiment}")

    rng = np.random.default_rng(bootstrap_seed)
    ref = (
        metrics_df[metrics_df["experiment"] == reference_experiment]
        .set_index(unit_index_cols)
        .sort_index()
    )

    rows: list[dict[str, Any]] = []
    candidates = sorted(set(metrics_df["experiment"].unique()) - {reference_experiment})
    for experiment_name in candidates:
        current = (
            metrics_df[metrics_df["experiment"] == experiment_name]
            .set_index(unit_index_cols)
            .sort_index()
        )
        merged = current.join(
            ref[METRIC_COLUMNS],
            how="inner",
            lsuffix="_candidate",
            rsuffix="_reference",
        )
        for metric_name in METRIC_COLUMNS:
            candidate_vals = merged[f"{metric_name}_candidate"].to_numpy(dtype=np.float64)
            reference_vals = merged[f"{metric_name}_reference"].to_numpy(dtype=np.float64)
            raw_diff = candidate_vals - reference_vals
            improvement = raw_diff if metric_name in HIGHER_IS_BETTER else -raw_diff

            if np.allclose(improvement, 0.0):
                wilcoxon_stat = 0.0
                pvalue = 1.0
            else:
                wilcoxon_result = wilcoxon(improvement, zero_method="wilcox", alternative="two-sided")
                wilcoxon_stat = float(wilcoxon_result.statistic)
                pvalue = float(wilcoxon_result.pvalue)

            ci_low, ci_high = bootstrap_mean_ci(
                improvement,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
            rows.append(
                {
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
                },
            )
    result = pd.DataFrame(rows).sort_values(by=["candidate_experiment", "metric"])
    result["wilcoxon_qvalue_bh"] = false_discovery_control(
        result["wilcoxon_pvalue"].to_numpy(dtype=np.float64),
        method="bh",
    )
    result["significant_q_lt_0_05"] = result["wilcoxon_qvalue_bh"] < 0.05
    return result


def main() -> None:
    """Run table generation and optional test evaluation."""
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(args.registry_path)
    log_flags = parse_flags_from_log(args.log_path)
    snapshot_df = make_snapshot_dataframe(
        registry=registry,
        experiments_dir=args.experiments_dir,
        log_flags=log_flags,
    )

    # Paper snapshot tables
    metrics_snapshot = snapshot_df[snapshot_df["paper_flag"].isin(["CORE", "SUPPORT"])].copy()
    metrics_snapshot = metrics_snapshot.sort_values(
        by=["paper_flag", "success_flag", "ssim", "ms_ssim", "mae"],
        ascending=[True, True, False, False, True],
        na_position="last",
    )
    ablation_snapshot = snapshot_df[snapshot_df["success_flag"].isin(["PARTIAL", "FAILED", "PENDING"])].copy()
    ablation_snapshot = ablation_snapshot.sort_values(
        by=["success_flag", "promising_partial", "ssim", "mae"],
        ascending=[True, False, False, True],
        na_position="last",
    )

    full_snapshot_out = args.tables_dir / f"experiment_snapshot_full_{args.snapshot_date}.csv"
    metrics_out = args.tables_dir / f"experiment_metrics_snapshot_{args.snapshot_date}.csv"
    ablation_out = args.tables_dir / f"experiment_ablation_snapshot_{args.snapshot_date}.csv"
    snapshot_df.to_csv(full_snapshot_out, index=False)
    metrics_snapshot.to_csv(metrics_out, index=False)
    ablation_snapshot.to_csv(ablation_out, index=False)
    logger.info(f"Wrote snapshot tables:\n- {full_snapshot_out}\n- {metrics_out}\n- {ablation_out}")

    if not args.test_experiment:
        logger.info("No --test-experiment specified; skipping test-set evaluation.")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")

    per_slice_frames: list[pd.DataFrame] = []
    for experiment_name in args.test_experiment:
        if experiment_name == "baseline_mean":
            ref_row = snapshot_df[snapshot_df["experiment"] == args.reference_experiment]
            if ref_row.empty:
                raise ValueError(
                    "Cannot evaluate baseline_mean because reference experiment "
                    f"'{args.reference_experiment}' is missing from snapshot.",
                )
            ref_weights = Path(ref_row.iloc[0]["weights_path"])
            ref_experiment_dir = args.experiments_dir / args.reference_experiment
            _model, _config, data_path = load_model_from_checkpoint(ref_experiment_dir, ref_weights)
            per_slice_df = evaluate_mean_baseline_on_test(
                experiment_name=experiment_name,
                data_path=data_path,
                device=device,
                batch_size=args.batch_size,
            )
        else:
            row_match = snapshot_df[snapshot_df["experiment"] == experiment_name]
            if row_match.empty:
                raise ValueError(f"Experiment not found in registry snapshot: {experiment_name}")
            row = row_match.iloc[0]
            weights_path_raw = row["weights_path"]
            if not weights_path_raw:
                raise ValueError(f"No weights path resolved for experiment: {experiment_name}")
            weights_path = Path(weights_path_raw)
            experiment_dir = args.experiments_dir / experiment_name

            model, _config, data_path = load_model_from_checkpoint(experiment_dir, weights_path)
            per_slice_df = evaluate_experiment_on_test(
                experiment_name=experiment_name,
                experiments_dir=args.experiments_dir,
                weights_path=weights_path,
                data_path=data_path,
                model=model,
                device=device,
                batch_size=args.batch_size,
            )
        per_slice_frames.append(per_slice_df)

    all_per_slice = pd.concat(per_slice_frames, ignore_index=True)
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

    per_slice_out = args.tables_dir / f"test_slice_metrics_{args.snapshot_date}.csv"
    summary_out = args.tables_dir / f"test_metrics_summary_{args.snapshot_date}.csv"
    patient_level_out = args.tables_dir / f"test_patient_metrics_{args.snapshot_date}.csv"
    patient_summary_out = args.tables_dir / f"test_patient_summary_{args.snapshot_date}.csv"
    paired_out = args.tables_dir / f"test_paired_stats_{args.snapshot_date}.csv"
    all_per_slice.to_csv(per_slice_out, index=False)
    summary.to_csv(summary_out, index=False)
    patient_level.to_csv(patient_level_out, index=False)
    patient_summary.to_csv(patient_summary_out, index=False)
    paired.to_csv(paired_out, index=False)
    logger.info(
        "Wrote test evaluation tables:\n"
        f"- {per_slice_out}\n"
        f"- {summary_out}\n"
        f"- {patient_level_out}\n"
        f"- {patient_summary_out}\n"
        f"- {paired_out}",
    )


if __name__ == "__main__":
    main()
