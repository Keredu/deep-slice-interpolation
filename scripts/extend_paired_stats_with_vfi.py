"""Extend the revision paired-stats CSV with RIFE and FILM.

The original `build_revision_tables.py` pipeline did not include the
VFI baselines (RIFE, FILM), because those baselines are evaluated by a
separate script (`evaluate_vfi_baselines.py`). Their per-slice metric
dumps live in `results/tables/{rife,film}_baseline_slice_metrics.csv`
with the same column schema as `revision_slice_metrics_<date>.csv`.

This script:
  1. Loads `revision_slice_metrics_<date>.csv` (the 9 experiments evaluated by
     `build_revision_tables.py`: 2 classical baselines + 7 learned models).
  2. Appends the RIFE and FILM per-slice rows.
  3. Aggregates to per-patient means.
  4. Recomputes paired Wilcoxon improvements (patient-level and slice-level)
     versus the reference model, applies Benjamini-Hochberg FDR over the full
     family, and writes a new `revision_paired_stats_<date>.csv`.

Invariants preserved w.r.t. `build_revision_tables.py.paired_stats_vs_reference`:
  - METRIC_COLUMNS order
  - HIGHER_IS_BETTER / LOWER_IS_BETTER conventions
  - bootstrap seed (42), bootstrap samples (2000)
  - wilcoxon(alternative="two-sided", zero_method="wilcox")

Usage:
    uv run scripts/extend_paired_stats_with_vfi.py \
        --snapshot-date 2026-04-16 \
        --reference-experiment msssim+l1_lr8e-4_bc1d65
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import false_discovery_control, wilcoxon

HIGHER_IS_BETTER = {"ssim", "ms_ssim", "psnr"}
LOWER_IS_BETTER = {"mae", "gradient_mae"}
METRIC_COLUMNS = ["ssim", "ms_ssim", "mae", "gradient_mae", "psnr"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p.add_argument("--snapshot-date", type=str, default="2026-04-16")
    p.add_argument("--reference-experiment", type=str, default="msssim+l1_lr8e-4_bc1d65")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    return p.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for the mean (same implementation as build_revision_tables)."""
    n = len(values)
    boot_means = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        boot_means[i] = values[rng.integers(0, n, size=n)].mean()
    low = float(np.percentile(boot_means, 100 * alpha / 2))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (low, high)


def paired_stats_vs_reference(
    *,
    metrics_df: pd.DataFrame,
    reference_experiment: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    unit_index_cols: list[str],
) -> pd.DataFrame:
    """Compute paired improvements + Wilcoxon tests vs reference.

    Mirrors `build_revision_tables.paired_stats_vs_reference` exactly so that
    existing rows reproduce bit-for-bit up to RNG state ordering.
    """
    if reference_experiment not in metrics_df["experiment"].unique():
        raise ValueError(f"Reference experiment not found: {reference_experiment}")

    rng = np.random.default_rng(bootstrap_seed)
    ref = metrics_df[metrics_df["experiment"] == reference_experiment].set_index(unit_index_cols).sort_index()

    rows: list[dict] = []
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


def main() -> None:
    """Extend the paired-stats CSV with RIFE and FILM rows."""
    args = parse_args()

    tag = args.snapshot_date
    slice_csv = args.tables_dir / f"revision_slice_metrics_{tag}.csv"
    rife_csv = args.tables_dir / "rife_baseline_slice_metrics.csv"
    film_csv = args.tables_dir / "film_baseline_slice_metrics.csv"
    out_csv = args.tables_dir / f"revision_paired_stats_{tag}.csv"

    logger.info(f"Loading per-slice revision metrics: {slice_csv}")
    slice_df = pd.read_csv(slice_csv)
    logger.info(f"  rows={len(slice_df)}, experiments={sorted(slice_df['experiment'].unique())}")

    logger.info(f"Loading RIFE per-slice metrics: {rife_csv}")
    rife_df = pd.read_csv(rife_csv)
    logger.info(f"  rows={len(rife_df)}, experiments={sorted(rife_df['experiment'].unique())}")

    logger.info(f"Loading FILM per-slice metrics: {film_csv}")
    film_df = pd.read_csv(film_csv)
    logger.info(f"  rows={len(film_df)}, experiments={sorted(film_df['experiment'].unique())}")

    # Confirm schema alignment
    expected_cols = ["experiment", "patient_id", "triplet_index", *METRIC_COLUMNS]
    for name, df in [("slice", slice_df), ("rife", rife_df), ("film", film_df)]:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    # Combine all per-slice rows
    all_slice = pd.concat([slice_df[expected_cols], rife_df[expected_cols], film_df[expected_cols]], ignore_index=True)
    logger.info(f"Combined per-slice rows: {len(all_slice)} from {all_slice['experiment'].nunique()} experiments")

    # Patient-level aggregation (same as build_revision_tables.aggregate_patient_means)
    patient_level = (
        all_slice.groupby(["experiment", "patient_id"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(by=["experiment", "patient_id"])
    )

    # Recompute paired stats
    logger.info("Computing slice-level paired stats (n=968)")
    paired_slice = paired_stats_vs_reference(
        metrics_df=all_slice,
        reference_experiment=args.reference_experiment,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        unit_index_cols=["patient_id", "triplet_index"],
    )
    logger.info(f"  slice-level rows: {len(paired_slice)}")

    logger.info("Computing patient-level paired stats (n=30)")
    paired_patient = paired_stats_vs_reference(
        metrics_df=patient_level,
        reference_experiment=args.reference_experiment,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        unit_index_cols=["patient_id"],
    )
    logger.info(f"  patient-level rows: {len(paired_patient)}")

    paired = pd.concat([paired_slice, paired_patient], ignore_index=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv} ({len(paired)} rows)")

    # Print RIFE/FILM patient-level rows for paper table update
    vfi_patient = paired_patient[
        paired_patient["candidate_experiment"].isin(["baseline_rife", "baseline_film"])
    ]
    logger.info("RIFE/FILM patient-level paired rows (for Table 4 update):")
    for _, row in vfi_patient.iterrows():
        logger.info(
            f"  {row['candidate_experiment']:14s} {row['metric']:12s} "
            f"Delta={row['mean_improvement']:+.4g} "
            f"[{row['improvement_ci_low']:+.4g}, {row['improvement_ci_high']:+.4g}] "
            f"p={row['wilcoxon_pvalue']:.4g} q={row['wilcoxon_qvalue_bh']:.4g}"
        )


if __name__ == "__main__":
    main()
