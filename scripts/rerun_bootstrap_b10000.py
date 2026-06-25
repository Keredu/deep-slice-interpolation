"""Re-derive bootstrap-dependent revision tables at B=10,000 from cached slice metrics.

The per-slice metrics in `results/tables/revision_slice_metrics_<tag>.csv`
are deterministic given the model checkpoints, so we reuse them and re-run only the
bootstrap stage (CIs, paired Wilcoxon CIs, hemorrhage stratification) at the higher
resample count. This avoids re-running GPU inference, which would not change the
slice-level numbers.

Inputs (reads from --tables-dir):
  - revision_slice_metrics_<input-tag>.csv
  - revision_hemorrhage_labels_<input-tag>.csv

Outputs (writes to --tables-dir with --output-tag suffix):
  - revision_metrics_summary_<output-tag>.csv
  - revision_patient_metrics_<output-tag>.csv
  - revision_patient_summary_<output-tag>.csv
  - revision_paired_stats_<output-tag>.csv
  - revision_hemorrhage_stratification_<output-tag>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from scripts.build_revision_tables import (
    aggregate_patient_means,
    paired_stats_vs_reference,
    stratify_by_hemorrhage,
    summarize_with_bootstrap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rerun bootstrap stage at higher B from cached slice metrics.")
    p.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    p.add_argument("--input-tag", type=str, required=True, help="Date tag of the cached slice metrics (e.g. 2026-04-16).")
    p.add_argument("--output-tag", type=str, required=True, help="Date tag for the regenerated outputs (e.g. 2026-05-03).")
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--reference-experiment", type=str, default="msssim+l1_lr8e-4_bc1d65")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    slice_path = args.tables_dir / f"revision_slice_metrics_{args.input_tag}.csv"
    labels_path = args.tables_dir / f"revision_hemorrhage_labels_{args.input_tag}.csv"
    if not slice_path.exists():
        raise FileNotFoundError(f"Cached slice metrics not found: {slice_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Cached hemorrhage labels not found: {labels_path}")

    logger.info(f"Loading cached slice metrics: {slice_path}")
    all_per_slice = pd.read_csv(slice_path)
    logger.info(f"Loading cached hemorrhage labels: {labels_path}")
    hemorrhage_labels = pd.read_csv(labels_path)

    logger.info(f"Recomputing bootstrap-derived tables at B={args.bootstrap_samples} (seed={args.bootstrap_seed})")

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

    stratification = stratify_by_hemorrhage(
        per_slice_df=all_per_slice,
        hemorrhage_labels=hemorrhage_labels,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    tag = args.output_tag
    outputs = {
        f"revision_metrics_summary_{tag}.csv": summary,
        f"revision_patient_metrics_{tag}.csv": patient_level,
        f"revision_patient_summary_{tag}.csv": patient_summary,
        f"revision_paired_stats_{tag}.csv": paired,
        f"revision_hemorrhage_stratification_{tag}.csv": stratification,
    }
    for filename, df in outputs.items():
        out_path = args.tables_dir / filename
        df.to_csv(out_path, index=False)
        logger.info(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
