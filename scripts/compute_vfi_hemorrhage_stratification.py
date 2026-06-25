"""Compute VFI baseline metrics stratified by hemorrhage presence.

Outputs rows matching the format of the existing hemorrhage stratification
table (Table 4) so they can be added directly.

Usage:
    uv run scripts/compute_vfi_hemorrhage_stratification.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

TABLES_DIR = Path("results/tables")
HEMORRHAGE_LABELS = TABLES_DIR / "revision_hemorrhage_labels_2026-03-03.csv"

VFI_SLICE_FILES = {
    "RIFE": TABLES_DIR / "rife_baseline_slice_metrics.csv",
    "FILM": TABLES_DIR / "film_baseline_slice_metrics.csv",
}

METRICS = ["ssim", "mae", "lpips"]


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.RandomState(seed)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_resamples)
    ])
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    # Load hemorrhage labels
    labels = pd.read_csv(HEMORRHAGE_LABELS)
    labels = labels.rename(columns={labels.columns[0]: "patient_id"})
    # Columns: patient_id, triplet_index, hemorrhage_flag(?), label
    label_col = [c for c in labels.columns if c not in ("patient_id", "triplet_index")
                 and labels[c].dtype == object]
    if label_col:
        labels["is_hemorrhage"] = labels[label_col[0]].str.lower() != "normal"
    else:
        # Assume a numeric flag column
        flag_col = [c for c in labels.columns if c not in ("patient_id", "triplet_index")][0]
        labels["is_hemorrhage"] = labels[flag_col].astype(bool)

    print(f"Hemorrhage labels: {labels['is_hemorrhage'].sum()} hemorrhage, "
          f"{(~labels['is_hemorrhage']).sum()} normal\n")

    rows = []
    for name, path in VFI_SLICE_FILES.items():
        df = pd.read_csv(path)
        # Merge with labels
        merged = df.merge(
            labels[["patient_id", "triplet_index", "is_hemorrhage"]],
            on=["patient_id", "triplet_index"],
            how="inner",
        )
        if len(merged) < len(df):
            print(f"WARNING: {name} lost {len(df) - len(merged)} rows in label merge")

        for group_name, is_hem in [("Hemorrhage", True), ("Normal", False)]:
            subset = merged[merged["is_hemorrhage"] == is_hem]
            row = {"method": name, "group": group_name, "n": len(subset)}
            for m in METRICS:
                mean, ci_lo, ci_hi = bootstrap_ci(subset[m].values)
                row[f"{m}_mean"] = mean
                row[f"{m}_ci_low"] = ci_lo
                row[f"{m}_ci_high"] = ci_hi
            rows.append(row)

    results = pd.DataFrame(rows)

    # Print LaTeX-ready rows
    print("=" * 80)
    print("LaTeX rows for Table 4 (hemorrhage stratification):")
    print("=" * 80)
    for name in VFI_SLICE_FILES:
        hem = results[(results["method"] == name) & (results["group"] == "Hemorrhage")].iloc[0]
        nor = results[(results["method"] == name) & (results["group"] == "Normal")].iloc[0]
        line = f"    {name} baseline"
        for m in METRICS:
            for r in [hem, nor]:
                mean = r[f"{m}_mean"]
                ci_lo = r[f"{m}_ci_low"]
                ci_hi = r[f"{m}_ci_high"]
                precision = 3 if m == "ssim" else 3
                line += f" & {mean:.{precision}f} [{ci_lo:.{precision}f}, {ci_hi:.{precision}f}]"
        line += r" \\"
        print(line)

    # Save CSV
    out_path = TABLES_DIR / "vfi_hemorrhage_stratification.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
