"""Analyze radiologist reader study results.

Computes:
  - Per-reader summary statistics (mean, median, IQR per criterion)
  - Inter-rater agreement: Cohen's Kappa (weighted, quadratic)
  - ICC(2,1) for absolute agreement
  - Stratified analysis: hemorrhage vs normal
  - Correlation with automated metrics (SSIM, LPIPS, MAE)
  - Summary table for paper inclusion

Usage:
    uv run scripts/reader_study_analyze.py \
        --scores reader_study/scores.csv \
        --cases reader_study/cases.csv \
        --slice-metrics results/tables/revision_slice_metrics_2026-03-03.csv \
        --output-dir reader_study/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

CRITERIA = ["anatomical_fidelity", "diagnostic_adequacy", "overall_quality"]
REQUIRED_SCORE_COLS = {"case_id", "reader_id", *CRITERIA}


# --- Data loading -----------------------------------------------------------


def load_scores(csv_path: Path) -> pd.DataFrame:
    """Load reader scores CSV and validate required columns."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_SCORE_COLS - set(df.columns)
    if missing:
        msg = f"Scores CSV missing columns: {missing}"
        raise ValueError(msg)

    # Validate score ranges
    for col in CRITERIA:
        vals = df[col]
        if vals.min() < 1 or vals.max() > 5:
            msg = f"Scores in '{col}' out of range [1,5]: min={vals.min()}, max={vals.max()}"
            raise ValueError(msg)

    return df


def load_cases(csv_path: Path) -> pd.DataFrame:
    """Load cases CSV with hemorrhage status."""
    df = pd.read_csv(csv_path)
    required = {"case_id", "patient_id", "triplet_index", "hemorrhage_status"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Cases CSV missing columns: {missing}"
        raise ValueError(msg)
    return df


def load_slice_metrics(csv_path: Path, experiment: str = "ssim_lr3e-3_94f982") -> pd.DataFrame:
    """Load per-triplet automated metrics for a specific experiment."""
    df = pd.read_csv(csv_path)
    df = df[df["experiment"] == experiment].copy()
    if df.empty:
        logger.warning(f"No metrics found for experiment '{experiment}' in {csv_path}")
    return df


# --- Per-reader statistics ---------------------------------------------------


def compute_per_reader_stats(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, median, IQR per reader per criterion.

    Returns a DataFrame with columns: reader_id, criterion, mean, median, q1, q3, iqr.
    """
    rows: list[dict] = []
    for reader_id in sorted(scores["reader_id"].unique()):
        reader_df = scores[scores["reader_id"] == reader_id]
        for criterion in CRITERIA:
            vals = reader_df[criterion].dropna()
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            rows.append(
                {
                    "reader_id": reader_id,
                    "criterion": criterion,
                    "n": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(vals.median()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                }
            )
    return pd.DataFrame(rows)


# --- Inter-rater agreement ---------------------------------------------------


def cohens_kappa_weighted(
    ratings_a: np.ndarray,
    ratings_b: np.ndarray,
    num_categories: int = 5,
) -> float:
    """Compute quadratic-weighted Cohen's Kappa.

    Args:
        ratings_a: Array of integer ratings from reader A (1-based).
        ratings_b: Array of integer ratings from reader B (1-based).
        num_categories: Number of categories (default 5 for 1-5 scale).

    Returns:
        Quadratic-weighted kappa coefficient.
    """
    n = len(ratings_a)
    if n != len(ratings_b):
        msg = f"Rating arrays must have same length: {len(ratings_a)} vs {len(ratings_b)}"
        raise ValueError(msg)
    if n == 0:
        return float("nan")

    # Build observed agreement matrix
    min_rating = 1
    confusion = np.zeros((num_categories, num_categories), dtype=np.float64)
    for a, b in zip(ratings_a, ratings_b, strict=True):
        i = int(a) - min_rating
        j = int(b) - min_rating
        confusion[i, j] += 1

    # Build weight matrix (quadratic)
    weights = np.zeros((num_categories, num_categories), dtype=np.float64)
    for i in range(num_categories):
        for j in range(num_categories):
            weights[i, j] = ((i - j) / (num_categories - 1)) ** 2

    # Expected agreement matrix
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    # Weighted kappa
    observed_disagreement = (weights * confusion).sum() / n
    expected_disagreement = (weights * expected).sum() / n

    if expected_disagreement == 0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def compute_icc_2_1(scores: pd.DataFrame, criterion: str) -> dict:
    """Compute ICC(2,1) for absolute agreement (two-way random, single measures).

    Uses the standard ANOVA-based formulation:
        ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)

    Where:
        MSR = mean square for rows (subjects)
        MSC = mean square for columns (raters)
        MSE = mean square for error (residual)
        k = number of raters
        n = number of subjects

    Args:
        scores: DataFrame with reader_id, case_id, and criterion columns.
        criterion: Which criterion column to analyze.

    Returns:
        Dict with icc, f_value, df1, df2, p_value, ci_lower, ci_upper.
    """
    readers = sorted(scores["reader_id"].unique())
    cases = sorted(scores["case_id"].unique())
    k = len(readers)
    n = len(cases)

    if k < 2:
        logger.warning(f"ICC requires at least 2 raters, got {k}")
        return {"icc": float("nan"), "f_value": float("nan"), "p_value": float("nan")}

    # Build ratings matrix (n subjects x k raters)
    matrix = np.full((n, k), np.nan)
    reader_map = {r: j for j, r in enumerate(readers)}
    case_map = {c: i for i, c in enumerate(cases)}

    for _, row in scores.iterrows():
        i = case_map[row["case_id"]]
        j = reader_map[row["reader_id"]]
        matrix[i, j] = row[criterion]

    # Check for missing data
    if np.any(np.isnan(matrix)):
        logger.warning(f"Missing ratings detected for {criterion}; results may be unreliable")
        # Use only complete cases
        complete = ~np.any(np.isnan(matrix), axis=1)
        matrix = matrix[complete]
        n = matrix.shape[0]
        if n < 2:
            return {"icc": float("nan"), "f_value": float("nan"), "p_value": float("nan")}

    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    # Sum of squares
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    ms_rows = ss_rows / df_rows if df_rows > 0 else 0
    ms_cols = ss_cols / df_cols if df_cols > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0

    # ICC(2,1) formula
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denominator == 0:
        icc = float("nan")
    else:
        icc = (ms_rows - ms_error) / denominator

    # F-test
    if ms_error == 0:
        f_value = float("inf")
        p_value = 0.0
    else:
        f_value = ms_rows / ms_error
        p_value = 1.0 - stats.f.cdf(f_value, df_rows, df_error)

    # 95% CI using Shrout & Fleiss (1979) approximation
    # For ICC(2,1):
    f_l = f_value / stats.f.ppf(0.975, df_rows, df_error) if df_error > 0 else float("nan")
    f_u = f_value * stats.f.ppf(0.975, df_error, df_rows) if df_rows > 0 else float("nan")

    if np.isnan(f_l) or np.isinf(f_l):
        ci_lower = 1.0 if np.isinf(f_l) else float("nan")
    else:
        ci_lower = (f_l - 1) / (f_l + k - 1)

    if np.isnan(f_u) or np.isinf(f_u):
        ci_upper = 1.0 if np.isinf(f_u) else float("nan")
    else:
        ci_upper = (f_u - 1) / (f_u + k - 1)

    return {
        "icc": float(icc),
        "f_value": float(f_value),
        "df1": int(df_rows),
        "df2": int(df_error),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def compute_inter_rater_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's Kappa and ICC for all criteria.

    Returns DataFrame with one row per criterion.
    """
    readers = sorted(scores["reader_id"].unique())
    if len(readers) != 2:
        logger.warning(f"Expected 2 readers for Cohen's Kappa, got {len(readers)}")

    rows: list[dict] = []
    for criterion in CRITERIA:
        result = {"criterion": criterion}

        # Cohen's Kappa (pairwise, only for exactly 2 readers)
        if len(readers) == 2:
            cases_both = set(
                scores[scores["reader_id"] == readers[0]]["case_id"]
            ) & set(
                scores[scores["reader_id"] == readers[1]]["case_id"]
            )
            mask_0 = (scores["reader_id"] == readers[0]) & (scores["case_id"].isin(cases_both))
            mask_1 = (scores["reader_id"] == readers[1]) & (scores["case_id"].isin(cases_both))
            r0 = scores[mask_0].sort_values("case_id")
            r1 = scores[mask_1].sort_values("case_id")
            result["cohens_kappa_weighted"] = cohens_kappa_weighted(
                r0[criterion].values,
                r1[criterion].values,
            )
        else:
            result["cohens_kappa_weighted"] = float("nan")

        # ICC(2,1)
        icc_result = compute_icc_2_1(scores, criterion)
        result.update({f"icc_{k}": v for k, v in icc_result.items()})

        rows.append(result)

    return pd.DataFrame(rows)


# --- Stratified analysis -----------------------------------------------------


def compute_stratified_stats(
    scores: pd.DataFrame,
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-criterion stats stratified by hemorrhage status.

    Returns a DataFrame with columns: hemorrhage_status, criterion, n, mean, std, median, q1, q3.
    """
    merged = scores.merge(cases[["case_id", "hemorrhage_status"]], on="case_id", how="left")

    rows: list[dict] = []
    for status in ["hemorrhage", "normal"]:
        subset = merged[merged["hemorrhage_status"] == status]
        for criterion in CRITERIA:
            vals = subset[criterion].dropna()
            q1 = float(np.percentile(vals, 25)) if len(vals) > 0 else float("nan")
            q3 = float(np.percentile(vals, 75)) if len(vals) > 0 else float("nan")
            rows.append(
                {
                    "hemorrhage_status": status,
                    "criterion": criterion,
                    "n": len(vals),
                    "mean": float(vals.mean()) if len(vals) > 0 else float("nan"),
                    "std": float(vals.std()) if len(vals) > 0 else float("nan"),
                    "median": float(vals.median()) if len(vals) > 0 else float("nan"),
                    "q1": q1,
                    "q3": q3,
                }
            )

    return pd.DataFrame(rows)


# --- Correlation with automated metrics --------------------------------------


def compute_metric_correlations(
    scores: pd.DataFrame,
    cases: pd.DataFrame,
    slice_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Spearman correlation between reader scores and automated metrics.

    For each case, we average the reader scores, then correlate with the
    automated metric for that triplet.

    Returns DataFrame with: criterion, metric, rho, p_value, n.
    """
    if slice_metrics.empty:
        logger.warning("No slice metrics available for correlation analysis")
        return pd.DataFrame()

    # Average reader scores per case
    avg_scores = scores.groupby("case_id")[CRITERIA].mean().reset_index()

    # Merge with cases to get patient_id and triplet_index
    avg_with_cases = avg_scores.merge(
        cases[["case_id", "patient_id", "triplet_index"]],
        on="case_id",
    )

    # Merge with slice metrics
    merged = avg_with_cases.merge(
        slice_metrics,
        on=["patient_id", "triplet_index"],
        how="inner",
    )

    if len(merged) == 0:
        logger.warning("No matching triplets between reader scores and slice metrics")
        return pd.DataFrame()

    logger.info(f"Computing correlations on {len(merged)} matched triplets")

    auto_metrics = ["ssim", "ms_ssim", "mae", "gradient_mae", "psnr", "ncc", "lpips"]
    available_metrics = [m for m in auto_metrics if m in merged.columns]

    rows: list[dict] = []
    for criterion in CRITERIA:
        for metric in available_metrics:
            vals_criterion = merged[criterion].values
            vals_metric = merged[metric].values

            # Drop NaN pairs
            valid = ~(np.isnan(vals_criterion) | np.isnan(vals_metric))
            if valid.sum() < 3:
                continue

            rho, p_value = stats.spearmanr(vals_criterion[valid], vals_metric[valid])
            rows.append(
                {
                    "criterion": criterion,
                    "automated_metric": metric,
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "n": int(valid.sum()),
                }
            )

    return pd.DataFrame(rows)


# --- Summary table for paper -------------------------------------------------


def format_summary_table(
    per_reader: pd.DataFrame,
    agreement: pd.DataFrame,
    stratified: pd.DataFrame,
) -> str:
    """Format a summary table suitable for inclusion in the paper."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("READER STUDY SUMMARY")
    lines.append("=" * 80)

    # Overall scores
    lines.append("\n--- Per-Reader Summary (Mean +/- SD [Median, IQR]) ---")
    for _, row in per_reader.iterrows():
        lines.append(
            f"  Reader {row['reader_id']}, {row['criterion']}: "
            f"{row['mean']:.2f} +/- {row['std']:.2f} "
            f"[{row['median']:.1f}, {row['q1']:.1f}-{row['q3']:.1f}]"
        )

    # Agreement
    lines.append("\n--- Inter-Rater Agreement ---")
    for _, row in agreement.iterrows():
        kappa_str = f"{row['cohens_kappa_weighted']:.3f}" if not np.isnan(row["cohens_kappa_weighted"]) else "N/A"
        icc_str = f"{row['icc_icc']:.3f}" if not np.isnan(row["icc_icc"]) else "N/A"
        ci_str = ""
        if not np.isnan(row.get("icc_ci_lower", float("nan"))) and not np.isnan(row.get("icc_ci_upper", float("nan"))):
            ci_str = f" [{row['icc_ci_lower']:.3f}, {row['icc_ci_upper']:.3f}]"
        lines.append(
            f"  {row['criterion']}: "
            f"Kappa_w={kappa_str}, "
            f"ICC(2,1)={icc_str}{ci_str}"
        )

    # Stratified
    lines.append("\n--- Stratified by Hemorrhage Status (Mean +/- SD) ---")
    for status in ["hemorrhage", "normal"]:
        sub = stratified[stratified["hemorrhage_status"] == status]
        for _, row in sub.iterrows():
            lines.append(
                f"  {status.capitalize():12s} | {row['criterion']}: "
                f"{row['mean']:.2f} +/- {row['std']:.2f} (n={row['n']})"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


# --- Main --------------------------------------------------------------------


def main() -> None:
    """Run reader study analysis."""
    parser = argparse.ArgumentParser(description="Analyze radiologist reader study results")
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("reader_study/scores.csv"),
        help="Path to filled scores CSV",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("reader_study/cases.csv"),
        help="Path to cases CSV from selection script",
    )
    parser.add_argument(
        "--slice-metrics",
        type=Path,
        default=Path("results/tables/revision_slice_metrics_2026-03-03.csv"),
        help="Path to per-triplet automated metrics CSV",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ssim_lr3e-3_94f982",
        help="Experiment name to use for automated metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reader_study/results"),
        help="Output directory for analysis results",
    )
    args = parser.parse_args()

    # Load data
    if not args.scores.exists():
        logger.error(
            f"Scores file not found: {args.scores}\n"
            "The radiologists need to fill in reader_study/scores.csv first.\n"
            "A template is available at reader_study/scores_template.csv"
        )
        return

    scores = load_scores(args.scores)
    cases = load_cases(args.cases)

    readers = sorted(scores["reader_id"].unique())
    logger.info(f"Loaded scores from {len(readers)} readers for {scores['case_id'].nunique()} cases")

    # Load automated metrics (optional)
    slice_metrics = pd.DataFrame()
    if args.slice_metrics.exists():
        slice_metrics = load_slice_metrics(args.slice_metrics, args.experiment)
        logger.info(f"Loaded {len(slice_metrics)} automated metric entries")
    else:
        logger.warning(f"Slice metrics file not found: {args.slice_metrics}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-reader statistics
    per_reader = compute_per_reader_stats(scores)
    per_reader.to_csv(args.output_dir / "per_reader_stats.csv", index=False)
    logger.info("Per-reader statistics computed")

    # 2. Inter-rater agreement
    agreement = compute_inter_rater_agreement(scores)
    agreement.to_csv(args.output_dir / "inter_rater_agreement.csv", index=False)
    logger.info("Inter-rater agreement computed")

    # 3. Stratified analysis
    stratified = compute_stratified_stats(scores, cases)
    stratified.to_csv(args.output_dir / "stratified_stats.csv", index=False)
    logger.info("Stratified analysis computed")

    # 4. Correlation with automated metrics
    if not slice_metrics.empty:
        correlations = compute_metric_correlations(scores, cases, slice_metrics)
        if not correlations.empty:
            correlations.to_csv(args.output_dir / "metric_correlations.csv", index=False)
            logger.info("Metric correlations computed")
        else:
            logger.warning("No metric correlations could be computed")
    else:
        logger.warning("Skipping metric correlations (no automated metrics available)")

    # 5. Summary
    summary = format_summary_table(per_reader, agreement, stratified)
    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info(f"Summary saved to {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
