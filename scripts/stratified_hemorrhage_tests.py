"""Inferential hemorrhage-vs-normal stratified tests for §4.4 of the paper.

For each evaluated model (two classical baselines, two VFI baselines, four
learned models), and for each available metric (SSIM, MS-SSIM, MAE,
gradient MAE, PSNR, NCC, LPIPS), test whether the slice-level metric
distribution on hemorrhage triplets differs from that on normal triplets. The
paper's inferential family is restricted to the three primary metrics reported
in §4.4 (SSIM, MAE, PSNR); the remaining metric rows are retained in the CSV as
descriptive artefact rows without BH-adjusted q-values.

Design notes
------------
* Hemorrhage is defined at the target middle slice I_{k+1} of each triplet using
  the RSNA ``any`` subtype label (§4.4 convention). Labels are loaded from
  ``results/tables/revision_hemorrhage_labels_2026-04-16.csv``.
* The two groups are independent (different slices) so we use the two-sample
  Mann–Whitney U test (``scipy.stats.mannwhitneyu``, two-sided,
  ``use_continuity=True``, ``method='auto'``), NOT the paired Wilcoxon
  signed-rank test.
* Slices within a patient are not i.i.d.; to respect that clustering we also
  report a cluster-bootstrap p-value that resamples patients with replacement
  and recomputes U on each resample (two-sided percentile p, 10,000 resamples,
  seed 42).
* Effect size: rank-biserial correlation r = 2U/(n1·n2) − 1, reported as
  ``effect_r`` with hemorrhage as group 1 (Kerby 2014 convention). Positive r
  ⇒ hemorrhage slices tend to score higher than normal slices on the metric;
  for higher-is-better metrics (SSIM, MS-SSIM, PSNR, NCC) that means
  hemorrhage performs better, for lower-is-better metrics (MAE, gradient_MAE,
  LPIPS) it means hemorrhage performs worse.
* Hodges–Lehmann shift estimator: the median of all pairwise differences
  x_hem_i − x_norm_j, with a 95 % cluster-bootstrap percentile CI (same 10,000
  patient-resamples, seed 42). For metrics where lower is better (MAE,
  gradient_MAE, LPIPS) a positive shift means hemorrhage performs worse; for
  higher-is-better metrics (SSIM, MS-SSIM, PSNR, NCC) a negative shift means
  hemorrhage performs worse.
* Multiple-comparison correction: Benjamini–Hochberg FDR
  (``scipy.stats.false_discovery_control``) applied jointly over the full
  8 × 3 primary-metric family (model × {SSIM, MAE, PSNR}) — raw Mann–Whitney
  p-values are used as input. The cluster-bootstrap p is reported
  descriptively.

The analysis is post-hoc / exploratory.

Outputs
-------
``results/tables/hemorrhage_stratified_tests.csv`` with columns:

    model, metric, n_hem, n_norm, median_hem, median_norm, shift_HL,
    shift_CI_low, shift_CI_high, effect_r, U_stat, p_raw, p_cluster_boot,
    in_BH_family, q_BH, significant_at_q05

Run with::

    uv run scripts/stratified_hemorrhage_tests.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

# ---------------------------------------------------------------------------
# Configuration

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "results" / "tables"

METRICS_CSVS = [
    TABLES_DIR / "revision_slice_metrics_2026-04-16.csv",
    TABLES_DIR / "rife_baseline_slice_metrics.csv",
    TABLES_DIR / "film_baseline_slice_metrics.csv",
]
LABELS_CSV = TABLES_DIR / "revision_hemorrhage_labels_2026-04-16.csv"
OUTPUT_CSV = TABLES_DIR / "hemorrhage_stratified_tests.csv"

# Experiments reported in §4.4's hemorrhage table, in display order.
MODELS: list[tuple[str, str]] = [
    ("baseline_cubic", "Cubic baseline"),
    ("baseline_mean", "Mean baseline"),
    ("baseline_rife", "RIFE"),
    ("baseline_film", "FILM"),
    ("ssim_lr3e-3_94f982", "SSIM†"),
    ("l1_lr8e-4_b39be9", "L1"),
    ("msssim+l1_lr8e-4_bc1d65", "MS-SSIM+L1 (ref.)"),
    ("mse_lr8e-4_b558b9", "MSE"),
]

METRICS: list[tuple[str, str]] = [
    ("ssim", "SSIM"),
    ("ms_ssim", "MS-SSIM"),
    ("mae", "MAE"),
    ("gradient_mae", "gradient MAE"),
    ("psnr", "PSNR"),
    ("ncc", "NCC"),
    ("lpips", "LPIPS"),
]

BH_FAMILY_METRIC_IDS = frozenset({"ssim", "mae", "psnr"})

N_BOOT = 10000
SEED = 42


# ---------------------------------------------------------------------------
# Loading / joining


def load_metrics() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in METRICS_CSVS]
    metrics = pd.concat(frames, ignore_index=True)
    return metrics


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[["patient_id", "triplet_index", "target_any"]].copy()
    labels["hemorrhage"] = labels["target_any"].astype(int) == 1
    return labels[["patient_id", "triplet_index", "hemorrhage"]]


def join(metrics: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = metrics.merge(
        labels, on=["patient_id", "triplet_index"], how="inner", validate="many_to_one"
    )
    return merged


# ---------------------------------------------------------------------------
# Statistics


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann–Whitney U1 = U(group1, group2).

    SciPy returns ``U1 = #{ (x_i, y_j) : x_i > y_j } + 0.5 · ties``. Positive
    r under the convention r = 2 U1 / (n1 n2) − 1 then means group1 tends to
    be *larger* than group2 (Kerby 2014). We pass ``hem`` as group1, so
    positive r means hemorrhage slices score higher on the metric. For
    higher-is-better metrics (SSIM, MS-SSIM, PSNR, NCC) this means hemorrhage
    performs better; for lower-is-better (MAE, gradient_MAE, LPIPS) it means
    hemorrhage performs worse.
    """
    return 2.0 * u / (n1 * n2) - 1.0


def hodges_lehmann(x: np.ndarray, y: np.ndarray) -> float:
    """Hodges–Lehmann shift estimator: median of pairwise differences x_i − y_j.

    For moderate n this materialises an n1·n2 array (≈ 396·572 ≈ 2.3·10^5
    elements per call) which is trivial in memory but non-negligible when
    called inside a 2,000-iteration bootstrap (≈ 4.5·10^8 scalars). Use the
    Hoeffding trick of sampling pairs only when it becomes a bottleneck; for
    the sizes here the direct computation is fine (< 30 s total on a laptop).
    """
    diffs = np.subtract.outer(x, y).ravel()
    return float(np.median(diffs))


def cluster_bootstrap(
    df_model_metric: pd.DataFrame,
    metric_col: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    """Patient-level cluster bootstrap.

    Resamples patient IDs with replacement (preserving within-patient
    dependence), recomputes the standardized Mann–Whitney z-statistic

        z = (U - n1 n2 / 2) / sqrt(n1 n2 (n1 + n2 + 1) / 12)

    on each resample, and also recomputes the Hodges–Lehmann shift. Returns

        (p_cluster_boot, shift_CI_low, shift_CI_high)

    where ``p_cluster_boot`` is the two-sided percentile p obtained by
    inverting the bootstrap distribution of ``z``:

        p = 2 · min(P(z* ≤ 0), P(z* ≥ 0))

    Interpretation: under the null of identical hemorrhage/normal
    distributions, E[z*] = 0; deviation of the bootstrap distribution of z*
    from 0 quantifies how often the observed effect direction could have
    flipped had we sampled different patients. This is the standard
    cluster-bootstrap test-inversion p (Davison & Hinkley 1997, Ch. 4.4). It
    is one-to-one with the 95 % CI for z excluding zero, so it respects the
    within-patient clustering that the asymptotic Mann–Whitney p ignores.

    Note this is NOT an exchangeability test; it conditions on the observed
    hemorrhage/normal labels per patient and captures uncertainty from the
    finite 30-patient sample.
    """
    unique_patients, pat_idx = np.unique(
        df_model_metric["patient_id"].to_numpy(), return_inverse=True
    )
    n_pat = unique_patients.size
    values = df_model_metric[metric_col].to_numpy()
    hem_mask = df_model_metric["hemorrhage"].to_numpy()

    # Per-patient slice indices for fast concatenation.
    per_patient_rows = [np.where(pat_idx == p)[0] for p in range(n_pat)]

    z_samples = np.empty(n_boot)
    shift_samples = np.empty(n_boot)

    for b in range(n_boot):
        picks = rng.integers(0, n_pat, size=n_pat)
        # Concatenate per-patient row indices from the resampled patient ids.
        sel = np.concatenate([per_patient_rows[p] for p in picks])
        vals = values[sel]
        mask = hem_mask[sel]
        hem = vals[mask]
        norm = vals[~mask]
        n1, n2 = hem.size, norm.size
        if n1 == 0 or n2 == 0:
            z_samples[b] = np.nan
            shift_samples[b] = np.nan
            continue
        res = mannwhitneyu(
            hem, norm, alternative="two-sided", use_continuity=True, method="asymptotic"
        )
        u = float(res.statistic)
        mu = 0.5 * n1 * n2
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1.0) / 12.0)
        z_samples[b] = (u - mu) / sigma if sigma > 0 else np.nan
        shift_samples[b] = hodges_lehmann(hem, norm)

    valid = ~np.isnan(z_samples)
    if valid.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    z_valid = z_samples[valid]
    # Two-sided cluster-bootstrap p by inverting the CI for z at z = 0.
    frac_below = float(np.mean(z_valid <= 0))
    frac_above = float(np.mean(z_valid >= 0))
    p_boot = 2.0 * min(frac_below, frac_above)
    # Bootstrap resolution floor.
    p_boot = max(p_boot, 1.0 / (n_boot + 1))
    p_boot = min(p_boot, 1.0)

    shift_valid = shift_samples[valid]
    ci_low, ci_high = np.percentile(shift_valid, [2.5, 97.5])
    return p_boot, float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# Orchestration


def run_all(metrics_df: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows: list[dict] = []
    rng_master = np.random.default_rng(SEED)

    for exp_id, display in MODELS:
        sub = metrics_df.loc[metrics_df["experiment"] == exp_id]
        if sub.empty:
            raise RuntimeError(f"No rows for experiment {exp_id!r}")
        for metric_col, metric_display in METRICS:
            if metric_col not in sub.columns:
                raise RuntimeError(f"Metric {metric_col!r} missing for {exp_id!r}")

            data = sub[["patient_id", "triplet_index", metric_col, "hemorrhage"]].dropna()
            hem = data.loc[data["hemorrhage"], metric_col].to_numpy()
            norm = data.loc[~data["hemorrhage"], metric_col].to_numpy()
            n_hem, n_norm = hem.size, norm.size

            mw = mannwhitneyu(
                hem, norm, alternative="two-sided", use_continuity=True, method="auto"
            )
            u_stat = float(mw.statistic)
            p_raw = float(mw.pvalue)
            r_eff = rank_biserial(u_stat, n_hem, n_norm)
            shift_hl = hodges_lehmann(hem, norm)

            # Seed each (model, metric) deterministically from the master RNG
            # so reruns reproduce exactly.
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            p_boot, ci_low, ci_high = cluster_bootstrap(data, metric_col, rng, n_boot=n_boot)

            rows.append(
                {
                    "model_id": exp_id,
                    "model": display,
                    "metric_id": metric_col,
                    "metric": metric_display,
                    "n_hem": n_hem,
                    "n_norm": n_norm,
                    "median_hem": float(np.median(hem)),
                    "median_norm": float(np.median(norm)),
                    "shift_HL": shift_hl,
                    "shift_CI_low": ci_low,
                    "shift_CI_high": ci_high,
                    "effect_r": r_eff,
                    "U_stat": u_stat,
                    "p_raw": p_raw,
                    "p_cluster_boot": p_boot,
                }
            )

    out = pd.DataFrame(rows)
    # BH FDR across the manuscript's full inferential family:
    # 8 methods × 3 primary metrics (SSIM, MAE, PSNR) = 24 tests.
    in_family = out["metric_id"].isin(BH_FAMILY_METRIC_IDS)
    out["in_BH_family"] = in_family
    out["q_BH"] = np.nan
    out["significant_at_q05"] = pd.NA
    out.loc[in_family, "q_BH"] = false_discovery_control(
        out.loc[in_family, "p_raw"].to_numpy(), method="bh"
    )
    out.loc[in_family, "significant_at_q05"] = out.loc[in_family, "q_BH"] < 0.05
    return out


def main() -> None:
    global N_BOOT  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    N_BOOT = args.n_boot

    print(f"Loading metrics from {len(METRICS_CSVS)} CSVs ...")
    metrics = load_metrics()
    print(f"  total rows: {len(metrics):,}")
    print(f"Loading labels from {LABELS_CSV.name} ...")
    labels = load_labels()
    print(f"  label rows: {len(labels):,}  (hem={int(labels['hemorrhage'].sum())}, "
          f"norm={int((~labels['hemorrhage']).sum())})")

    merged = join(metrics, labels)
    print(f"After join: {len(merged):,} rows, "
          f"{merged['experiment'].nunique()} experiments, "
          f"{merged['patient_id'].nunique()} patients")

    # Sanity: per-experiment row count should equal 968.
    per_exp = merged.groupby("experiment").size()
    bad = per_exp[per_exp != 968]
    if not bad.empty:
        print("WARNING: unexpected row counts per experiment:")
        print(bad)

    out = run_all(merged, n_boot=N_BOOT)
    cols = [
        "model",
        "metric",
        "n_hem",
        "n_norm",
        "median_hem",
        "median_norm",
        "shift_HL",
        "shift_CI_low",
        "shift_CI_high",
        "effect_r",
        "U_stat",
        "p_raw",
        "p_cluster_boot",
        "in_BH_family",
        "q_BH",
        "significant_at_q05",
        "model_id",
        "metric_id",
    ]
    out = out[cols]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}  ({len(out)} rows)")

    # Compact console summary.
    family = out.loc[out["in_BH_family"]].copy()
    summary = family.pivot(index="model", columns="metric", values="significant_at_q05")
    print(
        "\nBH-significant (q<0.05) over the 8 × 3 primary-metric family "
        "(SSIM, MAE, PSNR):"
    )
    print(summary.to_string())

    print("\nRaw p and BH q (key metrics):")
    key = family[
        ["model", "metric", "p_raw", "p_cluster_boot", "q_BH", "significant_at_q05"]
    ]
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
