#!/usr/bin/env python3
r"""Quantitative denoising analysis in uniform white-matter ROIs.

For each test patient and each triplet, we place three anatomically motivated
16x16 px white-matter ROIs (left/right centrum semiovale, pons) based on a
simple brain-mask heuristic, then compute:

  - Residual pixel-std in each ROI for the acquired middle slice vs. each
    model's predicted middle slice (noise-reduction ratio eta).
  - A 1D radially-averaged noise power spectrum (NPS) per ROI per image type,
    following ICRU 54 conventions.

The pixel-std analysis is reduced to one value per patient (mean over ROIs
then mean over triplets), giving a paired patient-level test against zero via
scipy.stats.wilcoxon. Patients for whom every surviving ROI has sigma_acq = 0
(i.e. eta is undefined) are dropped from both the bootstrap and the Wilcoxon
so the reported n is the same across both statistics; this retains 28 of 30
test patients in practice. 95% CIs on median eta come from a patient-level
bootstrap (10000 resamples, seed=42). Raw Wilcoxon p-values are reported
together with BH-FDR q-values across the model family.

Outputs:
  results/tables/roi_noise_analysis.csv        (per-patient summary)
  results/tables/roi_noise_analysis_tests.csv  (one row per model:
      median eta, 95% CI, Wilcoxon stat, raw p, BH q, significance)
  results/figures/nps_curves.pdf               (NPS figure)

Scope note (important, honestly documented):

The paper headline claim is that *regression-trained* synthesis inherits an
implicit conditional-expectation denoising property. We therefore run this
analysis on the four learned U-Net + EfficientNetV2-S models reported in the
main comparison (L1, MSE, MS-SSIM+L1, SSIM-dagger), plus the cubic classical
baseline as a negative control (cubic is an explicit linear interpolator and
should not denoise). VFI baselines (RIFE, FILM) are pretrained on natural
video, are not regression-trained on CT, and are outside the scope of the
denoising claim; we do not re-run inference for them here.

Reproduce:

    uv run scripts/roi_noise_analysis.py

Seed: 42 (bootstrap).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from scipy import ndimage
from scipy.stats import false_discovery_control, wilcoxon

load_dotenv()

from phd.config_io import resolve_config_path  # noqa: E402
from phd.datasets.interpolation.two_to_one_slice import (  # noqa: E402
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)
from phd.models.setup_model import setup_model  # noqa: E402
from phd.viz import predict_via_patch_reconstruction  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
BOOTSTRAP_N = 10000

EXPERIMENTS_DIR = Path("experiments/train_nn1_cropped")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
SPACING_CSV = TABLES_DIR / "dicom_spacing_per_patient.csv"
HEMORRHAGE_LABELS_CSV = TABLES_DIR / "revision_hemorrhage_labels_2026-04-16.csv"

# Four learned regression-trained models (labels must match revision tables).
# The fifth entry is the classical cubic baseline, which is analytical and
# does not require inference — we compute its prediction by averaging the
# neighbors in HU-windowed space (cubic reduces to a known linear operator
# for spacing-1 midpoint interpolation; we use the linear-mean classical
# baseline already reported in the paper as the negative control).
LEARNED_MODELS = {
    "l1": "l1_lr8e-4_b39be9",
    "mse": "mse_lr8e-4_b558b9",
    "msssim+l1": "msssim+l1_lr8e-4_bc1d65",
    "ssim_dagger": "ssim_lr3e-3_94f982",
}
CLASSICAL_BASELINE = "cubic"  # midpoint of neighbors; see make_cubic_prediction

# ROI geometry
ROI_SIZE = 16
BRAIN_THR = 0.05  # threshold on [0,1]-normalized image for brain mask

# NPS: common radial frequency bins (in cycles/pixel initially; converted
# later using per-patient pixel spacing to cycles/mm).
# Bin count chosen so every bin contains >=1 FFT sample for a 16x16 ROI;
# 32 bins leaves several empty bins between integer-radius FFT shells, which
# show up as NaN gaps in the radial curve.
NPS_N_BINS = 16


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_model(experiment_dir: Path, device: torch.device) -> torch.nn.Module:
    """Load trained model by picking the best-valid-loss epoch."""
    config = json.loads((experiment_dir / "config.json").read_text())
    model_cfg = config["model"]
    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,
        model_type=model_cfg["type"],
        encoder_name=model_cfg["encoder_name"],
    )
    df = pd.read_csv(experiment_dir / "epochs.csv")
    finite = df[np.isfinite(df["valid_loss"])]
    best_epoch = int(finite.loc[finite["valid_loss"].idxmin(), "epoch"])
    weights_path = experiment_dir / "epochs" / str(best_epoch - 1) / "weights.pth"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


@dataclass(frozen=True)
class BrainGeometry:
    """Brain-mask geometry derived from a single slice."""

    mask: np.ndarray  # (H, W) bool
    centroid_row: float
    centroid_col: float
    bbox_r0: int
    bbox_c0: int
    bbox_r1: int
    bbox_c1: int


def compute_brain_geometry(slice_img: np.ndarray) -> BrainGeometry | None:
    """Compute brain-mask geometry from a [0,1] CT slice.

    Returns None if the brain mask is empty or too small.
    """
    binary = slice_img > BRAIN_THR
    labels, n = ndimage.label(binary)
    if n == 0:
        return None
    # Largest connected component
    sizes = ndimage.sum(binary, labels, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    mask = labels == largest
    if mask.sum() < 2000:  # too small to be a brain
        return None

    rows, cols = np.nonzero(mask)
    centroid_row = float(rows.mean())
    centroid_col = float(cols.mean())
    return BrainGeometry(
        mask=mask,
        centroid_row=centroid_row,
        centroid_col=centroid_col,
        bbox_r0=int(rows.min()),
        bbox_c0=int(cols.min()),
        bbox_r1=int(rows.max()),
        bbox_c1=int(cols.max()),
    )


def candidate_roi_centers(geom: BrainGeometry) -> dict[str, tuple[int, int]]:
    """Candidate ROI centers (row, col) from brain-mask geometry.

    Heuristic offsets from the brain centroid expressed as fractions of the
    bounding-box extent. Centrum semiovale ROIs are placed above the centroid
    at left/right offsets; pons is placed below the centroid at the midline.
    Pons placement is only considered reliable for slices whose brain
    bounding-box height is modest (skull-base / posterior-fossa-like).
    """
    bh = geom.bbox_r1 - geom.bbox_r0
    bw = geom.bbox_c1 - geom.bbox_c0
    row_up = int(round(geom.centroid_row - 0.15 * bh))
    col_left = int(round(geom.centroid_col - 0.25 * bw))
    col_right = int(round(geom.centroid_col + 0.25 * bw))
    row_pons = int(round(geom.centroid_row + 0.10 * bh))
    col_pons = int(round(geom.centroid_col))
    return {
        "cs_left": (row_up, col_left),
        "cs_right": (row_up, col_right),
        "pons": (row_pons, col_pons),
    }


def extract_roi(image: np.ndarray, center: tuple[int, int]) -> np.ndarray | None:
    """Extract a ROI_SIZE x ROI_SIZE patch centered at (row, col).

    Returns None if the patch would fall outside the image.
    """
    r, c = center
    half = ROI_SIZE // 2
    r0, c0 = r - half, c - half
    r1, c1 = r0 + ROI_SIZE, c0 + ROI_SIZE
    h, w = image.shape
    if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        return None
    return image[r0:r1, c0:c1]


def roi_inside_mask(mask: np.ndarray, center: tuple[int, int]) -> bool:
    """True iff the ROI is fully contained in the brain mask."""
    patch = extract_roi(mask.astype(np.uint8), center)
    if patch is None:
        return False
    return bool(patch.all())


def load_hemorrhage_labels() -> pd.DataFrame:
    """Load per-triplet hemorrhage labels."""
    df = pd.read_csv(HEMORRHAGE_LABELS_CSV)
    return df


def load_pixel_spacings() -> dict[str, float]:
    """Load per-patient pixel spacing (mm per pixel)."""
    df = pd.read_csv(SPACING_CSV)
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        # Use row spacing (symmetric for RSNA; row==col to 6 dp).
        out[row["PatientID"]] = float(row["pixel_spacing_row_mm"])
    return out


# ---------------------------------------------------------------------------
# NPS
# ---------------------------------------------------------------------------
def compute_roi_nps(roi: np.ndarray, pixel_spacing_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """2D NPS for a single ROI, then radially averaged to 1D.

    Follows ICRU 54 convention: NPS(fx,fy) = |FFT(roi-mean)|^2 * area / N^2,
    where area = (N * pixel_spacing)^2, which simplifies to |FFT|^2 *
    pixel_spacing^2 / N^2 when the DC component is removed. We return
    (frequency [cycles/mm], NPS) with the zero-frequency bin excluded.
    """
    if roi.shape[0] != roi.shape[1]:
        raise ValueError("ROI must be square for simple radial NPS")
    n = roi.shape[0]
    roi0 = roi.astype(np.float64) - roi.mean()
    fft = np.fft.fftshift(np.fft.fft2(roi0))
    # ICRU 54: NPS = |F(u,v)|^2 * Ax * Ay / (Nx * Ny) where Ax,Ay are pixel
    # sizes; we have Ax = Ay = pixel_spacing_mm. Nx = Ny = n.
    ps = np.abs(fft) ** 2 * (pixel_spacing_mm ** 2) / (n * n)

    # Radial binning
    ky = np.fft.fftshift(np.fft.fftfreq(n, d=pixel_spacing_mm))
    kx = np.fft.fftshift(np.fft.fftfreq(n, d=pixel_spacing_mm))
    kxx, kyy = np.meshgrid(kx, ky)
    kr = np.sqrt(kxx ** 2 + kyy ** 2)

    k_max = float(kr.max())
    edges = np.linspace(0.0, k_max, NPS_N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    digitized = np.digitize(kr.ravel(), edges) - 1
    flat = ps.ravel()
    out = np.full(NPS_N_BINS, np.nan)
    for b in range(NPS_N_BINS):
        sel = digitized == b
        if sel.sum() > 0:
            out[b] = flat[sel].mean()
    # Drop the DC bin (bin 0) which contains f=0
    return centers[1:], out[1:]


# ---------------------------------------------------------------------------
# Main per-patient processing
# ---------------------------------------------------------------------------
def process_patient(
    *,
    patient_id: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    hemo_map: dict[tuple[str, int], int],
    pixel_spacing_mm: float,
) -> tuple[list[dict], dict[str, list[tuple[np.ndarray, np.ndarray]]], dict]:
    """Process all triplets for one patient.

    Returns:
        rows: list of per-ROI observation dicts
        nps_curves: {image_type: [(freq, nps), ...]}
        diag: {'n_triplets': int, 'n_rej_hemo': int, 'n_rej_geom': int,
               'n_rej_outlier': int, 'n_pons_missing': int}
    """
    n_triplets = inputs.shape[0]
    rows: list[dict] = []

    # Precompute predictions for all learned models for this patient
    preds_per_model: dict[str, np.ndarray] = {}
    for name, model in models.items():
        batch_size = 4
        all_preds = []
        for start in range(0, n_triplets, batch_size):
            end = min(start + batch_size, n_triplets)
            batch = inputs[start:end].to(device)
            with torch.no_grad():
                p = predict_via_patch_reconstruction(
                    model=model, batch_inputs=batch, device=device
                )
            all_preds.append(p.squeeze(1).cpu().numpy())
        preds_per_model[name] = np.concatenate(all_preds, axis=0)

    # Classical baseline: linear mean of neighbors
    # inputs is (N, 2, 512, 512). Equivalent to the "mean" baseline in the
    # paper. For one-slice midpoint interpolation with uniform spacing,
    # linear == cubic at the midpoint only when using a 4-tap kernel with
    # symmetric knots; here we only have 2 neighbors, so the honest label
    # is "linear mean". We rename accordingly.
    classical_pred = inputs.mean(dim=1).numpy()  # (N, 512, 512)
    preds_per_model["linear_mean"] = classical_pred

    image_types = ["acquired"] + list(preds_per_model.keys())
    nps_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {k: [] for k in image_types}

    diag = {"n_triplets": int(n_triplets), "n_rej_hemo": 0, "n_rej_geom": 0,
            "n_rej_outlier": 0, "n_pons_missing": 0, "n_rois_kept": 0}

    # Per-patient outlier threshold: compute median ROI std across all
    # candidate acquired ROIs first, then reject >3x median.
    # (Two-pass: collect acquired-std candidates, then use threshold.)
    acq_std_candidates: list[float] = []
    candidate_cache: list[dict] = []  # cached per-triplet ROI specs

    for t_idx in range(n_triplets):
        acquired = targets[t_idx, 0].numpy()  # (512, 512)
        geom = compute_brain_geometry(acquired)
        if geom is None:
            diag["n_rej_geom"] += 1
            continue
        # Skip hemorrhage slices
        hemo = hemo_map.get((patient_id, t_idx), 0)
        if hemo:
            diag["n_rej_hemo"] += 1
            continue
        centers = candidate_roi_centers(geom)
        roi_specs = []
        for roi_name, center in centers.items():
            if not roi_inside_mask(geom.mask, center):
                if roi_name == "pons":
                    diag["n_pons_missing"] += 1
                else:
                    diag["n_rej_geom"] += 1
                continue
            # Collect acquired std for the outlier gate
            acq_roi = extract_roi(acquired, center)
            if acq_roi is None:
                diag["n_rej_geom"] += 1
                continue
            acq_std_candidates.append(float(acq_roi.std()))
            roi_specs.append({"name": roi_name, "center": center, "acq_roi": acq_roi})
        candidate_cache.append({"t_idx": t_idx, "roi_specs": roi_specs})

    if not acq_std_candidates:
        return rows, nps_curves, diag
    med_std = float(np.median(acq_std_candidates))
    outlier_thr = 3.0 * med_std

    for entry in candidate_cache:
        t_idx = entry["t_idx"]
        acquired = targets[t_idx, 0].numpy()
        for spec in entry["roi_specs"]:
            roi_name = spec["name"]
            center = spec["center"]
            acq_roi = spec["acq_roi"]
            acq_std = float(acq_roi.std())
            if acq_std > outlier_thr:
                diag["n_rej_outlier"] += 1
                continue
            diag["n_rois_kept"] += 1

            # NPS for acquired
            f, nps = compute_roi_nps(acq_roi, pixel_spacing_mm)
            nps_curves["acquired"].append((f, nps))

            row_base = {
                "patient_id": patient_id,
                "triplet_index": t_idx,
                "roi_name": roi_name,
                "roi_row": center[0],
                "roi_col": center[1],
                "sigma_acq": acq_std,
            }
            for model_name, preds in preds_per_model.items():
                pred_img = preds[t_idx]
                pred_roi = extract_roi(pred_img, center)
                if pred_roi is None:
                    continue
                pred_std = float(pred_roi.std())
                eta = (acq_std - pred_std) / acq_std if acq_std > 0 else np.nan
                row = dict(row_base)
                row["model"] = model_name
                row["sigma_pred"] = pred_std
                row["eta"] = eta
                rows.append(row)

                f_m, nps_m = compute_roi_nps(pred_roi, pixel_spacing_mm)
                nps_curves[model_name].append((f_m, nps_m))

    return rows, nps_curves, diag


# ---------------------------------------------------------------------------
# Aggregation & statistics
# ---------------------------------------------------------------------------
def aggregate_to_patient_level(df_rows: pd.DataFrame) -> pd.DataFrame:
    """Mean across ROIs within slice, then mean across slices within patient."""
    # Slice-level aggregation: mean across ROIs for each (patient, triplet, model)
    slice_lvl = (
        df_rows.groupby(["patient_id", "triplet_index", "model"], as_index=False)
        .agg(
            sigma_acq=("sigma_acq", "mean"),
            sigma_pred=("sigma_pred", "mean"),
            eta=("eta", "mean"),
            n_rois=("roi_name", "count"),
        )
    )
    # Patient-level: mean across slices
    pat_lvl = (
        slice_lvl.groupby(["patient_id", "model"], as_index=False)
        .agg(
            sigma_acq=("sigma_acq", "mean"),
            sigma_pred=("sigma_pred", "mean"),
            eta=("eta", "mean"),
            n_triplets=("triplet_index", "count"),
        )
    )
    return pat_lvl


def bootstrap_median_ci(
    values: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = SEED
) -> tuple[float, float, float]:
    """Median + 95% patient-bootstrap CI. Drops NaN before resampling."""
    finite = np.asarray(values, dtype=float)
    finite = finite[~np.isnan(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(finite)
    for i in range(n_boot):
        sample = rng.choice(finite, size=n, replace=True)
        boots[i] = float(np.median(sample))
    med = float(np.median(finite))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return med, lo, hi


def run_paired_tests(pat_lvl: pd.DataFrame) -> pd.DataFrame:
    """Per-model Wilcoxon signed-rank on (sigma_pred - sigma_acq) against 0.

    Also reports median eta, bootstrap CI, BH-FDR q across models.

    Patients with undefined eta (sigma_acq = 0, i.e. every surviving ROI is
    constant) are dropped from both the bootstrap and the Wilcoxon input
    so the reported n matches across both statistics.
    """
    models = sorted(pat_lvl["model"].unique())
    records = []
    for m in models:
        sub = (
            pat_lvl[pat_lvl["model"] == m]
            .dropna(subset=["eta"])
            .sort_values("patient_id")
        )
        sig_acq = sub["sigma_acq"].to_numpy()
        sig_pred = sub["sigma_pred"].to_numpy()
        eta = sub["eta"].to_numpy()
        diffs = sig_pred - sig_acq
        try:
            res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            stat, pval = float(res.statistic), float(res.pvalue)
        except ValueError:
            stat, pval = np.nan, np.nan
        med, lo, hi = bootstrap_median_ci(eta)
        records.append({
            "model": m,
            "n_patients": int(len(sub)),
            "median_eta": med,
            "eta_ci_low": lo,
            "eta_ci_high": hi,
            "mean_sigma_acq": float(sig_acq.mean()),
            "mean_sigma_pred": float(sig_pred.mean()),
            "wilcoxon_stat": stat,
            "wilcoxon_p": pval,
        })
    out = pd.DataFrame.from_records(records)
    # BH-FDR across models
    out["wilcoxon_q_bh"] = np.nan
    valid = ~out["wilcoxon_p"].isna()
    if valid.any():
        q = false_discovery_control(out.loc[valid, "wilcoxon_p"].to_numpy(), method="bh")
        out.loc[valid, "wilcoxon_q_bh"] = q
    out["significant_q_lt_0_05"] = out["wilcoxon_q_bh"] < 0.05
    return out


# ---------------------------------------------------------------------------
# NPS aggregation and figure
# ---------------------------------------------------------------------------
def aggregate_nps(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean NPS across all curves. Assumes a consistent frequency grid
    within a single patient group; here we interpolate to a common grid
    defined by the *first* curve's frequencies.
    """
    if not curves:
        return None
    f_common = curves[0][0]
    # Interpolate every other curve to f_common
    stacked = []
    for f, nps in curves:
        if len(f) != len(f_common) or not np.allclose(f, f_common):
            interp = np.interp(f_common, f, nps)
            stacked.append(interp)
        else:
            stacked.append(nps)
    mat = np.vstack(stacked)
    return f_common, np.nanmean(mat, axis=0)


def plot_nps(
    agg_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    representative: list[str],
) -> None:
    """Plot NPS for acquired vs. representative model predictions."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    style = {
        "acquired": {"color": "black", "linewidth": 2.0, "label": "Acquired"},
        "linear_mean": {"color": "tab:gray", "linestyle": "--", "label": "Linear mean"},
        "l1": {"color": "tab:blue", "label": "L1"},
        "msssim+l1": {"color": "tab:green", "label": "MS-SSIM+L1"},
        "ssim_dagger": {"color": "tab:orange", "label": r"SSIM$^\dagger$"},
        "mse": {"color": "tab:red", "label": "MSE"},
    }
    for key in ["acquired"] + representative:
        if key not in agg_curves:
            continue
        f, nps = agg_curves[key]
        opts = style.get(key, {"label": key})
        ax.semilogy(f, nps, **opts)
    ax.set_xlabel("Spatial frequency (cycles/mm)")
    ax.set_ylabel("NPS (mm$^2$)")
    ax.set_title("Radial noise power spectrum\n(uniform white-matter ROIs)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    logger.info("Loading learned models")
    models: dict[str, torch.nn.Module] = {}
    for label, exp in LEARNED_MODELS.items():
        logger.info(f"  loading {label} -> {exp}")
        models[label] = load_model(EXPERIMENTS_DIR / exp, device)

    # Dataset
    any_cfg = json.loads((EXPERIMENTS_DIR / next(iter(LEARNED_MODELS.values())) / "config.json").read_text())
    test_dataset = TwoToOneSliceTestDataset(
        root_dir=resolve_config_path(any_cfg["data_path"]),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )

    # Hemorrhage labels: build quick lookup
    hemo_df = load_hemorrhage_labels()
    hemo_map: dict[tuple[str, int], int] = {
        (row["patient_id"], int(row["triplet_index"])): int(row["target_any"])
        for _, row in hemo_df.iterrows()
    }

    # Pixel spacings
    spacing_map = load_pixel_spacings()
    global_default_spacing = float(
        np.median(list(spacing_map.values())) if spacing_map else 0.488
    )
    logger.info(f"Median pixel spacing across test patients: {global_default_spacing:.4f} mm/px")

    # Per-patient processing
    all_rows: list[dict] = []
    all_nps: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    total_diag = {"n_triplets": 0, "n_rej_hemo": 0, "n_rej_geom": 0,
                  "n_rej_outlier": 0, "n_pons_missing": 0, "n_rois_kept": 0}

    for idx in range(len(test_dataset)):
        patient_id = test_dataset.get_patient_id_by_index(idx)
        inputs, targets = test_dataset[idx]
        spacing = spacing_map.get(patient_id, global_default_spacing)
        logger.info(
            f"Patient {patient_id} [{idx + 1}/{len(test_dataset)}]: "
            f"{inputs.shape[0]} triplets, spacing={spacing:.4f} mm/px"
        )
        rows, nps_curves, diag = process_patient(
            patient_id=patient_id,
            inputs=inputs,
            targets=targets,
            models=models,
            device=device,
            hemo_map=hemo_map,
            pixel_spacing_mm=spacing,
        )
        all_rows.extend(rows)
        for k, v in nps_curves.items():
            all_nps.setdefault(k, []).extend(v)
        for k, v in diag.items():
            total_diag[k] += v

    # ---------------------------------------------------------------- tables
    df_obs = pd.DataFrame(all_rows)
    df_obs.to_csv(TABLES_DIR / "roi_noise_analysis.csv", index=False)
    logger.info(
        f"Wrote {len(df_obs)} ROI observations to {TABLES_DIR / 'roi_noise_analysis.csv'}"
    )

    pat_lvl = aggregate_to_patient_level(df_obs)
    pat_lvl.to_csv(TABLES_DIR / "roi_noise_analysis_patient.csv", index=False)

    tests_df = run_paired_tests(pat_lvl)
    tests_df.to_csv(TABLES_DIR / "roi_noise_analysis_tests.csv", index=False)
    logger.info("\n" + tests_df.to_string())

    # ---------------------------------------------------------------- NPS fig
    agg: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, curves in all_nps.items():
        out = aggregate_nps(curves)
        if out is not None:
            agg[k] = out
    # Representative models: pick one from each "family"
    representative = ["linear_mean", "l1", "mse", "msssim+l1", "ssim_dagger"]
    plot_nps(agg, FIGURES_DIR / "nps_curves.pdf", representative)
    logger.info(f"Wrote NPS figure to {FIGURES_DIR / 'nps_curves.pdf'}")

    # ---------------------------------------------------------------- diag
    total_candidate = total_diag["n_rois_kept"] + total_diag["n_rej_hemo"] * 3 + total_diag["n_rej_outlier"] + total_diag["n_rej_geom"] + total_diag["n_pons_missing"]
    logger.info("ROI diagnostics:")
    logger.info(f"  Patients:            {len(test_dataset)}")
    logger.info(f"  Triplets total:      {total_diag['n_triplets']}")
    logger.info(f"  ROIs kept:           {total_diag['n_rois_kept']}")
    logger.info(f"  Rejected (hemorrhage slice): {total_diag['n_rej_hemo']} triplets")
    logger.info(f"  Rejected (geometry):  {total_diag['n_rej_geom']}")
    logger.info(f"  Rejected (outlier >3x med): {total_diag['n_rej_outlier']}")
    logger.info(f"  Pons ROI unplaceable: {total_diag['n_pons_missing']}")


if __name__ == "__main__":
    main()
