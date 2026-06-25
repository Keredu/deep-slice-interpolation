#!/usr/bin/env python3
r"""Measure the distribution of 11x11 Gaussian-weighted local variances on
brain-windowed CT training slices (paper S3.4-07, S3.4-17).

Context. Section 3.4 of the paper claims C_2 = 0.16 sits "comfortably above
the variance of a typical low-texture CT window" (intervention 1) and that
the Wang et al. default C_2 = 9e-4 falls "within an order of magnitude of
the local intensity variance of homogeneous parenchymal regions". Both are
numerical claims without a measured anchor. This script supplies one.

Procedure.
  1. Sample N slice IDs uniformly at random (seed fixed) from the training
     split of df.csv.
  2. Load each PNG (already pre-windowed to [-20, 107] HU and rescaled to
     [0, 1] on disk).
  3. Compute the Gaussian-weighted local variance map at every pixel using
     an 11x11 kernel with sigma = 1.5 (identical to the kernel used inside
     pytorch_msssim.SSIM). The operator is separable (two 1D convolutions)
     and identities sigma^2 = E[X^2] - (E[X])^2 with the SAME kernel for
     both moments.
  4. Flatten all per-pixel variance values across all sampled slices and
     report: quantiles (5/25/50/75/95/99), fraction of windows with
     sigma^2 <= C_2 (default) = 9e-4, and fraction with sigma^2 <= C_2
     (training) = 0.16.
  5. Also report the same statistics restricted to a "parenchymal" subset
     (mean intensity > 0.15 within the window, i.e. above the background-
     air band) to address S3.4-17 directly.

Outputs.
  results/tables/window_variance_summary.csv   (one row, all stats)

Reproduce: uv run scripts/window_variance_measurement.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

load_dotenv()

N_SLICES = 5000
SEED = 42
WINDOW_SIZE = 11
SIGMA = 1.5
C2_DEFAULT = 9e-4     # Wang et al. K_2 = 0.03  => C_2 = (0.03)^2
C2_TRAINING = 0.16    # Our training choice K_2 = 0.4 => C_2 = (0.4)^2
PARENCHYMA_THRESHOLD = 0.15  # Local mean > 0.15 excludes background air band

TABLES_DIR = Path("results/tables")


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _local_mean_and_variance(img: torch.Tensor, kernel_1d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply separable Gaussian-weighted local mean and variance.

    img: (H, W) float tensor on [0, 1].
    Returns (mean_map, var_map) same shape as img; border pixels are computed
    with 'valid' padding logic so only interior windows are returned.
    """
    k = kernel_1d.view(1, 1, -1)
    # Pad='valid': crop borders where the kernel doesn't fully fit.
    x = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Horizontal pass then vertical; each pass reduces the relevant dim.
    mean_h = F.conv2d(x, k.view(1, 1, 1, -1))
    mean = F.conv2d(mean_h, k.view(1, 1, -1, 1)).squeeze()
    sq_h = F.conv2d(x**2, k.view(1, 1, 1, -1))
    sq = F.conv2d(sq_h, k.view(1, 1, -1, 1)).squeeze()
    var = (sq - mean**2).clamp_min(0.0)
    return mean, var


def _sample_slice_ids(df_train: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    ids = df_train["SOPInstanceUID"].dropna().unique()
    idx = rng.choice(len(ids), size=n, replace=False)
    return ids[idx].tolist()


def main() -> None:
    rng = np.random.default_rng(SEED)
    datasets_dir = Path(os.environ["DATASETS_DIR"])
    pre_dir = datasets_dir / "pre/rsna-intracranial-hemorrhage-detection"
    img_dir = pre_dir / "1x512x512_-20_107"

    df = pd.read_csv(pre_dir / "df.csv").rename(columns={"split": "stage"})
    df_train = df[df["stage"] == "train"]
    logger.info(f"Training-split slices available: {len(df_train):,}")

    ids = _sample_slice_ids(df_train, n=N_SLICES, rng=rng)

    kernel = _gaussian_kernel_1d(WINDOW_SIZE, SIGMA)

    all_var: list[np.ndarray] = []
    parenchyma_var: list[np.ndarray] = []

    for i, sop_id in enumerate(ids):
        path = img_dir / f"{sop_id}.png"
        if not path.exists():
            continue
        with Image.open(path) as im:
            arr = np.asarray(im, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr)
        mean_map, var_map = _local_mean_and_variance(img, kernel)
        v = var_map.numpy().ravel()
        m = mean_map.numpy().ravel()
        all_var.append(v)
        parenchyma_var.append(v[m > PARENCHYMA_THRESHOLD])
        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{N_SLICES} slices processed")

    full = np.concatenate(all_var)
    paren = np.concatenate(parenchyma_var)
    logger.info(f"Window variance samples: full={full.size:,}, parenchymal={paren.size:,}")

    def _row(name: str, v: np.ndarray) -> dict:
        return {
            "subset": name,
            "n_windows": int(v.size),
            "p5": float(np.quantile(v, 0.05)),
            "p25": float(np.quantile(v, 0.25)),
            "p50": float(np.quantile(v, 0.50)),
            "p75": float(np.quantile(v, 0.75)),
            "p95": float(np.quantile(v, 0.95)),
            "p99": float(np.quantile(v, 0.99)),
            "max": float(v.max()),
            "frac_leq_C2_default_9e-4": float((v <= C2_DEFAULT).mean()),
            "frac_leq_C2_training_0.16": float((v <= C2_TRAINING).mean()),
        }

    rows = [_row("full_slice", full), _row("parenchyma_mean>0.15", paren)]
    summary = pd.DataFrame(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "window_variance_summary.csv"
    summary.to_csv(out, index=False)
    logger.info(f"Wrote {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
