"""Metrics module for evaluating CT slice interpolation quality.

This module provides various metrics for comparing predicted and target CT slices:
- Quality metrics: SSIM, MS-SSIM, PSNR
- Error metrics: MAE, Gradient MAE
- Correlation metrics: NCC

All metrics work on batched tensors (B, C, H, W) and return scalar values.
"""

from phd.metrics.compute import compute_all_metrics
from phd.metrics.correlation import ncc
from phd.metrics.error import gradient_mae, mae
from phd.metrics.quality import ms_ssim, psnr, ssim

__all__ = [
    "compute_all_metrics",
    "gradient_mae",
    "mae",
    "ms_ssim",
    "ncc",
    "psnr",
    "ssim",
]
