"""Unified interface to compute all metrics for CT slice interpolation."""

import torch

from phd.metrics.correlation import ncc
from phd.metrics.error import gradient_mae, mae
from phd.metrics.quality import ms_ssim, psnr, ssim


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute all metrics at once for CT slice interpolation evaluation.

    This function computes a comprehensive set of metrics comparing
    predicted and target CT slices:

    Quality metrics (higher is better):
    - ssim: Structural Similarity Index [0, 1]
    - ms_ssim: Multi-Scale SSIM [0, 1]
    - psnr: Peak Signal-to-Noise Ratio in dB

    Error metrics (lower is better):
    - mae: Mean Absolute Error
    - gradient_mae: Gradient Mean Absolute Error (edge preservation)

    Correlation metrics (higher is better):
    - ncc: Normalized Cross-Correlation [-1, 1]

    Args:
        pred: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)
        data_range: The range of the input data (default: 1.0 for normalized images)

    Returns:
        Dictionary with metric names as keys and computed values as floats.

    Example:
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> target = torch.rand(4, 1, 256, 256)
        >>> metrics = compute_all_metrics(pred, target)
        >>> print(f"SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f} dB")
    """
    channel = pred.shape[1]

    return {
        # Quality metrics (higher is better)
        "ssim": ssim(pred, target, data_range=data_range, channel=channel),
        "ms_ssim": ms_ssim(pred, target, data_range=data_range, channel=channel),
        "psnr": psnr(pred, target, data_range=data_range),
        # Error metrics (lower is better)
        "mae": mae(pred, target),
        "gradient_mae": gradient_mae(pred, target),
        # Correlation metrics (higher is better)
        "ncc": ncc(pred, target),
    }
