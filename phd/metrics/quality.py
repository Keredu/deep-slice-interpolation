"""Quality metrics for CT slice interpolation: SSIM, MS-SSIM, and PSNR.

These metrics measure the structural and signal quality of predicted images.
Higher values indicate better quality for all metrics in this module.
"""

import torch
from pytorch_msssim import MS_SSIM, SSIM


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    channel: int = 1,
) -> float:
    """Compute Structural Similarity Index (SSIM) between prediction and target.

    SSIM measures perceptual quality based on luminance, contrast, and structure.
    Higher values indicate better similarity (1.0 = identical).

    Args:
        pred: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)
        data_range: The range of the input data (default: 1.0 for normalized images)
        channel: Number of channels in the input (default: 1 for grayscale CT)

    Returns:
        SSIM value averaged over the batch, in range [0, 1]
    """
    ssim_module = SSIM(
        data_range=data_range,
        size_average=True,
        channel=channel,
    )
    ssim_module = ssim_module.to(pred.device)
    return ssim_module(pred, target).item()


def ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    channel: int = 1,
) -> float:
    """Compute Multi-Scale Structural Similarity Index (MS-SSIM).

    MS-SSIM extends SSIM by computing similarity at multiple scales,
    providing better correlation with human perception of image quality.
    Higher values indicate better similarity (1.0 = identical).

    Note: Input images must be at least 160x160 pixels for the default
    5-scale MS-SSIM computation.

    Args:
        pred: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)
        data_range: The range of the input data (default: 1.0 for normalized images)
        channel: Number of channels in the input (default: 1 for grayscale CT)

    Returns:
        MS-SSIM value averaged over the batch, in range [0, 1]
    """
    ms_ssim_module = MS_SSIM(
        data_range=data_range,
        size_average=True,
        channel=channel,
    )
    ms_ssim_module = ms_ssim_module.to(pred.device)
    return ms_ssim_module(pred, target).item()


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels.

    PSNR measures the ratio between the maximum possible signal power
    and the power of corrupting noise. Higher values indicate better quality.
    Typical values for good quality images are 30-50 dB.

    Formula: PSNR = 10 * log10(MAX^2 / MSE)

    Args:
        pred: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)
        data_range: The range of the input data (default: 1.0 for normalized images)

    Returns:
        PSNR value in decibels (dB). Returns float('inf') if images are identical.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(data_range**2 / mse)).item()
