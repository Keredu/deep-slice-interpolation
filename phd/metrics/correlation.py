"""Correlation metrics for CT slice interpolation: NCC.

Normalized Cross-Correlation measures linear correlation between images.
Higher values indicate better correlation.
"""

import torch


def ncc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Normalized Cross-Correlation (NCC) between prediction and target.

    NCC measures the linear correlation between two images, normalized by
    their standard deviations. It is invariant to linear intensity transformations.

    Formula: NCC = sum((x - mean_x) * (y - mean_y)) / (std_x * std_y * N)

    Higher values indicate better correlation:
    - 1.0 = perfect positive correlation (identical after normalization)
    - 0.0 = no correlation
    - -1.0 = perfect negative correlation (inverted)

    Args:
        pred: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)
        eps: Small constant for numerical stability (default: 1e-8)

    Returns:
        NCC value averaged over the batch, in range [-1, 1]
    """
    # Flatten spatial dimensions, keep batch and channel
    pred_flat = pred.flatten(start_dim=2)  # (B, C, H*W)
    target_flat = target.flatten(start_dim=2)

    # Compute means
    pred_mean = pred_flat.mean(dim=2, keepdim=True)
    target_mean = target_flat.mean(dim=2, keepdim=True)

    # Center the data
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    # Compute standard deviations
    pred_std = pred_centered.std(dim=2, keepdim=True) + eps
    target_std = target_centered.std(dim=2, keepdim=True) + eps

    # Compute NCC for each sample
    n_pixels = pred_flat.shape[2]
    ncc_values = (pred_centered * target_centered).sum(dim=2) / (pred_std.squeeze() * target_std.squeeze() * n_pixels)

    # Average over batch and channel
    return ncc_values.mean().item()
