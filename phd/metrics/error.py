"""Error metrics for CT slice interpolation: MAE and Gradient MAE.

These metrics measure pixel-level and edge-level errors between images.
Lower values indicate better quality for all metrics in this module.
"""

import torch
from torch.nn import functional


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error (MAE) between prediction and target.

    MAE measures the average absolute difference between pixels.
    Lower values indicate better reconstruction (0 = identical).

    Args:
        pred: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)

    Returns:
        MAE value averaged over all pixels and batch.
    """
    return torch.mean(torch.abs(pred - target)).item()


def _sobel_filter(x: torch.Tensor) -> torch.Tensor:
    """Apply Sobel filter to compute gradient magnitude.

    Computes both horizontal and vertical gradients using Sobel operators
    and returns the gradient magnitude.

    Args:
        x: Input tensor of shape (B, C, H, W)

    Returns:
        Gradient magnitude tensor of shape (B, C, H-2, W-2)
    """
    # Sobel kernels for gradient computation
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    # Process each channel separately
    _batch_size, channels, _, _ = x.shape
    gradients = []

    for c in range(channels):
        x_c = x[:, c : c + 1, :, :]
        grad_x = functional.conv2d(x_c, sobel_x, padding=0)
        grad_y = functional.conv2d(x_c, sobel_y, padding=0)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        gradients.append(grad_magnitude)

    return torch.cat(gradients, dim=1)


def gradient_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Gradient Mean Absolute Error between prediction and target.

    This metric applies Sobel filters to both images and computes MAE
    on the gradient magnitudes. It measures how well edges are preserved
    in the reconstruction.

    Lower values indicate better edge preservation (0 = identical edges).

    Args:
        pred: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)

    Returns:
        Gradient MAE value averaged over all pixels and batch.
    """
    pred_grad = _sobel_filter(pred)
    target_grad = _sobel_filter(target)
    return torch.mean(torch.abs(pred_grad - target_grad)).item()
