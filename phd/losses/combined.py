"""Combined loss functions for CT slice interpolation.

This module implements combinations of loss functions that target
different aspects of image quality: pixel-level accuracy and structural
similarity.
"""

import torch
from torch import nn


class MSSSIMPlusL1Loss(nn.Module):
    """Combined Multi-Scale SSIM and L1 loss.

    Combines:
    - MS-SSIM: Multi-scale structural similarity (captures structure at multiple scales)
    - L1: Pixel-level accuracy (reduces artifacts)

    MS-SSIM is better than single-scale SSIM for capturing both local
    and global structural information.
    """

    def __init__(
        self,
        msssim_weight: float = 0.8,
        l1_weight: float = 0.2,
        data_range: float = 1.0,
        channel: int = 1,
        K: tuple[float, float] = (0.01, 0.03),  # noqa: N803
    ) -> None:
        """Initialize combined loss.

        Args:
            msssim_weight: Weight for MS-SSIM component (default: 0.8)
            l1_weight: Weight for L1 component (default: 0.2)
            data_range: Data range for MS-SSIM (default: 1.0)
            channel: Number of channels for MS-SSIM (default: 1)
            K: SSIM stability constants (K1, K2). Use K2=0.4 for numerical stability.
        """
        super().__init__()
        self.msssim_weight = msssim_weight
        self.l1_weight = l1_weight

        try:
            from pytorch_msssim import MS_SSIM

            self.msssim = MS_SSIM(
                data_range=data_range,
                size_average=True,
                channel=channel,
                K=K,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError("Please install pytorch-msssim: pip install pytorch-msssim") from e

        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Combined loss value
        """
        # MS-SSIM requires fp32 - fp16 from AMP causes NaN in variance/covariance computations
        with torch.amp.autocast("cuda", enabled=False):
            msssim_loss = torch.clamp(1 - self.msssim(pred.float(), target.float()), 0.0, 2.0)
        l1_loss = self.l1(pred, target)

        return self.msssim_weight * msssim_loss + self.l1_weight * l1_loss
