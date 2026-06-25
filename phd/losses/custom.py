"""Custom loss module with config-based initialization.

This module provides a unified interface for the loss functions used
in CT slice interpolation: pixel-wise losses (MSE, L1), structural
losses (SSIM, MS-SSIM), and their pairwise combinations with L1.
"""

from typing import ClassVar

import torch
from torch import nn

from .combined import MSSSIMPlusL1Loss


class CombinedSSIML1Loss(nn.Module):
    """Combined SSIM and L1 loss for better structural and pixel-level accuracy.

    Combines:
    - SSIM: Structural similarity (perceptual quality)
    - L1: Pixel-level accuracy (reduces artifacts)

    Recommended weights: 0.8 SSIM + 0.2 L1
    """

    def __init__(
        self,
        ssim_weight: float = 0.8,
        l1_weight: float = 0.2,
        data_range: float = 1.0,
        size_average: bool = True,
        channel: int = 1,
        K: tuple[float, float] = (0.01, 0.03),  # noqa: N803
        nonnegative_ssim: bool = False,
    ) -> None:
        """Initialize combined loss.

        Args:
            ssim_weight: Weight for SSIM component (default: 0.8)
            l1_weight: Weight for L1 component (default: 0.2)
            data_range: Data range for SSIM (default: 1.0)
            size_average: Whether to average SSIM (default: True)
            channel: Number of channels for SSIM (default: 1)
            K: SSIM stability constants (K1, K2). Use K2=0.4 for numerical stability.
            nonnegative_ssim: If True, clamp SSIM to [0, 1] (prevents negative values).
        """
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight

        try:
            from pytorch_msssim import SSIM

            self.ssim = SSIM(
                data_range=data_range,
                size_average=size_average,
                channel=channel,
                K=K,
                nonnegative_ssim=nonnegative_ssim,
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
        # SSIM requires fp32 - fp16 from AMP causes NaN in variance/covariance computations
        with torch.amp.autocast("cuda", enabled=False):
            ssim_loss = torch.clamp(1 - self.ssim(pred.float(), target.float()), 0.0, 2.0)
        l1_loss = self.l1(pred, target)

        return self.ssim_weight * ssim_loss + self.l1_weight * l1_loss


class CustomLoss(nn.Module):
    """Custom loss module with config-based initialization.

    Supports the following loss functions for CT slice interpolation:

    Basic losses:
    - "mse": Mean Squared Error
    - "l1": Mean Absolute Error
    - "ssim": Structural Similarity Index
    - "msssim": Multi-Scale SSIM

    Combined losses (weights required in params):
    - "ssim+l1": SSIM + L1 (requires ssim_weight, l1_weight)
    - "msssim+l1": MS-SSIM + L1 (requires msssim_weight, l1_weight)

    For SSIM-based losses, the similarity score is converted to loss (1 - SSIM).
    """

    # Loss names that return similarity scores (need to convert to loss)
    _SIMILARITY_LOSSES: ClassVar[set[str]] = {"ssim", "msssim"}

    # Loss names that already return loss values
    _DIRECT_LOSSES: ClassVar[set[str]] = {
        "mse",
        "l1",
        "ssim+l1",
        "msssim+l1",
    }

    def __init__(self, loss_config: dict) -> None:
        """Initialize the loss function based on configuration.

        Args:
            loss_config: Dictionary containing loss configuration with
                        'name' and optional 'params' keys.

        Example configs:
            {"name": "mse"}
            {"name": "ssim", "params": {"data_range": 1.0, "channel": 1}}
            {"name": "ssim+l1", "params": {"ssim_weight": 0.8, "l1_weight": 0.2}}

        Raises:
            ValueError: If the loss function is not supported
            ImportError: If required dependencies are not installed
        """
        super().__init__()
        self.loss_name = loss_config["name"].lower()
        params = loss_config.get("params", {})

        self.criterion = self._create_criterion(self.loss_name, params)

    def _create_criterion(self, loss_name: str, params: dict) -> nn.Module:
        """Create the loss criterion based on name and parameters.

        Args:
            loss_name: Name of the loss function
            params: Parameters for the loss function

        Returns:
            Initialized loss module

        Raises:
            ValueError: If loss_name is not supported
        """
        factory_methods = {
            "mse": lambda p: nn.MSELoss(),
            "l1": lambda p: nn.L1Loss(),
            "ssim": self._create_ssim,
            "msssim": self._create_msssim,
            "ssim+l1": self._create_ssim_l1,
            "msssim+l1": self._create_msssim_l1,
        }

        if loss_name in factory_methods:
            return factory_methods[loss_name](params)

        msg = f"Unsupported loss function: {loss_name}"
        raise ValueError(msg)

    def _create_ssim_l1(self, params: dict) -> nn.Module:
        """Create SSIM+L1 combined loss."""
        if "ssim_weight" not in params or "l1_weight" not in params:
            raise ValueError("ssim+l1 loss requires 'ssim_weight' and 'l1_weight' in params")
        return CombinedSSIML1Loss(
            ssim_weight=params["ssim_weight"],
            l1_weight=params["l1_weight"],
            data_range=params.get("data_range", 1.0),
            size_average=params.get("size_average", True),
            channel=params.get("channel", 1),
            K=tuple(params.get("K", (0.01, 0.03))),
            nonnegative_ssim=params.get("nonnegative_ssim", False),
        )

    def _create_msssim_l1(self, params: dict) -> nn.Module:
        """Create MS-SSIM+L1 combined loss."""
        if "msssim_weight" not in params or "l1_weight" not in params:
            raise ValueError("msssim+l1 loss requires 'msssim_weight' and 'l1_weight' in params")
        return MSSSIMPlusL1Loss(
            msssim_weight=params["msssim_weight"],
            l1_weight=params["l1_weight"],
            data_range=params.get("data_range", 1.0),
            channel=params.get("channel", 1),
            K=tuple(params.get("K", (0.01, 0.03))),
        )

    def _create_ssim(self, params: dict) -> nn.Module:
        """Create SSIM loss module.

        Args:
            params: Parameters for SSIM

        Returns:
            SSIM module
        """
        try:
            from pytorch_msssim import SSIM

            return SSIM(
                data_range=params.get("data_range", 1.0),
                size_average=params.get("size_average", True),
                channel=params.get("channel", 1),
                K=tuple(params.get("K", (0.01, 0.03))),
                nonnegative_ssim=params.get("nonnegative_ssim", False),
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError("Please install pytorch-msssim: pip install pytorch-msssim") from e

    def _create_msssim(self, params: dict) -> nn.Module:
        """Create MS-SSIM loss module.

        Args:
            params: Parameters for MS-SSIM

        Returns:
            MS-SSIM module
        """
        try:
            from pytorch_msssim import MS_SSIM

            return MS_SSIM(
                data_range=params.get("data_range", 1.0),
                size_average=params.get("size_average", True),
                channel=params.get("channel", 1),
                K=tuple(params.get("K", (0.01, 0.03))),
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError("Please install pytorch-msssim: pip install pytorch-msssim") from e

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between prediction and target.

        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)

        Returns:
            torch.Tensor: Calculated loss value
        """
        if self.loss_name in self._SIMILARITY_LOSSES:
            # SSIM/MS-SSIM require fp32 - fp16 from AMP causes NaN in variance computations
            with torch.amp.autocast("cuda", enabled=False):
                return torch.clamp(1 - self.criterion(pred.float(), target.float()), 0.0, 2.0)

        # All other losses already return loss values
        return self.criterion(pred, target)
