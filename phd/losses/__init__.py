"""Loss functions for CT slice interpolation.

This module provides loss functions for training deep learning models
on CT slice interpolation tasks:

- Basic losses: MSE, L1, SSIM, MS-SSIM
- Combined losses: SSIM+L1, MS-SSIM+L1

Usage:
    from phd.losses import CustomLoss

    # Create loss from config
    loss_fn = CustomLoss({"name": "msssim+l1", "params": {"msssim_weight": 0.5, "l1_weight": 0.5}})

    # Use in training
    loss = loss_fn(predictions, targets)
"""

from .combined import MSSSIMPlusL1Loss
from .custom import CombinedSSIML1Loss, CustomLoss

__all__ = [
    "CombinedSSIML1Loss",
    "CustomLoss",
    "MSSSIMPlusL1Loss",
]
