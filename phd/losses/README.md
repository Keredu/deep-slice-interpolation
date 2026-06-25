# Loss Functions

Loss function library for CT slice interpolation.

## Usage

### Using CustomLoss (Recommended)

The `CustomLoss` class provides a unified interface for all loss functions:

```python
from phd.losses import CustomLoss

# Simple loss
loss_fn = CustomLoss({"name": "ssim"})

# Combined loss with custom weights
loss_fn = CustomLoss({
    "name": "ssim+l1",
    "params": {
        "ssim_weight": 0.8,
        "l1_weight": 0.2
    }
})

# Compute loss
loss = loss_fn(predictions, targets)
loss.backward()
```

### Direct Import

```python
from phd.losses import CombinedSSIML1Loss, MSSSIMPlusL1Loss
```

## Available Loss Functions

### Basic Losses

| Name | Description | Use Case |
|------|-------------|----------|
| `mse` | Mean Squared Error | Baseline, fast |
| `l1` | Mean Absolute Error | Less sensitive to outliers |
| `ssim` | Structural Similarity | Perceptual quality |
| `msssim` | Multi-Scale SSIM | Scale-invariant quality |

### Combined Losses

| Name | Formula | Use Case |
|------|---------|----------|
| `ssim+l1` | α·SSIM + (1−α)·L1 | Balanced quality + accuracy |
| `msssim+l1` | α·MS-SSIM + (1−α)·L1 | Multi-scale + accuracy |

## Parameters

### CombinedSSIML1Loss

```python
CombinedSSIML1Loss(
    ssim_weight=0.8,      # Weight for SSIM component
    l1_weight=0.2,        # Weight for L1 component
    data_range=1.0,       # Input data range
    channel=1,            # Number of channels
    K=(0.01, 0.03),       # SSIM stability constants; use K=(0.01, 0.4) for training
    nonnegative_ssim=False,
)
```

### MSSSIMPlusL1Loss

```python
MSSSIMPlusL1Loss(
    msssim_weight=0.8,
    l1_weight=0.2,
    data_range=1.0,
    channel=1,
    K=(0.01, 0.03),
)
```

## Input Requirements

- **Format**: `(B, C, H, W)` PyTorch tensors
- **Range**: `[0, 1]` normalized
- **Minimum size**: Varies by loss (MS-SSIM needs 160x160)

## Adding New Losses

1. Create loss class in appropriate file:

```python
class MyLoss(nn.Module):
    def __init__(self, param1=1.0):
        super().__init__()
        self.param1 = param1

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute loss
        return loss  # Must be scalar tensor
```

2. Register in `custom.py`:

```python
# Add to appropriate set
_DIRECT_LOSSES = {..., "myloss"}

# Add creation method
def _create_myloss(self) -> nn.Module:
    return MyLoss(**self.params)
```

3. Add tests in `tests/losses/test_losses.py`
4. Update documentation
