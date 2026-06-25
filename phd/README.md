# phd Package

Core Python package for CT slice interpolation research.

## Installation

The package is installed automatically with `uv sync` from the project root.

```bash
# From project root
uv sync

# Import in code
from phd.training import Trainer, create_config
from phd.losses import CustomLoss
from phd.metrics import compute_all_metrics
```

## Module Overview

### datasets/

Dataset implementations for loading CT slice triplets.

- **TwoToOneSliceDataset**: Base dataset loading triplets at 512x512
- **TwoToOneSliceCroppedDataset**: Augmented dataset with 9-crop strategy at 256x256
- **TwoToOneSliceTestDataset**: Test-time dataset with dual modes

### models/

Neural network architectures for slice interpolation.

- **InterpolationUNet**: U-Net with pretrained encoder

### losses/

Loss functions for training.

- **Basic**: MSE, L1, SSIM, MS-SSIM
- **Combined**: SSIM+L1, MS-SSIM+L1
- **Advanced**: Gradient (edge), Frequency (FFT), Perceptual (VGG)

### metrics/

Evaluation metrics for quality assessment.

- **Quality**: SSIM, MS-SSIM, PSNR
- **Error**: MAE, Gradient MAE
- **Perceptual**: LPIPS
- **Correlation**: NCC

### training/

Training infrastructure.

- **Trainer**: Main training loop with checkpointing and early stopping
- **Configuration**: Default config and config creation
- **Registry**: Experiment tracking and queue management

## Quick Start

```python
from phd.training import Trainer, create_config

# Create config
config = create_config(
    learning_rate=3e-4,
    batch_size=32,
    loss={"name": "ssim+l1", "params": {"ssim_weight": 0.8, "l1_weight": 0.2}}
)

# Train
trainer = Trainer(config)
trainer.run()
```

## Utilities

### config_io.py

Save experiment configurations as JSON.

```python
from phd.config_io import save_config

save_config(config, "experiment/config.json")
```

### plotting.py

Generate training visualizations.

```python
from phd.plotting import save_loss_plot, save_metric_plots, save_metrics_csv

save_loss_plot(train_losses, valid_losses, output_path)
save_metric_plots(metric_histories, output_dir)
save_metrics_csv(metric_histories, train_losses, valid_losses, output_path)
```

### viz.py

Test-time visualization with crop-based reconstruction and corner inferences.

#### Test Reconstruction Pipeline

For 512×512 test images, the model operates on 256×256 crops:

```
512×512 input → 9 center crops + 4 corner patches → inference → composite 512×512
```

**Step-by-step:**
1. `extract_crops()`: Extract 9 overlapping 256×256 crops from center 384×384 region
2. `extract_corner_patches()`: Extract 4 corner 256×256 patches for outer border
3. Run model inference on all 13 patches
4. `reconstruct_from_crops()`: Place center predictions, average overlaps → 384×384 center
5. `combine_corner_predictions()`: Combine 4 corner predictions → full 512×512 border
6. Create composite: corners as base + center 384×384 overlaid

#### Output Visualization (3×2 Grid)

```
+------------------+------------------+------------------+
| First slice      | Target slice     | Third slice      |
| (input ch0)      | (ground truth)   | (input ch1)      |
+------------------+------------------+------------------+
| Corner           | Composite        | 384×384 with     |
| inferences       | (merged output)  | gray padding     |
+------------------+------------------+------------------+
```

- **Position 4**: 4 corner 256×256 inferences combined into 512×512
- **Position 5**: Composite (corners + center merged) - final output without black borders
- **Position 6**: Center 384×384 reconstruction with gray 64px padding (for comparison)

#### Functions

```python
from phd.viz import (
    extract_crops,
    extract_corner_patches,
    combine_corner_predictions,
    save_test_visualization,
)

# Extract 9 center crops from 512×512 image
crops = extract_crops(tensor)  # (C, 512, 512) → (9, C, 256, 256)

# Extract 4 corner patches from 512×512 image
corners = extract_corner_patches(tensor)  # (C, 512, 512) → (4, C, 256, 256)

# Combine corner predictions into full 512×512
combined = combine_corner_predictions(corners)  # (4, C, 256, 256) → (C, 512, 512)

# Full test visualization (used by Trainer on best epochs)
save_test_visualization(
    test_dataset=test_dataset,  # TwoToOneSliceTestDataset
    model=model,
    device=device,
    save_dir=viz_dir,
    batch_size=32,
)
```

#### Reconstruction

```python
from phd.datasets.interpolation.two_to_one_slice_cropped import reconstruct_from_crops

# Reconstruct 512×512 from 9 crop predictions
output = reconstruct_from_crops(crops)  # (9, C, 256, 256) → (C, 512, 512)
# Center 384×384: averaged predictions
# Outer 64px: black (zeros)
```
