# Training Infrastructure

Modular training system for CT slice interpolation experiments.

## Overview

The training module provides a complete infrastructure for:
- Experiment configuration and management
- Training loop with validation
- Checkpointing and resume capability
- Early stopping
- Experiment registry and status tracking

## Quick Start

```python
from phd.training import Trainer, create_config

# Create configuration - all experiment params must be provided
config = create_config(
    exp_name="my_experiment",
    model={"type": "unet", "encoder_name": "tu-tf_efficientnetv2_s", "pretrained": True},
    loss={"name": "ssim", "params": {"data_range": 1.0, "channel": 1}},
    scheduler={
        "name": "cosine_warmup",
        "params": {"warmup_epochs": 5, "eta_min": 1e-6},
    },
    optimizer={"name": "adamw", "params": {"lr": 8e-4, "weight_decay": 1e-2}},
    batch_size=96,
    num_epochs=500,  # Ceiling; early stopping should trigger first
    early_stopping_patience=15,
)

# Train
trainer = Trainer(config)
trainer.run()
```

## Configuration

### Required Parameters

All experiment configuration must be provided explicitly:

```python
config = create_config(
    # Required - experiment identification
    exp_name="experiment_name",

    # Required - model architecture
    model={
        "type": "unet",
        "encoder_name": "tu-tf_efficientnetv2_s",
        "pretrained": True,
    },

    # Required - loss function
    loss={
        "name": "ssim+l1",
        "params": {"ssim_weight": 0.8, "l1_weight": 0.2, "data_range": 1.0, "channel": 1},
    },

    # Required - learning rate scheduler (all params explicit, no defaults)
    scheduler={
        "name": "cosine_warmup",
        "params": {"warmup_epochs": 5, "eta_min": 1e-6},
    },

    # Required - optimizer
    optimizer={"name": "adamw", "params": {"lr": 8e-4, "weight_decay": 1e-2}},

    # Required - training params
    batch_size=96,
    num_epochs=500,

    # Optional - early stopping (default: 7, recommended: 15 with cosine_warmup)
    early_stopping_patience=15,

    # Optional - dataset sizes (None = full dataset)
    train_size=None,
    valid_size=None,
)
```

### Infrastructure Defaults

These have sensible defaults but can be overridden:

```python
config = create_config(
    ...,
    # Infrastructure defaults
    crop_size=256,                      # Augmentation crop size
    flip_prob=0.5,                      # Augmentation flip probability
    num_workers=4,                      # DataLoader workers
    early_stopping_patience=7,          # Early stopping patience
    early_stopping_delta=0.0001,        # Early stopping min delta
)
```

### Augmentation Configuration

Control training augmentation behavior:

```python
config = create_config(
    ...,
    # Crop weights (9 values, one per position)
    crop_weights=(1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0),  # center=2x

    # Resize options
    include_resize_512=True,            # Include 512→256 resize
    resize_512_weight=0.5,              # Weight for 512→256 option
    include_resize_384=False,           # Include 384→256 resize
    resize_384_weight=0.5,              # Weight for 384→256 option

    # Rotation
    rotation_prob=0.5,                  # Probability of rotation
    rotation_max_angle=15.0,            # Max rotation in degrees (uniform [-max, +max])
)
```

### Test Visualization Configuration

Control which test visualizations are generated:

```python
config = create_config(
    ...,
    generate_test_viz_real=True,         # Visualize target_is_real mode
    generate_test_viz_interpolated=False, # Visualize target_is_interpolated mode
)
```

### Initialize From Another Experiment

Start a brand-new experiment from weights saved by another experiment:

```python
config = create_config(
    ...,
    init_from_experiment="msssim+l1_lr8e-4_e6d845",
    init_from_checkpoint="latest_epoch.pth",  # default
)
```

Notes:
- This initializes **model weights only**. Optimizer/scheduler/early-stopping state starts fresh.
- `init_from_checkpoint` can also point to a best checkpoint, for example: `"epochs/17/weights.pth"`.
- If the new experiment is resumed later, it resumes from its own `latest_epoch.pth` as usual.

### DataLoader Configuration

The training DataLoader uses `shuffle=True` (required). The dataset uses deterministic index expansion where each base triplet maps to 9 consecutive indices (one per crop position). Without shuffle, all 9 crops from the same triplet would end up in the same batch. With shuffle, crops are randomly mixed across different triplets.

### Schedulers

All scheduler parameters must be explicitly configured (no defaults). Available schedulers:

#### cosine
Standard cosine annealing. Decays LR smoothly from initial to `eta_min` over all epochs.
```python
"scheduler": {"name": "cosine", "params": {"eta_min": 1e-6}}
```
**Note**: With many epochs (e.g., 500), LR barely changes early on. Not recommended.

#### cosine_warmup (Recommended)
Linear warmup followed by cosine decay. Best for pretrained encoders.
```python
"scheduler": {
    "name": "cosine_warmup",
    "params": {"warmup_epochs": 5, "eta_min": 1e-6}
}
```
**Rationale**: Warmup prevents destabilizing pretrained encoder features during early training.
The randomly initialized decoder adapts first, then full learning rate kicks in.
- `warmup_epochs=5`: Sufficient for pretrained encoders (25 batches/epoch × 5 = 125 gradient updates)
- `eta_min=1e-6`: Low floor for fine-grained convergence
- Pair with `early_stopping_patience=15` to allow sufficient exploration before stopping

#### cosine_restarts
Periodic warm restarts to escape local minima. LR resets to initial value periodically.
```python
"scheduler": {
    "name": "cosine_restarts",
    "params": {
        "T_0": 20,      # Epochs until first restart
        "T_mult": 2,    # Multiply period after each restart
        "eta_min": 1e-6
    }
}
```
With `T_0=20, T_mult=2`: restarts at epochs 20, 60, 140, 300, ...
**Note**: If using with early stopping, set `patience > T_0` to let restarts work.

#### onecycle
One cycle: warmup to max_lr, then decay. Steps per batch (not per epoch).
```python
"scheduler": {
    "name": "onecycle",
    "params": {
        "max_lr": 0.001,
        "pct_start": 0.3,        # Fraction spent warming up
        "anneal_strategy": "cos" # or "linear"
    }
}
```

#### plateau
Reduce LR when validation loss plateaus. Reactive but may get stuck in local minima.
```python
"scheduler": {
    "name": "plateau",
    "params": {
        "mode": "min",
        "factor": 0.5,    # Multiply LR by this on plateau
        "patience": 10,   # Epochs to wait before reducing
        "min_lr": 1e-7
    }
}
```

#### step
Reduce LR by `gamma` every `step_size` epochs.
```python
"scheduler": {"name": "step", "params": {"step_size": 30, "gamma": 0.1}}
```

#### none
Constant learning rate (no scheduling).
```python
"scheduler": {"name": "none", "params": {}}
```

## Defining Experiments

Experiments are defined in `register_experiments.py`:

```python
BASE_CONFIG = {
    "model": {"type": "unet", "encoder_name": "tu-tf_efficientnetv2_s", "pretrained": True},
    "scheduler": {
        "name": "cosine_warmup",
        "params": {"warmup_epochs": 5, "eta_min": 1e-6},
    },
    "optimizer": {"name": "adamw", "params": {"lr": 8e-4, "weight_decay": 1e-2}},
    "batch_size": 96,
    "num_epochs": 500,  # Ceiling; early stopping should trigger first
    "early_stopping_patience": 15,
}

EXPERIMENTS = [
    {
        **BASE_CONFIG,
        "loss": {"name": "ssim", "params": {"data_range": 1.0, "channel": 1}},
    },
    {
        **BASE_CONFIG,
        "loss": {"name": "ssim+l1", "params": {"ssim_weight": 0.8, "l1_weight": 0.2, ...}},
    },
]
```

## Trainer Class

### Lifecycle

1. **Initialization**: `Trainer(config)`
2. **Setup**: `trainer.setup()` - device, model, datasets, optimizer
3. **Training Loop**: `trainer.run()` - epochs, validation, checkpoints

### Training Loop

```python
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch()
    valid_loss, metrics = trainer.validate_epoch(epoch)

    if valid_loss < best_valid_loss:
        # Save best checkpoint
        trainer._save_visualizations(epoch)
        trainer._save_checkpoint(f"epochs/{epoch}/weights.pth")

    # Always save latest
    trainer._save_checkpoint("latest_epoch.pth")

    if trainer.early_stopping(valid_loss):
        break
```

## Checkpointing

### Checkpoint Contents

```python
{
    "epoch": 42,
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "train_loss": 0.123,
    "valid_loss": 0.098,
    "best_valid_loss": 0.095,
    "train_losses": [...],
    "valid_losses": [...],
    "metric_histories": {"ssim": [...], "mae": [...]},
    "config": {...},
    "early_stopping_counter": 3,
    "early_stopping_best_loss": 0.095,
}
```

### Manual Checkpoint Loading

```python
from phd.training import load_checkpoint

checkpoint = load_checkpoint(
    "experiments/.../latest_epoch.pth",
    model, optimizer, scheduler, early_stopping
)
print(f"Resuming from epoch {checkpoint['epoch']}")
```

## Training Status

| Status | Description |
|--------|-------------|
| `NOT_STARTED` | Queued but never run |
| `RUNNING` | Currently training (or interrupted, will resume) |
| `ERROR` | Failed with exception |
| `FINISHED_EPOCHS` | Completed all epochs |
| `EARLY_STOPPING` | Stopped due to early stopping |

## Queue-Based Workflow

```bash
# 1. Register experiments (creates NOT_STARTED entries)
uv run register_experiments.py

# 2. Run next experiment from queue
uv run train.py

# 3. Run all until queue empty
uv run train.py --run-all

# 4. Check status
uv run train.py --show-queue

# 5. Retry failed experiments
uv run register_experiments.py --reset-errors
```

### Priority Order

1. **RUNNING** - Resume interrupted experiments first (by last_started)
2. **NOT_STARTED** - Run new queued experiments (by queued_at)
3. Skip: `FINISHED_EPOCHS`, `EARLY_STOPPING` (completed)
4. Skip: `ERROR` (requires `--reset-errors` to retry)

### Interrupt Handling (Ctrl+C)

Ctrl+C kills the process immediately. The status stays `RUNNING` and the experiment will resume on the next `train.py` invocation from the latest checkpoint.

## Resume Behavior

The trainer checks both the registry status and whether a checkpoint exists:

| Status | Checkpoint exists? | Action |
|--------|-------------------|--------|
| `RUNNING` | Yes/No | Resume from checkpoint |
| `FINISHED_EPOCHS` | - | Skip (already complete) |
| `EARLY_STOPPING` | - | Skip (already complete) |
| `ERROR` | Yes | Resume from checkpoint |
| `ERROR` | No | Delete directory, restart |
| `NOT_STARTED` | Yes | Resume from checkpoint (after `--reset-errors`) |
| `NOT_STARTED` | No | Start fresh |
| Unknown/Missing | - | Delete directory, restart |

## Early Stopping

```python
from phd.training import EarlyStopping

early_stopping = EarlyStopping(
    patience=7,      # Epochs without improvement
    min_delta=0.0001 # Minimum improvement threshold
)

for epoch in range(num_epochs):
    valid_loss = validate()
    if early_stopping(valid_loss):
        print("Early stopping triggered")
        break
```

## Experiment Registry

The registry tracks all experiments in `experiments/experiments_registry.json`:

```json
{
    "ssim_7e8d66": {
        "status": "NOT_STARTED",
        "queued_at": "2024-01-15T10:30:00",
        "created": "2024-01-15T10:30:00",
        "last_started": null,
        "finished": null,
        "runs": 0,
        "config": {
            "model": {"type": "unet", ...},
            "loss": {"name": "ssim", ...},
            "scheduler": {...},
            "optimizer": {...},
            "batch_size": 96,
            "num_epochs": 500
        }
    }
}
```

### Registry Functions

```python
from phd.training.registry import (
    # Queue management
    queue_experiment,           # Register without running
    get_next_experiment,        # Get next by priority
    get_queue_status,           # Get counts by status
    reset_error_experiments,    # Reset ERROR -> NOT_STARTED

    # Status tracking
    register_experiment,
    update_experiment_status,
    get_experiment_status,
    list_experiments,
)
```

## Output Structure

```
experiments/train_nn1_cropped/{experiment_name}/
├── config.json              # Full configuration
├── latest_epoch.pth         # Latest checkpoint (for resume)
├── error.log                # Error traceback (if failed/crashed)
├── epochs.csv               # Per-epoch data (losses, metrics, timing, lr)
└── epochs/
    └── {best_epoch}/
        ├── weights.pth      # Model weights
        └── viz/
            ├── target_is_real/
            └── target_is_interpolated/
```

## Environment Requirements

- **GPU**: Required (CUDA)
- **Environment Variable**: `DATASETS_DIR` in `.env`
- **Dataset**: Preprocessed RSNA dataset at `$DATASETS_DIR/pre/rsna-intracranial-hemorrhage-detection/`
