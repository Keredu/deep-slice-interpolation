"""Register experiments for later execution by train.py.

Define experiments here, then run to queue them.

Usage:
    uv run register_experiments.py              # Register all experiments
    uv run register_experiments.py --show       # Dry run (show what would be registered)
    uv run register_experiments.py --reset-errors  # Reset ERROR -> RUNNING
"""

import argparse
from pathlib import Path

from loguru import logger

from phd.training import create_config
from phd.training.experiments import generate_experiment_name
from phd.training.registry import (
    get_queue_status,
    list_experiments,
    queue_experiment,
    reset_error_experiments,
)

# =============================================================================
# DEFINE YOUR EXPERIMENTS HERE
# =============================================================================
# Each experiment must provide all required config:
# - model, loss, scheduler, optimizer, batch_size, num_epochs
# Optional transfer initialization:
# - init_from_experiment: source experiment name under experiments_dir
# - init_from_checkpoint: relative path inside source experiment (default: latest_epoch.pth)

# Shared config across experiments
BATCH_SIZE = 96

BASE_CONFIG = {
    "model": {"type": "unet", "encoder_name": "tu-tf_efficientnetv2_s", "pretrained": True},
    "scheduler": {
        "name": "cosine_warmup",
        "params": {"warmup_epochs": 5, "eta_min": 1e-6},
    },
    "optimizer": {"name": "adamw", "params": {"lr": 8e-4, "weight_decay": 1e-2}},
    "batch_size": BATCH_SIZE,
    "num_workers": 2,  # Reduced from 4 to avoid DataLoader shm race conditions
    "valid_batch_size": 64,
    "valid_num_workers": 0,
    "valid_pin_memory": False,
    "num_epochs": 500,  # Ceiling; early stopping should trigger first
    "early_stopping_patience": 15,
    "train_size": None,  # Full dataset
    "valid_size": None,  # Full dataset
    "generate_test_viz_real": False,
    "generate_test_viz_interpolated": False,
}


def config_with_lr(lr: float) -> dict:
    """Create a config with a custom learning rate."""
    config = BASE_CONFIG.copy()
    config["optimizer"] = {
        "name": "adamw",
        "params": {"lr": lr, "weight_decay": 1e-2},
    }
    return config


# All available loss functions for testing
EXPERIMENTS = [
    # === TRANSFER RESTART FROM PROMISING NaN RUN ===
    # Start explicitly from the best checkpoint (epoch folder 20 -> CSV epoch 21)
    # of msssim+l1_lr8e-4_e6d845.
    # and continue with a lower LR plus a stronger L1 component for stability/detail.
    {
        **config_with_lr(3e-4),
        "batch_size": 64,  # Keep same batch size as source run
        "loss": {
            "name": "msssim+l1",
            "params": {"msssim_weight": 0.2, "l1_weight": 0.8, "data_range": 1.0, "channel": 1},
        },
        "num_epochs": 300,
        "init_from_experiment": "msssim+l1_lr8e-4_e6d845",
        "init_from_checkpoint": "epochs/20/weights.pth",
    },
    # === TRANSFER CONTINUATION FROM BEST CHECKPOINT (CONSERVATIVE) ===
    # Continue from best checkpoint of msssim+l1_lr3e-4_9b7aea with the same
    # successful strategy (same LR/weights/batch size), changing only the init source.
    {
        **config_with_lr(3e-4),
        "batch_size": 64,
        "loss": {
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.2,
                "l1_weight": 0.8,
                "data_range": 1.0,
                "channel": 1,
            },
        },
        "num_epochs": 300,
        "init_from_experiment": "msssim+l1_lr3e-4_9b7aea",
        "init_from_checkpoint": "epochs/18/weights.pth",
    },
    # === FINE-TUNING FROM BEST CHECKPOINT (PARAMETER SWEEP) ===
    # Keep base weights fixed and only tune optimization/numerical-stability knobs.
    # Base: experiments/train_nn1_cropped/msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth
    # A) Lower LR + constant LR (no scheduler)
    {
        **config_with_lr(1e-4),
        "batch_size": 64,
        "scheduler": {"name": "none", "params": {}},
        "loss": {
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.2,
                "l1_weight": 0.8,
                "data_range": 1.0,
                "channel": 1,
            },
        },
        "num_epochs": 300,
        "init_from_experiment": "msssim+l1_lr3e-4_9b7aea",
        "init_from_checkpoint": "epochs/18/weights.pth",
    },
    # B) Same as A + explicit MS-SSIM stability constants
    {
        **config_with_lr(1e-4),
        "batch_size": 64,
        "scheduler": {"name": "none", "params": {}},
        "loss": {
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.2,
                "l1_weight": 0.8,
                "data_range": 1.0,
                "channel": 1,
                "K": [0.01, 0.4],
            },
        },
        "num_epochs": 300,
        "init_from_experiment": "msssim+l1_lr3e-4_9b7aea",
        "init_from_checkpoint": "epochs/18/weights.pth",
    },
    # C) Smaller LR + explicit MS-SSIM stability constants
    {
        **config_with_lr(5e-5),
        "batch_size": 64,
        "scheduler": {"name": "none", "params": {}},
        "loss": {
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.2,
                "l1_weight": 0.8,
                "data_range": 1.0,
                "channel": 1,
                "K": [0.01, 0.4],
            },
        },
        "num_epochs": 300,
        "init_from_experiment": "msssim+l1_lr3e-4_9b7aea",
        "init_from_checkpoint": "epochs/18/weights.pth",
    },

    # === PROVEN SUCCESSFUL ===
    # 1. MSE (Mean Squared Error) - Baseline
    {
        **BASE_CONFIG,
        "loss": {"name": "mse", "params": {}},
    },
    # 2. L1 (Mean Absolute Error) - Best performer
    {
        **BASE_CONFIG,
        "loss": {"name": "l1", "params": {}},
    },

    # === NEW BATCH: L1-DOMINANT COMBINATIONS ===
    # 3. SSIM + L1 (50/50 balanced)
    {
        **BASE_CONFIG,
        "loss": {
            "name": "ssim+l1",
            "params": {"ssim_weight": 0.5, "l1_weight": 0.5, "data_range": 1.0, "channel": 1},
        },
    },
    # 4. SSIM + L1 (30/70 L1-dominant)
    {
        **BASE_CONFIG,
        "loss": {
            "name": "ssim+l1",
            "params": {"ssim_weight": 0.3, "l1_weight": 0.7, "data_range": 1.0, "channel": 1},
        },
    },
    # 5. SSIM + L1 (20/80 very conservative)
    {
        **BASE_CONFIG,
        "loss": {
            "name": "ssim+l1",
            "params": {"ssim_weight": 0.2, "l1_weight": 0.8, "data_range": 1.0, "channel": 1},
        },
    },
    # 6. MS-SSIM + L1 (30/70 L1-dominant)
    {
        **BASE_CONFIG,
        "loss": {
            "name": "msssim+l1",
            "params": {"msssim_weight": 0.3, "l1_weight": 0.7, "data_range": 1.0, "channel": 1},
        },
    },
    # 7. MS-SSIM + L1 (50/50 balanced)
    {
        **BASE_CONFIG,
        "loss": {
            "name": "msssim+l1",
            "params": {"msssim_weight": 0.5, "l1_weight": 0.5, "data_range": 1.0, "channel": 1},
        },
    },
    # === BATCH 3: LOWER LR WITH SSIM-DOMINANT (ablation study) ===
    # Test if lower LR stabilizes SSIM-dominant losses
    # 8. SSIM + L1 (80/20 SSIM-dominant) with lr=1e-4
    {
        **config_with_lr(1e-4),
        "loss": {
            "name": "ssim+l1",
            "params": {"ssim_weight": 0.8, "l1_weight": 0.2, "data_range": 1.0, "channel": 1},
        },
    },
    # 9. SSIM + L1 (80/20 SSIM-dominant) with lr=3e-4
    {
        **config_with_lr(3e-4),
        "loss": {
            "name": "ssim+l1",
            "params": {"ssim_weight": 0.8, "l1_weight": 0.2, "data_range": 1.0, "channel": 1},
        },
    },
    # 10. MS-SSIM + L1 (80/20 MS-SSIM-dominant) with lr=1e-4
    {
        **config_with_lr(1e-4),
        "loss": {
            "name": "msssim+l1",
            "params": {"msssim_weight": 0.8, "l1_weight": 0.2, "data_range": 1.0, "channel": 1},
        },
    },

    # === BATCH 4: LR ABLATION ON SUCCESSFUL LOSSES ===
    # Only L1 and MSE produced usable results. Test at lower LRs.
    # 11. L1 with lr=3e-4 (best performer, lower LR may find deeper optimum)
    {
        **config_with_lr(3e-4),
        "loss": {"name": "l1", "params": {}},
    },
    # 12. L1 with lr=1e-4 (conservative LR for maximum stability)
    {
        **config_with_lr(1e-4),
        "loss": {"name": "l1", "params": {}},
    },
    # 13. MSE with lr=3e-4 (stable baseline at alternative LR for comparison)
    {
        **config_with_lr(3e-4),
        "loss": {"name": "mse", "params": {}},
    },

    # === BATCH 5: SSIM REPRODUCTION (matching old working config) ===
    # Old experiment used: batch_size=32, no AMP, no torch.compile, CosineAnnealingLR
    # Current code uses AMP + torch.compile. Testing with old hyperparameters.
    # 16. Pure SSIM, lr=3e-3, batch_size=32 (exact old config)
    {
        **config_with_lr(3e-3),
        "batch_size": 32,
        "loss": {"name": "ssim", "params": {"data_range": 1.0, "channel": 1}},
    },
    # 17. SSIM+L1 (0.8/0.2), lr=3e-4, batch_size=32 (exact old config)
    {
        **config_with_lr(3e-4),
        "batch_size": 32,
        "loss": {
            "name": "ssim+l1",
            "params": {"ssim_weight": 0.8, "l1_weight": 0.2, "data_range": 1.0, "channel": 1},
        },
    },

    # === BATCH 6: SSIM STABILITY FIX ===
    # Root cause: SSIM can return negative values with default K2=0.03, causing 1-SSIM > 1.
    # Fix: K2=0.4 (pytorch_msssim recommendation) + nonnegative_ssim=True + loss clamping.
    # Also reverted batch_size to 96 (64 caused NaN in previously-stable losses).

    # 18. SSIM+L1 (0.8/0.2) with stability fix
    {
        **BASE_CONFIG,
        "loss": {
            "name": "ssim+l1",
            "params": {
                "ssim_weight": 0.8, "l1_weight": 0.2,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4], "nonnegative_ssim": True,
            },
        },
    },
    # 19. SSIM+L1 (0.5/0.5) balanced with stability fix
    {
        **BASE_CONFIG,
        "loss": {
            "name": "ssim+l1",
            "params": {
                "ssim_weight": 0.5, "l1_weight": 0.5,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4], "nonnegative_ssim": True,
            },
        },
    },
    # 20. MS-SSIM+L1 (0.5/0.5) balanced with stability fix
    {
        **BASE_CONFIG,
        "loss": {
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.5, "l1_weight": 0.5,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4],
            },
        },
    },
    # 21. Pure SSIM with stability fix (lr=3e-3, batch_size=32)
    {
        **config_with_lr(3e-3),
        "batch_size": 32,
        "loss": {
            "name": "ssim",
            "params": {
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4], "nonnegative_ssim": True,
            },
        },
    },
    # 23. SSIM+L1 (0.8/0.2) with stability fix at lr=3e-4
    #     LR ablation for SSIM-dominant: if #18 (lr=8e-4) converges, lower LR may go deeper
    {
        **config_with_lr(3e-4),
        "loss": {
            "name": "ssim+l1",
            "params": {
                "ssim_weight": 0.8, "l1_weight": 0.2,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4], "nonnegative_ssim": True,
            },
        },
    },
    # 24. MS-SSIM+L1 (0.3/0.7) L1-dominant with stability fix
    #     Completes MS-SSIM weight sweep: 0.3/0.7, 0.5/0.5 (#20), 0.8/0.2 (#26)
    {
        **BASE_CONFIG,
        "loss": {
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.3, "l1_weight": 0.7,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4],
            },
        },
    },

    # === BATCH 7: EXTENDED STABILITY FIX + WEIGHT SWEEP ===
    # Complements Batch 6 by filling gaps in weight ratios and LR ablation.

    # SSIM+L1 (0.3/0.7) L1-dominant with stability fix
    #     Completes weight sweep: Batch 6 has 0.8/0.2 and 0.5/0.5
    {
        **BASE_CONFIG,
        "loss": {
            "name": "ssim+l1",
            "params": {
                "ssim_weight": 0.3, "l1_weight": 0.7,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4], "nonnegative_ssim": True,
            },
        },
    },
    # MS-SSIM+L1 (0.8/0.2) MS-SSIM-dominant with stability fix
    #     All previous MS-SSIM dominant attempts failed; K fix should stabilize
    {
        **BASE_CONFIG,
        "loss": {
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.8, "l1_weight": 0.2,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4],
            },
        },
    },
    # SSIM+L1 (0.5/0.5) balanced with stability fix at lr=3e-4
    #     LR ablation: if Batch 6 (lr=8e-4) converges, lower LR may find deeper optimum
    {
        **config_with_lr(3e-4),
        "loss": {
            "name": "ssim+l1",
            "params": {
                "ssim_weight": 0.5, "l1_weight": 0.5,
                "data_range": 1.0, "channel": 1,
                "K": [0.01, 0.4], "nonnegative_ssim": True,
            },
        },
    },
]


def get_experiment_configs(
    experiments: list[dict] | None = None,
    train_size: int | None = None,
    valid_size: int | None = None,
    num_epochs: int | None = None,
) -> list[dict]:
    """Build full configs from experiment definitions.

    Args:
        experiments: List of experiment overrides (defaults to EXPERIMENTS)
        train_size: Override training dataset size
        valid_size: Override validation dataset size
        num_epochs: Override number of epochs

    Returns:
        List of config override dicts ready for create_config()
    """
    if experiments is None:
        experiments = EXPERIMENTS

    configs = []
    for exp in experiments:
        config = exp.copy()

        # Apply CLI overrides if provided
        if train_size is not None:
            config["train_size"] = train_size
        if valid_size is not None:
            config["valid_size"] = valid_size
        if num_epochs is not None:
            config["num_epochs"] = num_epochs

        configs.append(config)

    return configs


def show_experiments(registry_dir: Path, experiment_configs: list[dict]) -> None:
    """Show what experiments would be registered."""
    registry = list_experiments(registry_dir)

    print("\n=== Experiment Registration Preview ===\n")

    new_experiments = []
    existing_experiments = []

    for exp_config in experiment_configs:
        # Generate name from config (doesn't need full validation)
        exp_name = generate_experiment_name(exp_config)

        if exp_name in registry:
            existing_experiments.append((exp_name, registry[exp_name].get("status")))
        else:
            new_experiments.append(exp_name)

    if new_experiments:
        print(f"New experiments to register ({len(new_experiments)}):")
        for name in new_experiments:
            print(f"  + {name}")
    else:
        print("No new experiments to register.")

    if existing_experiments:
        print(f"\nExisting experiments ({len(existing_experiments)}):")
        for name, status in existing_experiments:
            print(f"  = {name} [{status}]")

    print()


def register_all_experiments(registry_dir: Path, experiment_configs: list[dict]) -> int:
    """Register all experiments from config.

    Returns:
        Number of experiments registered
    """
    registered = 0

    for exp_config in experiment_configs:
        # Generate name first, then create full config
        exp_name = generate_experiment_name(exp_config)
        config = create_config(exp_name=exp_name, **exp_config)

        if queue_experiment(experiments_dir=registry_dir, exp_name=exp_name, config=config):
            logger.info(f"Registered: {exp_name}")
            registered += 1
        else:
            logger.debug(f"Already registered: {exp_name}")

    return registered


def main() -> None:
    """Register experiments for later execution."""
    parser = argparse.ArgumentParser(description="Register experiments for training")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Dry run: show what would be registered without registering",
    )
    parser.add_argument(
        "--reset-errors",
        action="store_true",
        help="Reset ERROR experiments to RUNNING status for priority resumption",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Training dataset size (None for full dataset)",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=None,
        help="Validation dataset size (None for full dataset)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs (None for default)",
    )
    args = parser.parse_args()

    # Directory paths (running from project root)
    registry_dir = Path("./experiments")
    experiments_dir = Path("./experiments/train_nn1_cropped")

    if args.reset_errors:
        reset_names = reset_error_experiments(registry_dir, output_dir=experiments_dir)
        if reset_names:
            logger.info(f"Reset {len(reset_names)} experiment(s) to RUNNING:")
            for name in reset_names:
                logger.info(f"  - {name}")
        else:
            logger.info("No ERROR experiments to reset.")
        return

    # Get experiment configurations
    experiment_configs = get_experiment_configs(
        train_size=args.train_size,
        valid_size=args.valid_size,
        num_epochs=args.num_epochs,
    )

    if args.show:
        show_experiments(registry_dir, experiment_configs)
        return

    # Register experiments
    registered = register_all_experiments(registry_dir, experiment_configs)

    if registered > 0:
        logger.info(f"Registered {registered} new experiment(s)")
    else:
        logger.info("No new experiments to register")

    # Show queue status
    counts = get_queue_status(registry_dir)
    pending = counts.get("NOT_STARTED", 0)
    logger.info(f"Queue status: {pending} pending, {counts.get('RUNNING', 0)} running")


if __name__ == "__main__":
    main()
