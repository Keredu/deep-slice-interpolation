"""Checkpoint save and load utilities."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .early_stopping import EarlyStopping


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loss: float,
    valid_loss: float,
    best_valid_loss: float,
    train_losses: list[float],
    valid_losses: list[float],
    metric_histories: dict[str, list[float]],
    config: dict,
    early_stopping: EarlyStopping,
    best_epochs: list[bool],
    per_crop_metric_histories: dict[int, dict[str, list[float]]],
    learning_rates: list[float],
    epoch_times: list[float],
    train_times: list[float],
    valid_times: list[float],
) -> None:
    """Save the model state, optimizer state, and training history to disk.

    Args:
        path: Path to save the checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        train_loss: Training loss for current epoch
        valid_loss: Validation loss for current epoch
        best_valid_loss: Best validation loss seen so far
        train_losses: List of training losses
        valid_losses: List of validation losses
        metric_histories: Dictionary of metric histories
        config: Training configuration
        early_stopping: Early stopping instance
        best_epochs: List of booleans indicating which epochs had best validation loss
        per_crop_metric_histories: Per-crop metric histories for validation
        learning_rates: List of learning rates at each epoch
        epoch_times: List of total epoch times in seconds
        train_times: List of training phase times in seconds
        valid_times: List of validation phase times in seconds
    """
    torch.save(
        obj={
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_valid_loss": best_valid_loss,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "metric_histories": metric_histories,
            "config": config,
            "early_stopping_counter": early_stopping.counter,
            "early_stopping_best_loss": early_stopping.best_loss,
            "best_epochs": best_epochs,
            "per_crop_metric_histories": per_crop_metric_histories,
            "learning_rates": learning_rates,
            "epoch_times": epoch_times,
            "train_times": train_times,
            "valid_times": valid_times,
        },
        f=path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    early_stopping: EarlyStopping | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore model/optimizer state.

    Args:
        path: Path to the checkpoint file
        model: Model to restore state to
        optimizer: Optimizer to restore state to
        scheduler: Optional scheduler to restore state to
        early_stopping: Optional early stopping instance to restore state to

    Returns:
        Dictionary with restored training state:
            - epoch: Last completed epoch
            - train_losses: List of training losses
            - valid_losses: List of validation losses
            - best_valid_loss: Best validation loss
            - metric_histories: Dictionary of metric histories
            - best_epochs: List of booleans indicating which epochs had best validation loss
    """
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if early_stopping is not None:
        early_stopping.counter = checkpoint["early_stopping_counter"]
        early_stopping.best_loss = checkpoint["early_stopping_best_loss"]

    return {
        "epoch": checkpoint["epoch"],
        "train_losses": checkpoint["train_losses"],
        "valid_losses": checkpoint["valid_losses"],
        "best_valid_loss": checkpoint["best_valid_loss"],
        "metric_histories": checkpoint.get("metric_histories", {}),
        "best_epochs": checkpoint.get("best_epochs", []),
        "per_crop_metric_histories": checkpoint.get("per_crop_metric_histories", {}),
        "learning_rates": checkpoint.get("learning_rates", []),
        "epoch_times": checkpoint.get("epoch_times", []),
        "train_times": checkpoint.get("train_times", []),
        "valid_times": checkpoint.get("valid_times", []),
    }
