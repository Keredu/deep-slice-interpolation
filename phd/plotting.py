"""Plotting utilities for training visualization."""

import csv
from pathlib import Path


def save_metrics_csv(
    metric_histories: dict[str, list[float]],
    save_dir: Path,
    train_losses: list[float] | None = None,
    valid_losses: list[float] | None = None,
    best_epochs: list[bool] | None = None,
    learning_rates: list[float] | None = None,
    epoch_times: list[float] | None = None,
    train_times: list[float] | None = None,
    valid_times: list[float] | None = None,
) -> None:
    """Save metric histories to a CSV file.

    Creates a CSV file with columns for each metric and rows for each epoch.
    Handles metrics with different lengths if needed.
    All losses are computed at 256×256 resolution.

    Args:
        metric_histories: Dictionary with metric names as keys and lists of values as values.
        save_dir: Directory path where to save the CSV file.
        train_losses: Optional list of training losses to include.
        valid_losses: Optional list of validation losses to include.
        best_epochs: Optional list of booleans indicating if each epoch was a new best.
        learning_rates: Optional list of learning rates at each epoch.
        epoch_times: Optional list of total epoch times in seconds.
        train_times: Optional list of training phase times in seconds.
        valid_times: Optional list of validation phase times in seconds.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine the maximum number of epochs
    all_lengths = [len(v) for v in metric_histories.values() if v]
    if train_losses:
        all_lengths.append(len(train_losses))
    if valid_losses:
        all_lengths.append(len(valid_losses))

    if not all_lengths:
        return

    max_epochs = max(all_lengths)

    # Build column headers and data
    columns = ["epoch", "is_best"]
    if train_losses:
        columns.append("train_loss")
    if valid_losses:
        columns.append("valid_loss")

    # Add metric columns in consistent order
    metric_order = ["ssim", "ms_ssim", "mae", "gradient_mae", "psnr", "ncc"]
    for metric in metric_order:
        if metric_histories.get(metric):
            columns.append(metric)

    # Add timing and learning rate columns
    if learning_rates:
        columns.append("learning_rate")
    if epoch_times:
        columns.append("epoch_time_s")
    if train_times:
        columns.append("train_time_s")
    if valid_times:
        columns.append("valid_time_s")

    # Write CSV
    csv_path = save_dir / "epochs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for epoch in range(max_epochs):
            row = [epoch + 1]  # 1-indexed epoch

            # Add is_best column
            if best_epochs and epoch < len(best_epochs):
                row.append(1 if best_epochs[epoch] else 0)
            else:
                row.append("")

            if train_losses:
                row.append(train_losses[epoch] if epoch < len(train_losses) else "")
            if valid_losses:
                row.append(valid_losses[epoch] if epoch < len(valid_losses) else "")

            for metric in metric_order:
                if metric_histories.get(metric):
                    values = metric_histories[metric]
                    row.append(values[epoch] if epoch < len(values) else "")

            # Add timing and learning rate values
            if learning_rates:
                row.append(learning_rates[epoch] if epoch < len(learning_rates) else "")
            if epoch_times:
                row.append(f"{epoch_times[epoch]:.2f}" if epoch < len(epoch_times) else "")
            if train_times:
                row.append(f"{train_times[epoch]:.2f}" if epoch < len(train_times) else "")
            if valid_times:
                row.append(f"{valid_times[epoch]:.2f}" if epoch < len(valid_times) else "")

            writer.writerow(row)
