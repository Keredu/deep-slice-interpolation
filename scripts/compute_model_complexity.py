"""Compute model complexity metrics for the paper (B3).

Reports:
  - Parameter count (total and trainable)
  - FLOPs at 512x512 input resolution (via torch.utils.flop_counter)
  - Inference time (mean ± std over N forward passes)

Usage:
    uv run scripts/compute_model_complexity.py --device cuda
    uv run scripts/compute_model_complexity.py --device cpu --output results/tables/model_complexity.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from loguru import logger

from phd.models.setup_model import setup_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Compute model complexity metrics.")
    p.add_argument("--device", type=str, default="cpu", help="Device for inference timing (cpu or cuda).")
    p.add_argument("--warmup-runs", type=int, default=10, help="Warmup forward passes before timing.")
    p.add_argument("--timing-runs", type=int, default=50, help="Forward passes for timing measurement.")
    p.add_argument("--input-size", type=int, default=512, help="Input spatial resolution (square).")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/model_complexity.csv"),
        help="Output CSV path.",
    )
    return p.parse_args()


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for n, p in model.named_parameters() if "encoder" in n)
    decoder_params = sum(p.numel() for n, p in model.named_parameters() if "decoder" in n)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
    }


def estimate_flops(model: torch.nn.Module, input_size: int, device: torch.device) -> int | None:
    """Estimate FLOPs using torch.utils.flop_counter if available."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 2, input_size, input_size, device=device)

    try:
        from torch.utils.flop_counter import FlopCounterMode

        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            model(dummy)
        return flop_counter.get_total_flops()
    except ImportError:
        logger.warning("torch.utils.flop_counter not available; skipping FLOPs estimation.")
        return None


def measure_inference_time(
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    warmup_runs: int,
    timing_runs: int,
) -> dict[str, float]:
    """Measure inference time in milliseconds."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 2, input_size, input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timing
    times_ms = []
    with torch.no_grad():
        for _ in range(timing_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_arr = torch.tensor(times_ms)
    return {
        "inference_time_mean_ms": float(times_arr.mean()),
        "inference_time_std_ms": float(times_arr.std()),
        "inference_time_median_ms": float(times_arr.median()),
    }


def main() -> None:
    """Compute and report model complexity."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")

    logger.info(f"Instantiating model (input: 2×{args.input_size}×{args.input_size})")
    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,
        model_type="unet",
        encoder_name="tu-tf_efficientnetv2_s",
    )
    model.eval()

    # Parameter count
    params = count_parameters(model)
    logger.info(f"Total parameters: {params['total_params']:,}")
    logger.info(f"  Encoder: {params['encoder_params']:,}")
    logger.info(f"  Decoder: {params['decoder_params']:,}")
    logger.info(f"  Trainable: {params['trainable_params']:,}")

    # FLOPs
    flops = estimate_flops(model, args.input_size, device)
    if flops is not None:
        logger.info(f"FLOPs (single forward pass): {flops:,} ({flops / 1e9:.2f} GFLOPs)")

    # Inference time
    logger.info(f"Measuring inference time on {device} ({args.warmup_runs} warmup + {args.timing_runs} timed runs)")
    timing = measure_inference_time(model, args.input_size, device, args.warmup_runs, args.timing_runs)
    logger.info(
        f"Inference time: {timing['inference_time_mean_ms']:.1f} ± {timing['inference_time_std_ms']:.1f} ms "
        f"(median: {timing['inference_time_median_ms']:.1f} ms)"
    )

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": "U-Net + EfficientNetV2-S",
        "encoder": "tu-tf_efficientnetv2_s",
        "input_channels": 2,
        "output_channels": 1,
        "input_resolution": f"{args.input_size}x{args.input_size}",
        "total_params": params["total_params"],
        "encoder_params": params["encoder_params"],
        "decoder_params": params["decoder_params"],
        "trainable_params": params["trainable_params"],
        "flops": flops if flops is not None else "",
        "gflops": f"{flops / 1e9:.2f}" if flops is not None else "",
        "device": str(device),
        "inference_time_mean_ms": f"{timing['inference_time_mean_ms']:.1f}",
        "inference_time_std_ms": f"{timing['inference_time_std_ms']:.1f}",
        "inference_time_median_ms": f"{timing['inference_time_median_ms']:.1f}",
    }

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
