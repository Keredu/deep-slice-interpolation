#!/usr/bin/env python3
"""Benchmark torch.compile modes for training.

This script compares different torch.compile modes to find the optimal
configuration for our static-shape training workload.

Usage:
    uv run scripts/benchmark_compile_modes.py

Results (2026-02-05, RTX 3080 Ti, batch_size=64):
    eager                         :  217.87 ms/iter  (baseline)
    default                       :  178.88 ms/iter  (1.22x vs eager)
    reduce-overhead               :  175.69 ms/iter  (1.24x vs eager)
    max-autotune-no-cudagraphs    :  171.87 ms/iter  (1.27x vs eager)
    max-autotune                  :  160.34 ms/iter  (1.36x vs eager)

Key finding: max-autotune is 7% faster than max-autotune-no-cudagraphs for
static-shape training because CUDA graphs work fine with:
  - Static input shapes (N, 2, 256, 256)
  - Standard autograd backward pass (AOTAutograd handles it)

CUDA graphs DON'T work with: dynamic shapes, control flow, input mutations.

See also: experiments/EXPERIMENT_LOG.md for detailed documentation.
"""

import time

import torch
from torch.amp import GradScaler, autocast

# Import your actual model
from phd.models.setup_model import setup_model


def benchmark_mode(mode: str | None, num_warmup: int = 3, num_iterations: int = 20):
    """Benchmark a torch.compile mode."""
    device = torch.device("cuda")

    # Create model (same as your training)
    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,  # Faster for benchmarking
        model_type="unet",
        encoder_name="tu-tf_efficientnetv2_s",
    ).to(device)

    if mode is not None:
        print(f"\nCompiling with mode='{mode}'...")
        compile_start = time.perf_counter()
        model = torch.compile(model, mode=mode)
        compile_time = time.perf_counter() - compile_start
        print(f"  torch.compile() call: {compile_time:.2f}s")
    else:
        print("\nEager mode (no compilation)...")

    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    criterion = torch.nn.L1Loss()

    # Static input shape
    batch_size = 64
    x = torch.randn(batch_size, 2, 256, 256, device=device)
    target = torch.randn(batch_size, 1, 256, 256, device=device)

    # Warmup (includes Triton compilation for autotune modes)
    print(f"  Warming up ({num_warmup} iterations)...")
    warmup_start = time.perf_counter()
    for _ in range(num_warmup):
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            output = model(x)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warmup time: {warmup_time:.2f}s")

    # Benchmark
    print(f"  Benchmarking ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            output = model(x)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_iter = (elapsed / num_iterations) * 1000
    print(f"  Result: {ms_per_iter:.2f} ms/iteration")

    # Cleanup - aggressive memory clearing between modes
    del model, optimizer, scaler, x, target
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # Reset dynamo to clear compiled graphs
    torch._dynamo.reset()

    return ms_per_iter


def main():
    print("=" * 60)
    print("torch.compile Mode Benchmark for Training")
    print("=" * 60)
    print("Input shape: (64, 2, 256, 256)")
    print("Target shape: (64, 1, 256, 256)")
    print(f"GPU: {torch.cuda.get_device_name()}")

    modes = [
        None,  # Eager (baseline)
        "default",
        "reduce-overhead",
        "max-autotune-no-cudagraphs",  # Current
        "max-autotune",
    ]

    results = {}
    for mode in modes:
        try:
            results[mode or "eager"] = benchmark_mode(mode)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[mode or "eager"] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    baseline = results.get("eager")
    for mode, ms in results.items():
        if ms is None:
            print(f"  {mode:30s}: FAILED")
        elif baseline and mode != "eager":
            speedup = baseline / ms
            print(f"  {mode:30s}: {ms:7.2f} ms/iter  ({speedup:.2f}x vs eager)")
        else:
            print(f"  {mode:30s}: {ms:7.2f} ms/iter  (baseline)")

    # Recommendation
    print("\n" + "-" * 60)
    valid_results = {k: v for k, v in results.items() if v is not None and k != "eager"}
    if valid_results:
        best_mode = min(valid_results, key=valid_results.get)
        print(f"RECOMMENDATION: Use mode='{best_mode}'")


if __name__ == "__main__":
    main()
