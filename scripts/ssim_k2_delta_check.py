#!/usr/bin/env python3
r"""Evaluate the metric impact of K2 choice (0.03 vs 0.4) at test time.

Context (paper S3.4-05). We train SSIM-family losses with K2 = 0.4 as a
numerical-conditioning knob (without it, the SSIM denominator vanishes on
low-texture CT windows and training diverges; see methods.tex). We report
test-set SSIM and MS-SSIM at Wang et al.'s K2 = 0.03 so the numbers are
comparable to prior literature. This script quantifies the delta introduced
by that reporting choice: for every trained model in the paper's evaluation,
it computes per-slice SSIM and MS-SSIM at BOTH K settings on the same
predictions, and reports the per-model mean (and max) absolute delta.

Expected outcome: the delta is small (dominated by low-texture windows where
K2 sits in the denominator), so the metric ranking of models is unaffected
by the K2 choice and the two-system concern reduces to a reporting convention.

Outputs:
  results/tables/ssim_k2_delta_slice.csv     (per-slice, per-model)
  results/tables/ssim_k2_delta_summary.csv   (per-model summary)

Reproduce:
    uv run scripts/ssim_k2_delta_check.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from pytorch_msssim import MS_SSIM, SSIM

load_dotenv()

from phd.config_io import resolve_config_path  # noqa: E402
from phd.datasets.interpolation.two_to_one_slice import (  # noqa: E402
    STANDARD_TRANSFORM,
    TwoToOneSliceTestDataset,
)
from phd.models.setup_model import setup_model  # noqa: E402
from phd.viz import predict_via_patch_reconstruction  # noqa: E402

EXPERIMENTS_DIR = Path("experiments/train_nn1_cropped")
TABLES_DIR = Path("results/tables")

# Trained experiments reported in the paper's test evaluation
# (from revision_slice_metrics_2026-04-16.csv, excluding baselines).
EXPERIMENTS = [
    "l1_lr8e-4_b39be9",
    "l1_lr1e-4_66e5e2",
    "mse_lr8e-4_b558b9",
    "ssim_lr3e-3_94f982",
    "ssim_lr8e-4_1b8c15",
    "msssim+l1_lr8e-4_bc1d65",
    "msssim+l1_lr8e-4_e6d845",
]

BATCH_SIZE = 16


def _resolve_best_weights(experiment_dir: Path) -> Path | None:
    epochs_csv = experiment_dir / "epochs.csv"
    if not epochs_csv.exists():
        return None
    df = pd.read_csv(epochs_csv)
    finite = df[np.isfinite(df["valid_loss"])]
    if finite.empty:
        return None
    best_epoch_1based = int(finite.loc[finite["valid_loss"].idxmin(), "epoch"])
    weights = experiment_dir / "epochs" / str(best_epoch_1based - 1) / "weights.pth"
    if weights.exists():
        return weights
    epochs_dir = experiment_dir / "epochs"
    available = sorted(
        int(d.name) for d in epochs_dir.iterdir()
        if d.is_dir() and (d / "weights.pth").exists()
    )
    if not available:
        return None
    return epochs_dir / str(available[-1]) / "weights.pth"


def _load_model(experiment_dir: Path, weights_path: Path):
    config = json.loads((experiment_dir / "config.json").read_text())
    model_cfg = config["model"]
    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,
        model_type=model_cfg["type"],
        encoder_name=model_cfg["encoder_name"],
    )
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, resolve_config_path(config["data_path"])


def _infer_datasets_dir(data_path: Path) -> Path:
    parts = list(data_path.resolve().parts)
    pre_idx = parts.index("pre")
    return Path(*parts[:pre_idx])


def _load_test_dataset(data_path: Path) -> TwoToOneSliceTestDataset:
    if not os.getenv("DATASETS_DIR"):
        os.environ["DATASETS_DIR"] = str(_infer_datasets_dir(data_path))
    return TwoToOneSliceTestDataset(
        root_dir=str(data_path),
        stage="test",
        mode="target_is_real",
        transform=STANDARD_TRANSFORM,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Paper's default (Wang et al.) and training-time K.
    ssim_003 = SSIM(data_range=1.0, size_average=False, channel=1, K=(0.01, 0.03)).to(device)
    ssim_04  = SSIM(data_range=1.0, size_average=False, channel=1, K=(0.01, 0.4 )).to(device)
    ms_003   = MS_SSIM(data_range=1.0, size_average=False, channel=1, K=(0.01, 0.03)).to(device)
    ms_04    = MS_SSIM(data_range=1.0, size_average=False, channel=1, K=(0.01, 0.4 )).to(device)

    rows: list[dict] = []

    for exp_name in EXPERIMENTS:
        exp_dir = EXPERIMENTS_DIR / exp_name
        weights = _resolve_best_weights(exp_dir)
        if weights is None:
            logger.warning(f"No weights for {exp_name}, skipping")
            continue

        model, data_path = _load_model(exp_dir, weights)
        model = model.to(device)
        test_dataset = _load_test_dataset(data_path)
        logger.info(f"[{exp_name}] weights={weights.parent.name}, patients={len(test_dataset)}")

        with torch.no_grad():
            for patient_idx in range(len(test_dataset)):
                patient_id = test_dataset.get_patient_id_by_index(patient_idx)
                inputs, targets = test_dataset[patient_idx]
                n = inputs.shape[0]
                for start in range(0, n, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n)
                    b_in = inputs[start:end].to(device)
                    b_tg = targets[start:end].to(device)
                    b_pred = predict_via_patch_reconstruction(
                        model=model, batch_inputs=b_in, device=device
                    )
                    s003 = ssim_003(b_pred, b_tg).cpu().numpy()
                    s04  = ssim_04 (b_pred, b_tg).cpu().numpy()
                    m003 = ms_003  (b_pred, b_tg).cpu().numpy()
                    m04  = ms_04   (b_pred, b_tg).cpu().numpy()
                    for i in range(end - start):
                        rows.append({
                            "experiment": exp_name,
                            "patient_id": patient_id,
                            "triplet_index": start + i,
                            "ssim_k003": float(s003[i]),
                            "ssim_k04":  float(s04[i]),
                            "ms_ssim_k003": float(m003[i]),
                            "ms_ssim_k04":  float(m04[i]),
                        })
                if (patient_idx + 1) % 10 == 0:
                    logger.info(f"[{exp_name}]   {patient_idx + 1}/{len(test_dataset)} patients")

        # Free GPU memory before next model.
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    slice_csv = TABLES_DIR / "ssim_k2_delta_slice.csv"
    df.to_csv(slice_csv, index=False)
    logger.info(f"Wrote {len(df)} rows to {slice_csv}")

    # Patient-level aggregation (mean over triplets within each patient)
    pat = (
        df.groupby(["experiment", "patient_id"])
        .agg(
            ssim_k003=("ssim_k003", "mean"),
            ssim_k04=("ssim_k04", "mean"),
            ms_ssim_k003=("ms_ssim_k003", "mean"),
            ms_ssim_k04=("ms_ssim_k04", "mean"),
        )
        .reset_index()
    )
    pat["d_ssim"] = pat["ssim_k003"] - pat["ssim_k04"]
    pat["d_ms_ssim"] = pat["ms_ssim_k003"] - pat["ms_ssim_k04"]

    summary = (
        pat.groupby("experiment")
        .agg(
            n_patients=("d_ssim", "size"),
            ssim_mean_k003=("ssim_k003", "mean"),
            ssim_mean_k04=("ssim_k04", "mean"),
            ssim_mean_delta=("d_ssim", "mean"),
            ssim_std_delta=("d_ssim", "std"),
            ms_ssim_mean_k003=("ms_ssim_k003", "mean"),
            ms_ssim_mean_k04=("ms_ssim_k04", "mean"),
            ms_ssim_mean_delta=("d_ms_ssim", "mean"),
            ms_ssim_std_delta=("d_ms_ssim", "std"),
        )
        .reset_index()
    )
    summary_csv = TABLES_DIR / "ssim_k2_delta_summary.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info(f"Wrote patient-level summary to {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
