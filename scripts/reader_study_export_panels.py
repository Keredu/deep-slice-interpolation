"""Export 2x2 panel PNGs for the radiologist reader study.

For each selected case, creates a side-by-side panel showing:
  Top row:    I_k (Slice k, input)  |  I_{k+2} (Slice k+2, input)
  Bottom row: Real I_{k+1} (GT)     |  AI I_{k+1} (prediction)

Usage:
    uv run scripts/reader_study_export_panels.py \
        --cases reader_study/cases.csv \
        --image-dir /path/to/preprocessed/pngs \
        --predictions-dir /path/to/export/patient_dirs \
        --output-dir reader_study/panels

The script needs:
  1. Preprocessed PNG images (512x512, grayscale, [0,1] range as uint8 0-255)
     These are the same PNGs used for training, located under DATASETS_DIR.
  2. AI predictions from the export script output.
     Each patient directory should contain info.json with output_slices metadata.

If running on a machine without the data, the script will print what's needed
and exit gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

# Use non-interactive backend for headless rendering
matplotlib.use("Agg")


def load_cases(csv_path: Path) -> pd.DataFrame:
    """Load cases CSV produced by reader_study_select_cases.py."""
    df = pd.read_csv(csv_path)
    required = {"case_id", "patient_id", "triplet_index"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Cases CSV missing columns: {missing}"
        raise ValueError(msg)
    return df


def load_png_grayscale(path: Path) -> np.ndarray:
    """Load a grayscale PNG and return float32 array in [0, 1]."""
    with Image.open(path) as img:
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    return arr


def find_sop_ids_for_triplet(
    patient_id: str,
    triplet_index: int,
    pre_dir: Path,
) -> tuple[str, str, str]:
    """Look up SOP Instance UIDs for a triplet from the preprocessed df.csv.

    Returns (sop_k, sop_k1, sop_k2) for the triplet at the given index.
    """
    df_path = pre_dir / "df.csv"
    if not df_path.exists():
        msg = f"Missing preprocessed metadata: {df_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(df_path)
    if "split" in df.columns and "stage" not in df.columns:
        df = df.rename(columns={"split": "stage"})
    df = df[df["stage"] == "test"]

    patient_df = df[df["PatientID"] == patient_id].sort_values("order").reset_index(drop=True)
    if len(patient_df) < triplet_index + 3:
        msg = (
            f"Patient {patient_id} has {len(patient_df)} slices, "
            f"but triplet_index={triplet_index} requires at least {triplet_index + 3}"
        )
        raise ValueError(msg)

    sop_k = patient_df.iloc[triplet_index]["SOPInstanceUID"]
    sop_k1 = patient_df.iloc[triplet_index + 1]["SOPInstanceUID"]
    sop_k2 = patient_df.iloc[triplet_index + 2]["SOPInstanceUID"]
    return sop_k, sop_k1, sop_k2


def find_ai_prediction_png(
    patient_id: str,
    triplet_index: int,
    predictions_dir: Path,
) -> Path:
    """Find the AI prediction PNG for a given triplet from the export directory.

    The export script creates per-patient directories with info.json.
    We look for the AI slice between triplet_index and triplet_index+1
    (which corresponds to the interpolation of the middle slice).
    """
    patient_dir = predictions_dir / patient_id
    info_path = patient_dir / "info.json"
    if not info_path.exists():
        msg = f"Missing export info: {info_path}"
        raise FileNotFoundError(msg)

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    # The export interleaves real and AI slices. For triplet at index t,
    # the AI slice between slice t and t+1 is what we want.
    # In the export, real slice at original order t has index 2*t in output,
    # and the AI slice between t and t+1 has index 2*t+1.
    target_between_orders = [triplet_index, triplet_index + 1]

    for entry in info["output_slices"]:
        if entry["kind"] == "ai":
            between = entry.get("between_orders")
            if between == target_between_orders:
                png_path = patient_dir / entry["png"]
                if png_path.exists():
                    return png_path
                msg = f"AI prediction PNG listed but missing: {png_path}"
                raise FileNotFoundError(msg)

    msg = (
        f"Could not find AI prediction for patient={patient_id}, "
        f"triplet_index={triplet_index} (between_orders={target_between_orders}) "
        f"in {info_path}"
    )
    raise ValueError(msg)


def create_panel(
    slice_k: np.ndarray,
    slice_k2: np.ndarray,
    ground_truth: np.ndarray,
    ai_prediction: np.ndarray,
    case_id: int,
    save_path: Path,
    dpi: int = 150,
) -> None:
    """Create a 2x2 panel PNG for one case.

    Layout:
        Top-left:     Slice k (input)         | Top-right:    Slice k+2 (input)
        Bottom-left:  Ground Truth (k+1)       | Bottom-right: AI Interpolation (k+1)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    panels = [
        (axes[0, 0], slice_k, "Slice k (input)"),
        (axes[0, 1], slice_k2, "Slice k+2 (input)"),
        (axes[1, 0], ground_truth, "Ground Truth (k+1)"),
        (axes[1, 1], ai_prediction, "AI Interpolation (k+1)"),
    ]

    for ax, img, title in panels:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Case {case_id}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main() -> None:
    """Export reader study panels."""
    parser = argparse.ArgumentParser(description="Export 2x2 panel PNGs for reader study")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("reader_study/cases.csv"),
        help="Path to cases.csv from selection script",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing preprocessed PNGs (512x512). "
            "If not provided, inferred from DATASETS_DIR env var."
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-patient export directories with info.json. "
            "Default: output/test_interpolated_export/train_nn1_cropped/ssim_lr3e-3_94f982/epoch_43"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reader_study/panels"),
        help="Output directory for panel PNGs",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output PNGs")
    args = parser.parse_args()

    load_dotenv()

    # Resolve image directory
    if args.image_dir is None:
        datasets_dir = os.getenv("DATASETS_DIR")
        if datasets_dir:
            # Standard preprocessed PNG location
            candidates = list(Path(datasets_dir).glob("pre/rsna-intracranial-hemorrhage-detection/*_windowed*"))
            if candidates:
                args.image_dir = candidates[0]
                logger.info(f"Auto-detected image directory: {args.image_dir}")
            else:
                logger.error(
                    f"DATASETS_DIR={datasets_dir} set but no windowed PNG directory found. "
                    "Provide --image-dir explicitly."
                )
                return
        else:
            logger.error(
                "No --image-dir provided and DATASETS_DIR not set. "
                "This script needs access to the preprocessed PNG images. "
                "On the GPU machine, set DATASETS_DIR in .env or provide --image-dir."
            )
            return

    # Resolve predictions directory
    if args.predictions_dir is None:
        default_pred = Path("output/test_interpolated_export/train_nn1_cropped/ssim_lr3e-3_94f982/epoch_43")
        if default_pred.exists():
            args.predictions_dir = default_pred
        else:
            logger.error(
                f"Default predictions directory not found: {default_pred}\n"
                "Run the export script first on the GPU machine:\n"
                "  uv run scripts/export_test_interpolated_series.py\n"
                "Or provide --predictions-dir explicitly."
            )
            return

    # Resolve pre_dir for metadata lookup
    datasets_dir = os.getenv("DATASETS_DIR")
    if datasets_dir:
        pre_dir = Path(datasets_dir) / "pre/rsna-intracranial-hemorrhage-detection"
    else:
        logger.error("DATASETS_DIR not set. Cannot look up SOP Instance UIDs for triplets.")
        return

    if not pre_dir.exists():
        logger.error(f"Preprocessed metadata directory not found: {pre_dir}")
        return

    # Load cases
    cases = load_cases(args.cases)
    logger.info(f"Loaded {len(cases)} cases from {args.cases}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each case
    n_success = 0
    n_fail = 0
    for _, row in cases.iterrows():
        case_id = int(row["case_id"])
        patient_id = row["patient_id"]
        triplet_index = int(row["triplet_index"])

        try:
            # Look up SOP UIDs
            sop_k, sop_k1, sop_k2 = find_sop_ids_for_triplet(patient_id, triplet_index, pre_dir)

            # Load real slices
            slice_k = load_png_grayscale(args.image_dir / f"{sop_k}.png")
            slice_k1 = load_png_grayscale(args.image_dir / f"{sop_k1}.png")
            slice_k2 = load_png_grayscale(args.image_dir / f"{sop_k2}.png")

            # Load AI prediction
            ai_png_path = find_ai_prediction_png(patient_id, triplet_index, args.predictions_dir)
            ai_pred = load_png_grayscale(ai_png_path)

            # Create panel
            save_path = args.output_dir / f"case_{case_id:02d}.png"
            create_panel(slice_k, slice_k2, slice_k1, ai_pred, case_id, save_path, dpi=args.dpi)
            logger.info(f"Case {case_id:2d}: exported panel to {save_path}")
            n_success += 1

        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Case {case_id:2d}: skipped - {e}")
            n_fail += 1

    logger.info(f"Done: {n_success} panels exported, {n_fail} skipped")
    if n_fail > 0:
        logger.warning(
            "Some panels could not be exported. Ensure the export script has been run "
            "on the GPU machine and all data is available."
        )


if __name__ == "__main__":
    main()
