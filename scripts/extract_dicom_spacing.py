"""Extract slice thickness and inter-slice spacing from DICOM headers for test patients.

Addresses reviewer feedback R2-M1 (slice thickness distribution) and R2-M4 (uniform spacing).

Reads raw DICOM headers for all test-set patients, computes actual inter-slice distances
from ImagePositionPatient, and reports SliceThickness/SpacingBetweenSlices if available.

Usage:
    uv run scripts/extract_dicom_spacing.py

Output:
    results/tables/dicom_spacing_per_patient.csv   — per-patient spacing stats
    results/tables/dicom_spacing_summary.csv       — aggregate summary
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydicom import dcmread

warnings.filterwarnings("ignore", message="Invalid value for VR UI:")

load_dotenv("./.env", override=True)

# Paths
RAW_DIR = Path(os.getenv("DATASETS_DIR"), "raw/rsna-intracranial-hemorrhage-detection")
PRE_DIR = Path(os.getenv("DATASETS_DIR"), "pre/rsna-intracranial-hemorrhage-detection")
TRAIN_RAW_DIR = Path(RAW_DIR, "stage_2_train")

OUTPUT_DIR = Path("results/tables")


def get_dicom_spacing_fields(dcm_path: Path) -> dict:
    """Extract spacing-related fields from a single DICOM file.

    Returns dict with SliceThickness, SpacingBetweenSlices, ImagePositionPatient,
    PixelSpacing. Missing fields are returned as None.
    """
    ds = dcmread(dcm_path, stop_before_pixels=True)

    result = {
        "slice_thickness": None,
        "spacing_between_slices": None,
        "image_position": None,
        "pixel_spacing_row": None,
        "pixel_spacing_col": None,
    }

    if hasattr(ds, "SliceThickness"):
        result["slice_thickness"] = float(ds.SliceThickness)

    if hasattr(ds, "SpacingBetweenSlices"):
        result["spacing_between_slices"] = float(ds.SpacingBetweenSlices)

    if hasattr(ds, "ImagePositionPatient"):
        result["image_position"] = [float(v) for v in ds.ImagePositionPatient]

    if hasattr(ds, "PixelSpacing"):
        result["pixel_spacing_row"] = float(ds.PixelSpacing[0])
        result["pixel_spacing_col"] = float(ds.PixelSpacing[1])

    return result


def compute_patient_spacing(patient_df: pd.DataFrame) -> dict:
    """Compute inter-slice spacing statistics for a single patient.

    Args:
        patient_df: DataFrame rows for one patient, sorted by 'order'.

    Returns:
        Dict with spacing statistics for this patient.
    """
    patient_id = patient_df["PatientID"].iloc[0]
    n_slices = len(patient_df)

    # Read DICOM headers for all slices
    positions = []
    slice_thicknesses = []
    spacings_between = []
    pixel_spacings_row = []
    pixel_spacings_col = []

    for _, row in patient_df.iterrows():
        dcm_path = TRAIN_RAW_DIR / f"{row['SOPInstanceUID']}.dcm"
        if not dcm_path.exists():
            logger.warning(f"DICOM not found: {dcm_path}")
            continue

        fields = get_dicom_spacing_fields(dcm_path)

        if fields["image_position"] is not None:
            positions.append(fields["image_position"])

        if fields["slice_thickness"] is not None:
            slice_thicknesses.append(fields["slice_thickness"])

        if fields["spacing_between_slices"] is not None:
            spacings_between.append(fields["spacing_between_slices"])

        if fields["pixel_spacing_row"] is not None:
            pixel_spacings_row.append(fields["pixel_spacing_row"])
            pixel_spacings_col.append(fields["pixel_spacing_col"])

    # Compute actual inter-slice distances from ImagePositionPatient
    actual_spacings = []
    if len(positions) >= 2:
        positions_arr = np.array(positions)
        # Compute Euclidean distance between consecutive slices
        diffs = np.diff(positions_arr, axis=0)
        actual_spacings = np.sqrt(np.sum(diffs**2, axis=1)).tolist()

    # Nominal slice thickness from DICOM header
    nominal_thickness = None
    if slice_thicknesses:
        unique_thicknesses = list({round(t, 2) for t in slice_thicknesses})
        nominal_thickness = unique_thicknesses[0] if len(unique_thicknesses) == 1 else np.median(slice_thicknesses)

    # Nominal spacing between slices from DICOM header
    nominal_spacing = None
    if spacings_between:
        unique_spacings = list({round(s, 2) for s in spacings_between})
        nominal_spacing = unique_spacings[0] if len(unique_spacings) == 1 else np.median(spacings_between)

    # Pixel spacing
    pixel_row = np.median(pixel_spacings_row) if pixel_spacings_row else None
    pixel_col = np.median(pixel_spacings_col) if pixel_spacings_col else None

    # Uniformity check: is the spacing consistent?
    is_uniform = False
    if actual_spacings:
        spacing_std = np.std(actual_spacings)
        is_uniform = spacing_std < 0.1  # < 0.1 mm std considered uniform

    result = {
        "PatientID": patient_id,
        "n_slices": n_slices,
        "nominal_slice_thickness_mm": nominal_thickness,
        "nominal_spacing_between_mm": nominal_spacing,
        "computed_mean_spacing_mm": np.mean(actual_spacings) if actual_spacings else None,
        "computed_std_spacing_mm": np.std(actual_spacings) if actual_spacings else None,
        "computed_min_spacing_mm": np.min(actual_spacings) if actual_spacings else None,
        "computed_max_spacing_mm": np.max(actual_spacings) if actual_spacings else None,
        "is_uniform": is_uniform,
        "interpolation_gap_mm": 2 * np.mean(actual_spacings) if actual_spacings else None,
        "pixel_spacing_row_mm": pixel_row,
        "pixel_spacing_col_mm": pixel_col,
    }

    return result


def main() -> None:
    """Extract spacing information for all test patients."""
    # Load metadata
    df_path = PRE_DIR / "df.csv"
    logger.info(f"Loading metadata from {df_path}")
    df = pd.read_csv(df_path)

    # Handle legacy column name
    if "split" in df.columns and "stage" not in df.columns:
        df = df.rename(columns={"split": "stage"})

    # Filter test patients
    df_test = df[df["stage"] == "test"].copy()
    test_patients = sorted(df_test["PatientID"].unique())
    logger.info(f"Found {len(test_patients)} test patients, {len(df_test)} total slices")

    # Verify raw DICOMs are accessible
    sample_sop = df_test["SOPInstanceUID"].iloc[0]
    sample_dcm = TRAIN_RAW_DIR / f"{sample_sop}.dcm"
    if not sample_dcm.exists():
        raise FileNotFoundError(
            f"Raw DICOM not found at {sample_dcm}. "
            f"Ensure DATASETS_DIR is set correctly and raw DICOMs are at {TRAIN_RAW_DIR}"
        )

    # Process each patient
    results = []
    for i, patient_id in enumerate(test_patients):
        logger.info(f"[{i + 1}/{len(test_patients)}] Processing {patient_id}")
        patient_df = df_test[df_test["PatientID"] == patient_id].sort_values("order")
        result = compute_patient_spacing(patient_df)
        results.append(result)

    # Create per-patient DataFrame
    df_patients = pd.DataFrame(results)

    # Save per-patient results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patient_path = OUTPUT_DIR / "dicom_spacing_per_patient.csv"
    df_patients.to_csv(patient_path, index=False)
    logger.info(f"Saved per-patient spacing to {patient_path}")

    # Compute summary statistics
    summary = {
        "n_patients": len(df_patients),
        "n_total_slices": int(df_patients["n_slices"].sum()),
        # Nominal slice thickness
        "nominal_thickness_median_mm": df_patients["nominal_slice_thickness_mm"].median(),
        "nominal_thickness_min_mm": df_patients["nominal_slice_thickness_mm"].min(),
        "nominal_thickness_max_mm": df_patients["nominal_slice_thickness_mm"].max(),
        "nominal_thickness_unique": str(sorted(df_patients["nominal_slice_thickness_mm"].dropna().unique().tolist())),
        # Computed inter-slice spacing
        "computed_spacing_mean_mm": df_patients["computed_mean_spacing_mm"].mean(),
        "computed_spacing_std_across_patients_mm": df_patients["computed_mean_spacing_mm"].std(),
        "computed_spacing_min_mm": df_patients["computed_min_spacing_mm"].min(),
        "computed_spacing_max_mm": df_patients["computed_max_spacing_mm"].max(),
        # Uniformity
        "n_uniform_patients": int(df_patients["is_uniform"].sum()),
        "pct_uniform": float(df_patients["is_uniform"].mean() * 100),
        # Interpolation gap (distance between input slices = 2× spacing)
        "interpolation_gap_mean_mm": df_patients["interpolation_gap_mm"].mean(),
        "interpolation_gap_min_mm": df_patients["interpolation_gap_mm"].min(),
        "interpolation_gap_max_mm": df_patients["interpolation_gap_mm"].max(),
        # Pixel spacing
        "pixel_spacing_median_mm": df_patients["pixel_spacing_row_mm"].median(),
    }

    df_summary = pd.DataFrame([summary])
    summary_path = OUTPUT_DIR / "dicom_spacing_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DICOM SPACING SUMMARY (Test Set)")
    print("=" * 60)
    print(f"Patients: {summary['n_patients']}")
    print(f"Total slices: {summary['n_total_slices']}")
    print("\nNominal slice thickness (DICOM header):")
    print(f"  Median: {summary['nominal_thickness_median_mm']:.1f} mm")
    print(f"  Range: [{summary['nominal_thickness_min_mm']:.1f}, {summary['nominal_thickness_max_mm']:.1f}] mm")
    print(f"  Unique values: {summary['nominal_thickness_unique']}")
    print("\nComputed inter-slice spacing (from ImagePositionPatient):")
    print(f"  Mean: {summary['computed_spacing_mean_mm']:.2f} mm")
    print(f"  Std across patients: {summary['computed_spacing_std_across_patients_mm']:.2f} mm")
    print(f"  Range: [{summary['computed_spacing_min_mm']:.2f}, {summary['computed_spacing_max_mm']:.2f}] mm")
    print("\nUniformity (std < 0.1mm within patient):")
    print(f"  Uniform: {summary['n_uniform_patients']}/{summary['n_patients']} ({summary['pct_uniform']:.0f}%)")
    print("\nInterpolation gap (2x spacing, distance between input slices):")
    print(f"  Mean: {summary['interpolation_gap_mean_mm']:.1f} mm")
    print(f"  Range: [{summary['interpolation_gap_min_mm']:.1f}, {summary['interpolation_gap_max_mm']:.1f}] mm")
    print(f"\nPixel spacing: {summary['pixel_spacing_median_mm']:.3f} mm")
    print("=" * 60)


if __name__ == "__main__":
    import time

    t_start = time.time()
    main()
    t_end = time.time()
    logger.info(f"Completed in {t_end - t_start:.1f}s")
