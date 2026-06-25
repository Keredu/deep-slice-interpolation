"""Export test-patient slice interpolation outputs as PNGs + a DICOM series.

This script:
- Loads a trained interpolation model checkpoint (e.g. epoch 52 `weights.pth`)
- Iterates over *test* patients (same split logic as `TwoToOneSliceTestDataset`)
- For each patient, generates an interpolated slice for every pair of consecutive slices
- Writes:
  - `<out>/<patient_id>/viz/target_is_real/*.png` (triplet-style debug tiles)
  - `<out>/<patient_id>/viz/target_is_interpolated/*.png` (pair-style debug tiles)
  - `<out>/<patient_id>/viz/series/*.png` (one PNG per output slice: real + AI)
  - `<out>/<patient_id>/dcm/*.dcm` (one SC DICOM per output slice: real + AI)
  - `<out>/<patient_id>/info.json` (metadata describing which slices are real vs AI)

Notes
-----
- The model operates on *preprocessed PNGs* (windowed) in [0,1]. The exported DICOMs
  are **Secondary Capture** (8-bit MONOCHROME2) for portability.
- Requires `DATASETS_DIR` to locate the raw RSNA DICOMs (for patient/study metadata).
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from pydicom.valuerep import format_number_as_ds

from phd.config_io import resolve_config_path
from phd.models.setup_model import setup_model
from phd.viz import predict_via_patch_reconstruction, save_prediction_visualization

# RSNA uses `ID_...` strings for fields stored with VR=UI (UID). pydicom warns a lot.
warnings.filterwarnings("ignore", message="Invalid value for VR UI:*")

# HU windowing parameters used during preprocessing.
# The PNG dataset contains images windowed to this HU range, then normalized to [0, 255].
# Export supports two metadata modes:
# - display (default): keep pixel values display-ready (match PNG look in viewers)
# - hu: attach linear HU rescale tags for CT-style value interpretation
HU_WINDOW_CENTER = 44
HU_WINDOW_WIDTH = 128
HU_MIN = HU_WINDOW_CENTER - HU_WINDOW_WIDTH // 2  # -20 HU
HU_MAX = HU_WINDOW_CENTER + HU_WINDOW_WIDTH // 2 - 1  # 107 HU (128 discrete values: -20 to 107)


@dataclass(frozen=True)
class Paths:
    """Paths to dataset directories."""

    pre_dir: Path
    raw_dcm_dir: Path
    img_dir: Path


def _now_dicom_date_time() -> tuple[str, str]:
    dt = datetime.now(UTC).astimezone()
    return dt.strftime("%Y%m%d"), dt.strftime("%H%M%S")


def _load_df_test(pre_dir: Path) -> pd.DataFrame:
    df_path = pre_dir / "df.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing {df_path}")
    df = pd.read_csv(df_path)
    if "stage" not in df.columns and "split" in df.columns:
        df = df.rename(columns={"split": "stage"})
    if "stage" not in df.columns:
        raise ValueError(f"{df_path} missing `stage` or `split` column")
    df = df[df["stage"] == "test"].copy()
    if df.empty:
        raise ValueError(f"No test rows found in {df_path}")
    for col in ["PatientID", "StudyInstanceUID", "SOPInstanceUID", "order"]:
        if col not in df.columns:
            raise ValueError(f"{df_path} missing required column `{col}`")
    return df


def _load_uint8_png(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as pil_img:
        grayscale_img = pil_img.convert("L")
        arr = np.array(grayscale_img, dtype=np.uint8)
    return arr


def _to_float01(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _infer_in_batches(
    model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int
) -> torch.Tensor:
    """Run test-time inference via 13-patch reconstruction (9 center + 4 corner)."""
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(inputs), batch_size):
            batch_inputs = inputs[j : j + batch_size].to(device)
            batch_outputs = predict_via_patch_reconstruction(
                model=model,
                batch_inputs=batch_inputs,
                device=device,
            ).detach().cpu()
            outputs.append(batch_outputs)
            del batch_inputs
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def _make_sc_dicom(
    template: FileDataset,
    pixel_u8: np.ndarray,
    *,
    sop_instance_uid: str,
    series_instance_uid: str,
    instance_number: int,
    image_position_patient: list[float] | None,
    derivation_description: str,
    intensity_mode: str = "display",
) -> FileDataset:
    if pixel_u8.ndim != 2:
        raise ValueError(f"Expected 2D grayscale pixel array, got shape={pixel_u8.shape}")

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Patient/study identity
    for tag in [
        "PatientID",
        "PatientName",
        "PatientSex",
        "PatientAge",
        "PatientBirthDate",
        "StudyInstanceUID",
        "StudyID",
        "StudyDate",
        "StudyTime",
        "AccessionNumber",
        "ReferringPhysicianName",
        # Useful for consistent viewer grouping/orientation
        "FrameOfReferenceUID",
        "BodyPartExamined",
        "StudyDescription",
        "InstitutionName",
    ]:
        if hasattr(template, tag):
            setattr(ds, tag, getattr(template, tag))

    ds.Modality = "OT"
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesDescription = "CT slice interpolation export (real+AI as Secondary Capture)"

    ds.InstanceNumber = int(instance_number)
    if image_position_patient is not None:
        ds.ImagePositionPatient = [float(x) for x in image_position_patient]
    if hasattr(template, "ImageOrientationPatient"):
        ds.ImageOrientationPatient = template.ImageOrientationPatient
    if hasattr(template, "PixelSpacing"):
        ds.PixelSpacing = template.PixelSpacing
    if hasattr(template, "SliceThickness"):
        ds.SliceThickness = template.SliceThickness
    if hasattr(template, "SpacingBetweenSlices"):
        ds.SpacingBetweenSlices = template.SpacingBetweenSlices

    content_date, content_time = _now_dicom_date_time()
    ds.ContentDate = content_date
    ds.ContentTime = content_time
    ds.DerivationDescription = derivation_description

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = int(pixel_u8.shape[0]), int(pixel_u8.shape[1])
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    if intensity_mode == "hu":
        # Optional compatibility mode: expose the 8-bit data as HU using a linear
        # modality transform so viewers can apply CT-like window presets.
        ds.RescaleIntercept = float(HU_MIN)  # -20
        ds.RescaleSlope = float(HU_MAX - HU_MIN) / 255.0  # 127/255
        ds.RescaleType = "HU"
        ds.WindowCenter = float(HU_WINDOW_CENTER)  # 44 HU
        ds.WindowWidth = float(HU_WINDOW_WIDTH)  # 128 HU
    else:
        # Default display mode: data is already windowed to [-20, 107] and mapped
        # to 8-bit [0, 255] during preprocessing. Avoid additional HU rescale tags
        # so viewers render this exactly like the PNGs.
        ds.WindowCenter = 127.5
        ds.WindowWidth = 255.0
        ds.PresentationLUTShape = "IDENTITY"
        ds.VOILUTFunction = "LINEAR"

    ds.PixelData = pixel_u8.tobytes()

    return ds


def _make_ct_dicom(
    template: FileDataset,
    pixel_hu_i16: np.ndarray,
    *,
    sop_instance_uid: str,
    series_instance_uid: str,
    series_number: int,
    instance_number: int,
    image_position_patient: list[float] | None,
    derivation_description: str,
    slice_thickness_mm: float | None = None,
    spacing_between_slices_mm: float | None = None,
) -> FileDataset:
    """Emit a CT Image Storage dataset (16-bit signed HU) suitable for stackable viewing.

    Built by deep-copying the input CT template and replacing only: pixel data,
    identifying UIDs, InstanceNumber, IPP/SliceLocation, rescale tags, and a
    couple of geometry tags. Keeping the rest of the template (gantry tilt,
    ImageType detail, SourceImageSequence-adjacent tags, ContributingEquipmentSequence,
    etc.) gives Weasis a complete CT IOD, which is what it needs to stack a
    series as a scrollable 3D volume.
    """
    if pixel_hu_i16.ndim != 2:
        raise ValueError(f"Expected 2D HU array, got shape={pixel_hu_i16.shape}")
    if pixel_hu_i16.dtype != np.int16:
        raise ValueError(f"Expected int16 HU array, got dtype={pixel_hu_i16.dtype}")

    ds = copy.deepcopy(template)

    # Keep the template's TransferSyntaxUID (and matching VR encoding flags)
    # so the resulting file byte layout mirrors the input. Only mint a new
    # MediaStorageSOPInstanceUID for the new SOP.
    ds.file_meta = copy.deepcopy(template.file_meta)
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.preamble = b"\0" * 128

    ds.Modality = "CT"
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = int(series_number)
    ds.InstanceNumber = int(instance_number)
    # Preserve the template's 4-element ImageType when present (e.g.
    # ['DERIVED', 'PRIMARY', 'AXIAL', 'NONE']); only override the second
    # value so we don't switch a PRIMARY image to SECONDARY (some viewers
    # treat SECONDARY images as non-stackable snapshots).
    tmpl_image_type = list(getattr(template, "ImageType", []) or [])
    if len(tmpl_image_type) >= 2:
        tmpl_image_type[0] = "DERIVED"
        ds.ImageType = tmpl_image_type
    else:
        ds.ImageType = ["DERIVED", "PRIMARY", "AXIAL"]

    if image_position_patient is not None:
        ipp = [format_number_as_ds(float(x)) for x in image_position_patient]
        ds.ImagePositionPatient = ipp
        ds.SliceLocation = format_number_as_ds(float(image_position_patient[2]))
    if slice_thickness_mm is not None:
        ds.SliceThickness = format_number_as_ds(float(slice_thickness_mm))
    if spacing_between_slices_mm is not None:
        ds.SpacingBetweenSlices = format_number_as_ds(float(spacing_between_slices_mm))

    # Generate a fresh IrradiationEventUID per synthesized slice — sharing one
    # across multiple slices (as a naive deep-copy would) confuses dose-tracking
    # viewers and flags the series as multi-frame of a single exposure.
    ds.IrradiationEventUID = generate_uid()

    content_date, content_time = _now_dicom_date_time()
    ds.ContentDate = content_date
    ds.ContentTime = content_time
    ds.DerivationDescription = derivation_description

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = int(pixel_hu_i16.shape[0]), int(pixel_hu_i16.shape[1])
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    for tag in ("SmallestImagePixelValue", "LargestImagePixelValue", "PixelPaddingValue"):
        if tag in ds:
            del ds[tag]

    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.RescaleType = "HU"
    ds.WindowCenter = float(HU_WINDOW_CENTER)
    ds.WindowWidth = float(HU_WINDOW_WIDTH)

    ds.PixelData = pixel_hu_i16.tobytes()

    return ds


def _save_prediction_visualization_compat(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    save_dir: Path,
) -> None:
    """Call `save_prediction_visualization` across old/new function signatures."""
    params = inspect.signature(save_prediction_visualization).parameters
    if "predictions" in params:
        save_prediction_visualization(
            inputs=inputs,
            targets=targets,
            predictions=predictions,
            save_dir=save_dir,
        )
        return

    # New API: requires 3 prediction tensors (center/corner/composite).
    save_prediction_visualization(
        inputs=inputs,
        targets=targets,
        center_predictions=predictions,
        corner_predictions=predictions,
        composite_predictions=predictions,
        save_dir=save_dir,
    )


def _read_experiment_config(experiment_dir: Path) -> dict[str, Any]:
    cfg_path = experiment_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _resolve_paths_from_env_and_config(config: dict[str, Any]) -> Paths:
    datasets_dir = os.getenv("DATASETS_DIR")
    if not datasets_dir:
        # Fallback: infer datasets root from the configured PNG dataset path:
        #   <DATASETS_DIR>/pre/rsna-intracranial-hemorrhage-detection/<DS_NAME>
        img_dir_guess = resolve_config_path(config["data_path"]).resolve()
        parts = list(img_dir_guess.parts)
        try:
            pre_idx = parts.index("pre")
        except ValueError as e:
            raise RuntimeError(
                "DATASETS_DIR environment variable is not set and could not infer it from config['data_path'] "
                f"({img_dir_guess}). Expected to contain '/pre/rsna-intracranial-hemorrhage-detection/'.",
            ) from e
        datasets_dir = str(Path(*parts[:pre_idx]).resolve())
        logger.warning(f"DATASETS_DIR not set; inferred DATASETS_DIR={datasets_dir} from data_path={img_dir_guess}")

    pre_dir = Path(datasets_dir) / "pre/rsna-intracranial-hemorrhage-detection"
    raw_dcm_dir = Path(datasets_dir) / "raw/rsna-intracranial-hemorrhage-detection/stage_2_train"
    img_dir = resolve_config_path(config["data_path"])

    if not pre_dir.exists():
        raise FileNotFoundError(f"Missing preprocessed metadata directory: {pre_dir}")
    if not raw_dcm_dir.exists():
        raise FileNotFoundError(f"Missing raw DICOM directory: {raw_dcm_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing preprocessed image directory: {img_dir}")

    return Paths(pre_dir=pre_dir, raw_dcm_dir=raw_dcm_dir, img_dir=img_dir)


def _load_model_for_inference(config: dict[str, Any], weights_path: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and "type" in model_cfg:
        model_type = model_cfg["type"]
        encoder_name = model_cfg.get("encoder_name")
    else:
        # Backward compatibility with older experiment configs
        model_type = config.get("model_type")
        encoder_name = config.get("encoder_name")

    if not model_type:
        raise KeyError(
            "Missing model configuration. Expected either config['model']['type'] "
            "or legacy config['model_type'] in config.json."
        )

    model = setup_model(
        in_channels=2,
        out_channels=1,
        pretrained=False,  # weights loaded from checkpoint; do not pull encoder weights
        model_type=model_type,
        encoder_name=encoder_name,
    ).to(device)

    checkpoint = torch.load(weights_path, weights_only=False, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint missing `model_state_dict`: {weights_path}")
    state_dict = checkpoint["model_state_dict"]
    # Training may save compiled-module keys prefixed with `_orig_mod.`.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _find_highest_epoch_with_weights(experiment_dir: Path) -> int | None:
    epochs_dir = experiment_dir / "epochs"
    if not epochs_dir.exists():
        return None

    epoch_numbers: list[int] = []
    for epoch_dir in epochs_dir.iterdir():
        if not epoch_dir.is_dir():
            continue
        try:
            epoch_num = int(epoch_dir.name)
        except ValueError:
            continue
        if (epoch_dir / "weights.pth").exists():
            epoch_numbers.append(epoch_num)

    if not epoch_numbers:
        return None

    return max(epoch_numbers)


def _resolve_weights_path(experiment_dir: Path, requested_epoch: int | None) -> tuple[Path, int | None]:
    if requested_epoch is not None:
        weights_path = experiment_dir / "epochs" / str(requested_epoch) / "weights.pth"
        if weights_path.exists():
            return weights_path, requested_epoch
    else:
        auto_epoch = _find_highest_epoch_with_weights(experiment_dir)
        if auto_epoch is not None:
            weights_path = experiment_dir / "epochs" / str(auto_epoch) / "weights.pth"
            return weights_path, auto_epoch

    # Some runs only keep `latest_epoch.pth` in the experiment root
    alt = experiment_dir / "latest_epoch.pth"
    if alt.exists():
        return alt, None

    if requested_epoch is not None:
        missing = experiment_dir / "epochs" / str(requested_epoch) / "weights.pth"
        raise FileNotFoundError(f"Missing weights: {missing} (and no {alt})")
    raise FileNotFoundError(f"Missing weights under {experiment_dir / 'epochs'}/*/weights.pth (and no {alt})")


def _resolve_experiment_subpath(experiment_dir: Path) -> Path:
    """Return experiment path relative to `experiments/` when possible."""
    parts = experiment_dir.resolve().parts
    if "experiments" in parts:
        idx = len(parts) - 1 - parts[::-1].index("experiments")
        rel_parts = parts[idx + 1 :]
        if rel_parts:
            return Path(*rel_parts)
    logger.warning(f"Could not resolve path relative to 'experiments/': {experiment_dir}; using experiment name only")
    return Path(experiment_dir.name)


def export_patient(
    *,
    patient_id: str,
    patient_df: pd.DataFrame,
    paths: Paths,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
    patient_idx: int | None = None,
    patient_total: int | None = None,
    dicom_intensity_mode: str = "display",
) -> None:
    """Export interpolated DICOM series for a single patient."""
    patient_out = out_dir / patient_id
    viz_dir = patient_out / "viz"
    # DICOM outputs:
    # - dcm_real: original real CT slices copied as-is (for reference)
    # - dcm_mixed: *single* synthetic series containing real+ai interleaved, all belonging to the same SeriesInstanceUID
    dcm_real_dir = patient_out / "dcm_real"
    dcm_mixed_dir = patient_out / "dcm_mixed"
    viz_real_dir = viz_dir / "target_is_real"
    viz_interp_dir = viz_dir / "target_is_interpolated"
    viz_series_dir = viz_dir / "series"
    for d in [viz_real_dir, viz_interp_dir, viz_series_dir, dcm_real_dir, dcm_mixed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    patient_df = patient_df.sort_values(by="order").reset_index(drop=True)
    sops: list[str] = patient_df["SOPInstanceUID"].tolist()
    if len(sops) < 2:
        logger.warning(f"{patient_id}: skipping, <2 slices")
        return

    info_path = patient_out / "info.json"
    prefix = f"[{patient_idx}/{patient_total}] " if patient_idx is not None and patient_total is not None else ""
    logger.info(f"{prefix}{patient_id}: exporting {len(sops)} real slices -> {patient_out}")

    # Load raw template (for tags/geometry) and prepare raw DICOM reader
    import pydicom

    template_path = paths.raw_dcm_dir / f"{sops[0]}.dcm"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing raw DICOM for template: {template_path}")
    template_ds = pydicom.dcmread(template_path, force=True)

    # Load all real slices from PNG dataset (uint8 + float01 for inference)
    real_u8: list[np.ndarray] = []
    real_f: list[np.ndarray] = []
    logger.debug(f"{patient_id}: loading {len(sops)} PNGs")
    for sop in sops:
        arr_u8 = _load_uint8_png(paths.img_dir / f"{sop}.png")
        real_u8.append(arr_u8)
        real_f.append(_to_float01(arr_u8))

    h, w = real_u8[0].shape
    if any(a.shape != (h, w) for a in real_u8):
        raise ValueError(f"{patient_id}: inconsistent image shapes in PNG dataset")

    real_stack = np.stack(real_f, axis=0)  # (N,H,W)

    # --- Debug viz: target_is_real (triplets) ---
    if len(real_stack) >= 3:
        inputs_trip = np.stack([real_stack[:-2], real_stack[2:]], axis=1)  # (N-2,2,H,W)
        targets_trip = real_stack[1:-1, None, :, :]  # (N-2,1,H,W)
        inputs_trip_t = torch.from_numpy(inputs_trip)
        targets_trip_t = torch.from_numpy(targets_trip)
        preds_trip = _infer_in_batches(model, inputs_trip_t, device=device, batch_size=batch_size).clamp(0.0, 1.0)
        _save_prediction_visualization_compat(
            inputs=inputs_trip_t,
            targets=targets_trip_t,
            predictions=preds_trip,
            save_dir=viz_real_dir,
        )

    # --- Interpolation: every consecutive pair ---
    inputs_pairs = np.stack([real_stack[:-1], real_stack[1:]], axis=1)  # (N-1,2,H,W)
    targets_pairs = np.mean(inputs_pairs, axis=1, keepdims=True)  # (N-1,1,H,W) (baseline)
    inputs_pairs_t = torch.from_numpy(inputs_pairs)
    targets_pairs_t = torch.from_numpy(targets_pairs)
    preds_pairs = _infer_in_batches(model, inputs_pairs_t, device=device, batch_size=batch_size).clamp(0.0, 1.0)

    _save_prediction_visualization_compat(
        inputs=inputs_pairs_t,
        targets=targets_pairs_t,
        predictions=preds_pairs,
        save_dir=viz_interp_dir,
    )

    preds_pairs_u8 = (preds_pairs.numpy()[:, 0] * 255.0).round().clip(0, 255).astype(np.uint8)  # (N-1,H,W)

    # --- Write per-slice PNG series and DICOM series ---
    series_uid = generate_uid()
    out_slices: list[dict[str, Any]] = []

    # Attempt to reuse ImagePositionPatient for real slices; midpoint for AI slices
    positions: list[list[float] | None] = []
    logger.debug(f"{patient_id}: reading positions from {len(sops)} DICOMs")
    for sop in sops:
        dcm_path = paths.raw_dcm_dir / f"{sop}.dcm"
        if dcm_path.exists():
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
            if hasattr(ds, "ImagePositionPatient"):
                positions.append([float(x) for x in ds.ImagePositionPatient])
            else:
                positions.append(None)
        else:
            positions.append(None)

    def write_one(index: int, kind: str, pixel_u8: np.ndarray, pos: list[float] | None, meta: dict[str, Any]) -> None:
        png_name = f"{index:04d}_{kind}.png"
        dcm_name_mixed = f"{index:04d}_{kind}.dcm"
        Image.fromarray(pixel_u8).save(viz_series_dir / png_name)

        dcm_rel_mixed = f"dcm_mixed/{dcm_name_mixed}"
        mixed_sop_uid: str

        if kind == "real":
            # Copy the original raw DICOM slice (preserve original metadata/pixels)
            src_sop = meta.get("original_sop_instance_uid")
            if not src_sop:
                raise ValueError("Missing `original_sop_instance_uid` for real slice export")
            original_order = meta.get("original_order")
            if original_order is None:
                raise ValueError("Missing `original_order` for real slice export")
            src_path = paths.raw_dcm_dir / f"{src_sop}.dcm"
            if not src_path.exists():
                raise FileNotFoundError(f"Missing raw DICOM: {src_path}")
            ds_raw = pydicom.dcmread(src_path, force=True)
            raw_sop_uid = str(getattr(ds_raw, "SOPInstanceUID", src_sop))
            # Use original order for deterministic sorting in viewers.
            # Set InstanceNumber explicitly (RSNA DICOMs lack it, causing random viewer order).
            raw_name = f"{int(original_order):04d}.dcm"
            ds_raw.InstanceNumber = int(original_order) + 1
            ds_raw.save_as(dcm_real_dir / raw_name)

            # ALSO write this real slice into the mixed (single-series) directory as Secondary Capture
            mixed_sop_uid = generate_uid()
            ds_mixed = _make_sc_dicom(
                template=template_ds,
                pixel_u8=pixel_u8,
                sop_instance_uid=mixed_sop_uid,
                series_instance_uid=series_uid,
                instance_number=index + 1,
                image_position_patient=pos,
                derivation_description=meta.get("derivation_description", ""),
                intensity_mode=dicom_intensity_mode,
            )
            ds_mixed.save_as(dcm_mixed_dir / dcm_name_mixed)
            meta = {
                **meta,
                "dicom_real_source": "raw",
                "dicom_real_path": f"dcm_real/{raw_name}",
                "dicom_real_sop_instance_uid": raw_sop_uid,
                "dicom_mixed_source": "secondary_capture",
                "dicom_mixed_path": dcm_rel_mixed,
                "dicom_mixed_sop_instance_uid": mixed_sop_uid,
                "dicom_mixed_series_instance_uid": series_uid,
            }
        else:
            # Write an SC DICOM for AI slice (into mixed directory)
            if pos is None:
                logger.warning(f"{patient_id}: missing ImagePositionPatient for AI slice at mixed index={index}")
            mixed_sop_uid = generate_uid()
            ds_out = _make_sc_dicom(
                template=template_ds,
                pixel_u8=pixel_u8,
                sop_instance_uid=mixed_sop_uid,
                series_instance_uid=series_uid,
                instance_number=index + 1,
                image_position_patient=pos,
                derivation_description=meta.get("derivation_description", ""),
                intensity_mode=dicom_intensity_mode,
            )
            ds_out.save_as(dcm_mixed_dir / dcm_name_mixed)
            meta = {
                **meta,
                "dicom_mixed_source": "ai_secondary_capture",
                "dicom_mixed_path": dcm_rel_mixed,
                "dicom_mixed_sop_instance_uid": mixed_sop_uid,
                "dicom_mixed_series_instance_uid": series_uid,
            }

        out_slices.append(
            {
                "index": index,
                "kind": kind,
                "png": f"viz/series/{png_name}",
                **meta,
            },
        )

    out_idx = 0
    logger.debug(f"{patient_id}: writing {len(sops) - 1} pairs")
    for i in range(len(sops) - 1):
        # Real slice i
        write_one(
            index=out_idx,
            kind="real",
            pixel_u8=real_u8[i],
            pos=positions[i],
            meta={
                "original_sop_instance_uid": sops[i],
                "original_order": int(patient_df.loc[i, "order"]),
                "image_position_patient": positions[i],
                "derivation_description": "Real slice (from windowed PNG dataset) exported as Secondary Capture",
            },
        )
        out_idx += 1

        # AI slice between i and i+1
        pos_mid: list[float] | None = None
        if positions[i] is not None and positions[i + 1] is not None:
            pos_mid = [float((a + b) / 2.0) for a, b in zip(positions[i], positions[i + 1], strict=False)]
        else:
            logger.warning(
                f"{patient_id}: cannot compute midpoint ImagePositionPatient for pair orders "
                f"{int(patient_df.loc[i, 'order'])}->{int(patient_df.loc[i + 1, 'order'])} "
                f"(missing position on left/right)",
            )
        write_one(
            index=out_idx,
            kind="ai",
            pixel_u8=preds_pairs_u8[i],
            pos=pos_mid,
            meta={
                "left_sop_instance_uid": sops[i],
                "right_sop_instance_uid": sops[i + 1],
                "between_orders": [int(patient_df.loc[i, "order"]), int(patient_df.loc[i + 1, "order"])],
                "image_position_patient": pos_mid,
                "derivation_description": (
                    "AI interpolated slice between consecutive real slices (model output) exported as Secondary Capture"
                ),
            },
        )
        out_idx += 1

    # Last real slice
    write_one(
        index=out_idx,
        kind="real",
        pixel_u8=real_u8[-1],
        pos=positions[-1],
        meta={
            "original_sop_instance_uid": sops[-1],
            "original_order": int(patient_df.loc[len(sops) - 1, "order"]),
            "image_position_patient": positions[-1],
            "derivation_description": "Real slice (from windowed PNG dataset) exported as Secondary Capture",
        },
    )

    info = {
        "patient_id": patient_id,
        "study_instance_uid": patient_df["StudyInstanceUID"].iloc[0],
        "num_real_slices": len(sops),
        "num_ai_slices": len(sops) - 1,
        "num_total_slices": len(out_slices),
        "dicom": {
            "dcm_real_dir": "dcm_real/",
            "dcm_mixed_dir": "dcm_mixed/",
            "mixed_series_instance_uid": series_uid,
        },
        "output_slices": out_slices,
        "viz": {
            "target_is_real": "viz/target_is_real/",
            "target_is_interpolated": "viz/target_is_interpolated/",
            "series": "viz/series/",
        },
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    n_real, n_ai = len(sops), len(sops) - 1
    logger.info(f"{prefix}{patient_id}: done. wrote {len(out_slices)} slices (real={n_real}, ai={n_ai})")


def main() -> None:
    """Export test set interpolations as DICOM series."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("./experiments/train_nn1_cropped/unet_tu-tf_efficientnetv2_s_ssim_lr_0.003_wd_0.01_full_dataset"),
        help="Experiment directory containing config.json and epochs/<epoch>/weights.pth",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch folder to use. If omitted, uses the highest epoch with weights in experiments/<name>/epochs/",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output/test_interpolated_export"),
        help="Output root directory; experiment path under `experiments/` is mirrored beneath this root",
    )
    parser.add_argument("--limit-patients", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--dicom-intensity-mode",
        choices=["display", "hu"],
        default="display",
        help=(
            "How to encode pixel intensity metadata in exported mixed DICOMs. "
            "'display' (default) keeps images display-ready and matches viz PNGs; "
            "'hu' writes RescaleSlope/Intercept to expose HU mapping."
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    exp_dir: Path = args.experiment_dir
    config = _read_experiment_config(exp_dir)
    paths = _resolve_paths_from_env_and_config(config)
    df = _load_df_test(paths.pre_dir)

    weights_path, resolved_epoch = _resolve_weights_path(exp_dir, args.epoch)
    if args.epoch is None and resolved_epoch is not None:
        logger.info(f"--epoch not specified; using highest available epoch with weights: {resolved_epoch}")
    elif resolved_epoch is None:
        logger.warning("--epoch could not be resolved to a numbered folder; using latest_epoch.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device={device}")
    model = _load_model_for_inference(config=config, weights_path=weights_path, device=device)

    epoch_label = f"epoch_{resolved_epoch}" if resolved_epoch is not None else "epoch_latest"
    exp_subpath = _resolve_experiment_subpath(exp_dir)
    out_dir = args.output_dir / exp_subpath / epoch_label
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted(df["PatientID"].unique().tolist())
    if args.limit_patients and args.limit_patients > 0:
        patient_ids = patient_ids[: args.limit_patients]

    logger.info(
        "Export run:\n"
        f"- experiment_dir={exp_dir}\n"
        f"- weights={weights_path}\n"
        f"- img_dir={paths.img_dir}\n"
        f"- raw_dcm_dir={paths.raw_dcm_dir}\n"
        f"- out_dir={out_dir}\n"
        f"- patients={len(patient_ids)}\n"
        f"- batch_size={args.batch_size}\n",
    )

    n_patients = len(patient_ids)
    for i, pid in enumerate(patient_ids, start=1):
        logger.info(f"Exporting patient {i}/{n_patients}: {pid}")
        pdf = df[df["PatientID"] == pid]
        export_patient(
            patient_id=pid,
            patient_df=pdf,
            paths=paths,
            model=model,
            device=device,
            batch_size=args.batch_size,
            out_dir=out_dir,
            patient_idx=i,
            patient_total=n_patients,
            dicom_intensity_mode=args.dicom_intensity_mode,
        )


if __name__ == "__main__":
    main()
