import os
import warnings
from pathlib import Path

import pandas as pd
import tqdm
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pydicom import read_file

warnings.filterwarnings("ignore", message="Invalid value for VR UI:")

load_dotenv("../.env", override=True)

# Paths to the raw and preprocessed data
RAW_DIR = Path(os.getenv("DATASETS_DIR"), "raw/rsna-intracranial-hemorrhage-detection")
PRE_DIR = Path(os.getenv("DATASETS_DIR"), "pre/rsna-intracranial-hemorrhage-detection")
TRAIN_RAW_DIR = Path(RAW_DIR, "stage_2_train")

# Image Dataset parameters
WINDOW_CENTER = 44
WINDOW_WIDTH = 128
MAX_SIZE = 512
# HU window bounds: [-20, 107] (128 discrete values)
HU_MIN = WINDOW_CENTER - WINDOW_WIDTH // 2  # -20
HU_MAX = WINDOW_CENTER + WINDOW_WIDTH // 2 - 1  # 107
DS_NAME = f"1x{MAX_SIZE}x{MAX_SIZE}_{HU_MIN}_{HU_MAX}"
IMG_DATASET_DIR = Path(PRE_DIR, DS_NAME)


def main() -> None:
    """Main function to run the script."""  # noqa: D401
    logger.info(f"Reading {Path(PRE_DIR, 'df.csv')}")
    df = pd.read_csv(Path(PRE_DIR, "df.csv"))
    df = df[df["split"] == "test"]
    patient_ids = df["PatientID"].unique()
    for patient_id in tqdm.tqdm(patient_ids, total=len(patient_ids)):
        df_patient = df[df["PatientID"] == patient_id]
        df_patient = df_patient.sort_values(by="order")
        for _, row in tqdm.tqdm(df_patient.iterrows(), total=len(df_patient), leave=False):
            img_src_path = IMG_DATASET_DIR / f"{row['SOPInstanceUID']}.png"
            img_dst_dir = Path(f"output/raw-test/{DS_NAME}/{patient_id}/img")
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = img_dst_dir / f"{row['order']}-{row['SOPInstanceUID']}.png"
            if not dst_path.exists():
                try:
                    img = Image.open(img_src_path)
                    img.save(dst_path)
                except Exception as e:
                    logger.error(f"Error processing PNG {img_src_path}: {e}")
                    continue
            dcm_src_path = TRAIN_RAW_DIR / f"{row['SOPInstanceUID']}.dcm"
            dcm_dst_dir = Path(f"output/raw-test/{DS_NAME}/{patient_id}/dcm")
            dcm_dst_dir.mkdir(parents=True, exist_ok=True)
            dcm_dst_path = dcm_dst_dir / f"{row['order']}-{row['SOPInstanceUID']}.dcm"
            if not dcm_dst_path.exists():
                try:
                    dcm = read_file(dcm_src_path)
                    dcm.save_as(dcm_dst_path)
                except Exception as e:
                    logger.error(f"Error processing DCM {dcm_src_path}: {e}")
                    continue


if __name__ == "__main__":
    import time

    t_start = time.time()
    logger.info("Starting preprocessing")
    main()
    logger.info("Finished preprocessing")
    t_end = time.time()
    logger.info(f"Time taken: {t_end - t_start:.2f} seconds")
