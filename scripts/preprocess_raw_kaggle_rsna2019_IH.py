import multiprocessing as mp
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import FileDataset

warnings.filterwarnings("ignore", message="Invalid value for VR UI:")

load_dotenv("./.env", override=True)

if os.getenv("DATASETS_DIR") is None:
    raise ValueError("DATASETS_DIR is not set")

N_THREADS = 14
SEED = 42
RGN = np.random.default_rng(SEED)

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
HU_MAX = WINDOW_CENTER + WINDOW_WIDTH // 2 - 1  # 107 (not 108)
DS_NAME = f"1x{MAX_SIZE}x{MAX_SIZE}_{HU_MIN}_{HU_MAX}"
IMG_DATASET_DIR = Path(PRE_DIR, DS_NAME)

# Create directories if they don't exist
PRE_DIR.mkdir(parents=True, exist_ok=True)
IMG_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Log const configuration
const_conf_str = (
    "Configuration:"
    f"\nN_THREADS: {N_THREADS}"
    f"\nSEED: {SEED}"
    f"\nRAW_DIR: {RAW_DIR}"
    f"\nPRE_DIR: {PRE_DIR}"
    f"\nTRAIN_RAW_DIR: {TRAIN_RAW_DIR}"
    f"\nWINDOW_CENTER: {WINDOW_CENTER}"
    f"\nWINDOW_WIDTH: {WINDOW_WIDTH}"
    f"\nMAX_SIZE: {MAX_SIZE}"
    f"\nIMG_DATASET_DIR: {IMG_DATASET_DIR}"
)
logger.info(const_conf_str)


# Custom exception for script errors
class ScriptError(Exception):
    """Custom exception for script errors."""


def flatten_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the subtypes in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with the subtypes.

    Returns:
        pd.DataFrame: DataFrame with the subtypes flattened.

    Raises:
        ScriptError: If the sum of the labels is not equal after flattening the subtypes.
    """
    # Split 'ID' into 'Image_ID' and 'Subtype'
    df[["Image_ID", "Subtype"]] = df["ID"].str.extract(r"^(ID_[^_]+)_([^_]+)$")
    df.head(10)

    sum_check_0 = df.groupby("Subtype")["Label"].sum()

    df_pivot = df.pivot_table(index="Image_ID", columns="Subtype", values="Label", aggfunc="max").reset_index()
    df_pivot = df_pivot.rename(columns={"Image_ID": "ID"})

    # Remove the multi-level index
    df_pivot.columns.name = None  # Remove the 'Subtype' name for the columns
    df_pivot = df_pivot.reset_index(drop=True)  # Reset the index to make 'ID' a column
    df_pivot = df_pivot.set_index("ID")  # Set 'ID' as the index again
    df_pivot.head(10)

    # Sum values per column
    sum_check_1 = df_pivot.sum()

    # Check if sum_check_0 and sum_check_1 are equal
    if not sum_check_0.equals(sum_check_1):
        raise ScriptError("The sum of the labels is not equal after flattening the subtypes")

    return df_pivot


def split_df_into_n_parts(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Split a DataFrame into n parts.

    Args:
        df (pd.DataFrame): DataFrame to split.
        n (int): Number of parts to split the DataFrame into.

    Returns:
        list: List of n parts.

    Raises:
        ScriptError: If the DataFrames are not split evenly.
    """
    len_df = len(df)
    len_mini_df = len_df // n
    mini_dfs = []
    for i in range(n - 1):
        start = i * len_mini_df
        end = (i + 1) * len_mini_df
        mini_df = df.iloc[start:end]
        mini_dfs.append(mini_df)
    start = (n - 1) * len_mini_df
    mini_df = df.iloc[start:]
    mini_dfs.append(mini_df)
    len_mini_dfs = [len(mini_df) for mini_df in mini_dfs]
    if max(len_mini_dfs) - min(len_mini_dfs) > n:
        raise ScriptError(f"The DataFrames are not split evenly: {len_mini_dfs}")
    return mini_dfs


def split_list_into_n_parts(lst: list, n: int) -> list[list]:
    """Split a list into n parts.

    Args:
        lst (list): List to split.
        n (int): Number of parts to split the list into.

    Returns:
        list: List of n parts.

    Raises:
        ScriptError: If the lists are not split evenly.
    """
    len_lst = len(lst)
    len_mini_lst = len_lst // n
    mini_lsts = []
    for i in range(n - 1):
        start = i * len_mini_lst
        end = (i + 1) * len_mini_lst
        mini_lst = lst[start:end]
        mini_lsts.append(mini_lst)
    start = (n - 1) * len_mini_lst
    mini_lst = lst[start:]
    mini_lsts.append(mini_lst)
    len_mini_lsts = [len(mini_lst) for mini_lst in mini_lsts]
    if max(len_mini_lsts) - min(len_mini_lsts) > n:
        raise ScriptError(f"The lists are not split evenly: {len_mini_lsts}")
    return mini_lsts


def add_ids_to_df_mp_func(mini_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Process a mini DataFrame."""
    l = []  # noqa: E741
    for _, row in tqdm.tqdm(mini_df.iterrows(), total=len(mini_df)):
        ds = dcmread(Path(TRAIN_RAW_DIR, f"{row.name}.dcm"))
        l.append((ds.PatientID, ds.StudyInstanceUID, ds.SeriesInstanceUID))
    return l


def add_ids_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add PatientID, StudyInstanceUID and SeriesInstanceUID to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with the DICOM files data.

    Returns:
        pd.DataFrame: DataFrame with the PatientID, StudyInstanceUID and SeriesInstanceUID.
    """
    mini_dfs = split_df_into_n_parts(df, N_THREADS)
    with mp.Pool(N_THREADS) as p:
        res = p.map(add_ids_to_df_mp_func, mini_dfs)

    d = {"PatientID": [], "StudyInstanceUID": [], "SeriesInstanceUID": []}
    for l in tqdm.tqdm(res, total=N_THREADS, desc="Merging results"):  # noqa: E741
        for t in tqdm.tqdm(l, leave=False, desc="Processing mini DataFrame"):
            d["PatientID"].append(t[0])
            d["StudyInstanceUID"].append(t[1])
            d["SeriesInstanceUID"].append(t[2])

    # Create new columns in the DataFrame
    df["PatientID"] = d["PatientID"]
    df["StudyInstanceUID"] = d["StudyInstanceUID"]
    df["SeriesInstanceUID"] = d["SeriesInstanceUID"]
    df = df.reset_index()
    df = df.rename(columns={"ID": "SOPInstanceUID"})
    df = df[
        [
            "SOPInstanceUID",
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "any",
            "epidural",
            "intraparenchymal",
            "intraventricular",
            "subarachnoid",
            "subdural",
        ]
    ]
    df = df.sort_values(by=["PatientID", "StudyInstanceUID"])
    df = df.reset_index(drop=True)
    return df


def sort_df_patient_study(df_patient_study: pd.DataFrame) -> list[int]:
    """Sort the DICOM files in a patient study based on the ImagePositionPatient.

    Args:
        df_patient_study (pd.DataFrame): DataFrame with the DICOM files data.

    Returns:
        list[int]: List of integers with the order of the DICOM files.

    Raises:
        ScriptError: If there are more than one patient in the study or if there are more than one moving axis.
    """
    n_patients = df_patient_study["PatientID"].nunique()
    if n_patients != 1:
        raise ScriptError(f"More than one patient in the study: {n_patients}")
    sops = df_patient_study["SOPInstanceUID"]
    image_positions = [dcmread(Path(TRAIN_RAW_DIR, f"{sop}.dcm")).ImagePositionPatient for sop in sops]

    # Find the moving axis
    axis_values = list(zip(*image_positions, strict=False))
    moving_axis = [len(set(a)) != 1 for a in axis_values]

    if sum(moving_axis) == 0:
        # raise Exception("All axes are the same")
        # Return list of -2s to indicate this error
        return [-2] * len(image_positions)

    if sum(moving_axis) > 1:
        # raise Exception(f"More than one moving axis: {moving_axis}. PatientID: {df_patient_study['PatientID'].iloc[0]}, StudyInstanceUID: {df_patient_study['StudyInstanceUID'].iloc[0]}\nPositions: {image_positions}")  # noqa: E501
        # Return list of -1s to indicate this error
        return [-1] * len(image_positions)

    sort_axis = moving_axis.index(True)
    order = sorted(range(len(image_positions)), key=lambda i: image_positions[i][sort_axis])
    return [order.index(i) for i in range(len(order))]


def add_order_to_df_mp_func(mini_group_dfs: pd.DataFrame) -> dict[str, list[int]]:
    """Process a mini group DFs.

    Args:
        mini_group_dfs (pd.DataFrame): DataFrame with the DICOM files data.

    Returns:
        dict[str, list[int]]: Dictionary with the SOPInstanceUID as key and the order as value.
    """
    sop_to_order = {}
    for group_df in tqdm.tqdm(mini_group_dfs, total=len(mini_group_dfs)):
        order = sort_df_patient_study(group_df)
        sop_to_order.update(dict(zip(group_df["SOPInstanceUID"], order, strict=False)))
    return sop_to_order


def add_order_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add the order of the DICOM files to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with the DICOM files data.

    Returns:
        pd.DataFrame: DataFrame with the order of the DICOM files.

    Raises:
        ScriptError: If the DataFrame length + invalid DataFrame length != previous DataFrame length.
    """
    group_dfs = [t[1] for t in list(df.groupby(["PatientID", "StudyInstanceUID"]))]
    mini_group_dfs = split_list_into_n_parts(group_dfs, N_THREADS)
    with mp.Pool(N_THREADS) as p:
        res = p.map(add_order_to_df_mp_func, mini_group_dfs)

    merged_sop_to_order = {}
    for d in tqdm.tqdm(res, total=N_THREADS, desc="Merging results"):
        merged_sop_to_order.update(d)

    df["order"] = df["SOPInstanceUID"].map(merged_sop_to_order.get)

    # Filter the original DataFrame to get the invalid rows (order == -1 or -2)
    prev_len = len(df)
    mask = (df["order"] == -1) | (df["order"] == -2)
    invalid_df = df[mask]
    df = df[~mask]
    if len(df) + len(invalid_df) != prev_len:
        raise ScriptError("Current DataFrame length + invalid DataFrame length != previous DataFrame length")

    df = df.sort_values(by=["PatientID", "StudyInstanceUID", "order"])
    df = df.reset_index(drop=True)
    return df


def window_without_correction(ds: FileDataset, window_center: int, window_width: int) -> np.ndarray:
    """Apply the window without correcting the slope and intercept.

    Args:
        ds (FileDataset): DICOM file dataset.
        window_center (int): Center of the window.
        window_width (int): Width of the window.

    Returns:
        np.ndarray: Windowed image array.
    """
    arr = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    arr_min = window_center - window_width // 2  # -20
    arr_max = window_center + window_width // 2 - 1  # 107
    return np.clip(arr, arr_min, arr_max)


# [-112, 400]
# [-20, 107]
def img_from_dcm(
    ds: FileDataset,
    window_center: int,
    window_width: int,
    max_size: int,
    windowing_function: callable,
) -> Image:  # [-20, 107]
    """Preprocesses the DICOM image.

    Args:
        ds (FileDataset): DICOM file dataset.
        window_center (int): Center of the window.
        window_width (int): Width of the window.
        max_size (int): Maximum size for width and height. Images larger than this will be resized.
        windowing_function (callable): Function to apply windowing.

    Returns:
        Image: Preprocessed image with windowing applied.
    """
    arr = windowing_function(ds=ds, window_center=window_center, window_width=window_width)

    # Use fixed window bounds for normalization (NOT per-image min/max).
    # This ensures consistent HU-to-pixel mapping across all images:
    # pixel 0 = HU_MIN (-20), pixel 255 = HU_MAX (107)
    window_min = window_center - window_width // 2  # -20 HU
    window_max = window_center + window_width // 2 - 1  # 107 HU

    # Normalize from [window_min, window_max] to [0, 255]
    arr = (arr - window_min) / (window_max - window_min) * 255
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)

    # Resize if necessary
    if img.width > max_size or img.height > max_size:
        # Calculate aspect ratio
        aspect_ratio = img.width / img.height

        if img.width > img.height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def create_img_dataset_mp_func(mini_sops: list[str]) -> list[str]:
    """Process a mini group DFs.

    Args:
        mini_sops (list[str]): List of SOPInstanceUIDs.

    Returns:
        list[str]: List of corrupted SOPInstanceUIDs.
    """
    corrupted_sops = []
    for sop in tqdm.tqdm(mini_sops, total=len(mini_sops)):
        try:
            src_path = Path(TRAIN_RAW_DIR, f"{sop}.dcm")
            dst_path = Path(IMG_DATASET_DIR, f"{sop}.png")
            if dst_path.exists():
                continue
            img = img_from_dcm(
                ds=dcmread(src_path),
                window_center=WINDOW_CENTER,
                window_width=WINDOW_WIDTH,
                max_size=MAX_SIZE,
                windowing_function=window_without_correction,
            )
            img.save(dst_path)
        except Exception as e:
            logger.error(f"Error processing {sop} -> {type(e).__name__}: {e}")
            corrupted_sops.append(sop)
    return corrupted_sops


def create_img_dataset(df: pd.DataFrame) -> dict:
    """Create the image dataset from the DICOM files.

    Args:
        df (pd.DataFrame): DataFrame with the DICOM files data.

    Returns:
        dict: Dictionary with the corrupted SOPInstanceUIDs.
    """
    mini_sops = split_list_into_n_parts(df["SOPInstanceUID"], N_THREADS)
    with mp.Pool(N_THREADS) as p:
        res = p.map(create_img_dataset_mp_func, mini_sops)
    corrupted_sops = sum(res, [])  # noqa: RUF017
    return {
        "corrupted_sops": corrupted_sops,
    }


def add_dcm_fields_to_df_mp_func(mini_df: pd.DataFrame) -> list[tuple]:
    """Extract DICOM fields from a mini DataFrame.

    Args:
        mini_df (pd.DataFrame): Mini DataFrame with the DICOM files data.

    Returns:
        list[tuple]: List of tuples with the DICOM fields.
    """
    l = []  # noqa: E741
    for _, row in tqdm.tqdm(mini_df.iterrows(), total=len(mini_df)):
        ds = dcmread(Path(TRAIN_RAW_DIR, f"{row['SOPInstanceUID']}.dcm"))
        l.append(
            (
                ds.Modality,
                ds.ImagePositionPatient,
                ds.ImageOrientationPatient,
                ds.SamplesPerPixel,
                ds.PhotometricInterpretation,
                ds.Rows,
                ds.Columns,
                ds.PixelSpacing,
                ds.BitsAllocated,
                ds.BitsStored,
                ds.HighBit,
                ds.PixelRepresentation,
                ds.WindowCenter,
                ds.WindowWidth,
                ds.RescaleIntercept,
                ds.RescaleSlope,
            ),
        )
    return l


def add_dcm_fields_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add DICOM fields to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with the DICOM files data.

    Returns:
        pd.DataFrame: DataFrame with the DICOM fields.
    """

    def merge_ds_data_dicts(dicts: list[dict]) -> dict:
        d = {
            "Modality": [],
            "Image Position 0": [],
            "Image Position 1": [],
            "Image Position 2": [],
            "Image Orientation 0": [],
            "Image Orientation 1": [],
            "Image Orientation 2": [],
            "Image Orientation 3": [],
            "Image Orientation 4": [],
            "Image Orientation 5": [],
            "Samples per Pixel": [],
            "Photometric Interpretation": [],
            "Rows": [],
            "Columns": [],
            "Pixel Spacing 0": [],
            "Pixel Spacing 1": [],
            "Bits Allocated": [],
            "Bits Stored": [],
            "High Bit": [],
            "Pixel Representation": [],
            # 'Window Center': [],
            # 'Window Width': [],
            "Rescale Intercept": [],
            "Rescale Slope": [],
        }
        for l in tqdm.tqdm(dicts, total=N_THREADS, desc="Merging results"):  # noqa: E741
            for t in tqdm.tqdm(l, leave=False, desc="Processing mini DataFrame"):
                d["Modality"].append(t[0])
                d["Image Position 0"].append(t[1][0])
                d["Image Position 1"].append(t[1][1])
                d["Image Position 2"].append(t[1][2])
                d["Image Orientation 0"].append(t[2][0])
                d["Image Orientation 1"].append(t[2][1])
                d["Image Orientation 2"].append(t[2][2])
                d["Image Orientation 3"].append(t[2][3])
                d["Image Orientation 4"].append(t[2][4])
                d["Image Orientation 5"].append(t[2][5])
                d["Samples per Pixel"].append(t[3])
                d["Photometric Interpretation"].append(t[4])
                d["Rows"].append(t[5])
                d["Columns"].append(t[6])
                d["Pixel Spacing 0"].append(t[7][0])
                d["Pixel Spacing 1"].append(t[7][1])
                d["Bits Allocated"].append(t[8])
                d["Bits Stored"].append(t[9])
                d["High Bit"].append(t[10])
                d["Pixel Representation"].append(t[11])
                # For some reason the Window Center and Window Width are [00036, 00036] [00080, 00080] in some cases
                # d['Window Center'].append(float(t[12]))
                # d['Window Width'].append(float(t[13]))
                d["Rescale Intercept"].append(float(t[14]))
                d["Rescale Slope"].append(float(t[15]))
        return d

    mini_dfs = split_df_into_n_parts(df, N_THREADS)
    with mp.Pool(N_THREADS) as p:
        res = p.map(add_dcm_fields_to_df_mp_func, mini_dfs)

    # %%
    # (0008, 0018) SOP Instance UID                    UI: ID_00e680819
    # (0008, 0060) Modality                            CS: 'CT'
    # (0010, 0020) Patient ID                          LO: 'ID_0002cd41'
    # (0020, 000d) Study Instance UID                  UI: ID_66929e09d4
    # (0020, 000e) Series Instance UID                 UI: ID_e22a5534e6
    # (0020, 0010) Study ID                            SH: ''
    # (0020, 0032) Image Position (Patient)            DS: [-125.000, -122.596, 56.098]
    # (0020, 0037) Image Orientation (Patient)         DS: [1.000000, 0.000000, 0.000000, 0.000000, 0.993572, -0.113203]
    # (0028, 0002) Samples per Pixel                   US: 1
    # (0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
    # (0028, 0010) Rows                                US: 512
    # (0028, 0011) Columns                             US: 512
    # (0028, 0030) Pixel Spacing                       DS: [0.488281, 0.488281]
    # (0028, 0100) Bits Allocated                      US: 16
    # (0028, 0101) Bits Stored                         US: 16
    # (0028, 0102) High Bit                            US: 15
    # (0028, 0103) Pixel Representation                US: 1
    # (0028, 1050) Window Center                       DS: '30.0'
    # (0028, 1051) Window Width                        DS: '80.0'
    # (0028, 1052) Rescale Intercept                   DS: '-1024.0'
    # (0028, 1053) Rescale Slope                       DS: '1.0'

    d = merge_ds_data_dicts(dicts=res)

    # Create new columns in the DataFrame
    df["Modality"] = d["Modality"]
    df["Image Position 0"] = d["Image Position 0"]
    df["Image Position 1"] = d["Image Position 1"]
    df["Image Position 2"] = d["Image Position 2"]
    df["Image Orientation 0"] = d["Image Orientation 0"]
    df["Image Orientation 1"] = d["Image Orientation 1"]
    df["Image Orientation 2"] = d["Image Orientation 2"]
    df["Image Orientation 3"] = d["Image Orientation 3"]
    df["Image Orientation 4"] = d["Image Orientation 4"]
    df["Image Orientation 5"] = d["Image Orientation 5"]
    df["Samples per Pixel"] = d["Samples per Pixel"]
    df["Photometric Interpretation"] = d["Photometric Interpretation"]
    df["Rows"] = d["Rows"]
    df["Columns"] = d["Columns"]
    df["Pixel Spacing 0"] = d["Pixel Spacing 0"]
    df["Pixel Spacing 1"] = d["Pixel Spacing 1"]
    df["Bits Allocated"] = d["Bits Allocated"]
    df["Bits Stored"] = d["Bits Stored"]
    df["High Bit"] = d["High Bit"]
    df["Pixel Representation"] = d["Pixel Representation"]
    # df['Window Center'] = d['Window Center']
    # df['Window Width'] = d['Window Width']
    df["Rescale Intercept"] = d["Rescale Intercept"]
    df["Rescale Slope"] = d["Rescale Slope"]

    return df


def train_valid_test_split(df: pd.DataFrame, train_percentage: float, test_patients_per_subtype: int) -> pd.DataFrame:
    """Split the DataFrame into train, validation and test sets.

    Args:
        df (pd.DataFrame): DataFrame with the DICOM files data.
        train_percentage (float): Percentage of the data to use for training.
        test_patients_per_subtype (int): Number of patients per subtype in the test set.

    Returns:
        pd.DataFrame: DataFrame with the train, validation and test sets.

    Raises:
        ScriptError: If the DataFrame does not contain the expected values in the 'split' column.
    """
    # Create a new column 'stage' to store the split information
    df["stage"] = "placeholder-stage"

    # Filter the DataFrame to get the patients with only one study
    study_counts = df.groupby("PatientID")["StudyInstanceUID"].nunique()
    patients_with_only_one_study = study_counts[study_counts == 1].index.tolist()
    df_one_study = df[df["PatientID"].isin(patients_with_only_one_study)]

    # Split the PatientIDs into train, valid and test sets
    subtypes = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "NO-IH"]
    test_patients_ids = []
    for subtype in subtypes:
        # Get the DataFrame for the current subtype
        if subtype == "NO-IH":
            df_subtype = df_one_study[df_one_study["any"] == 0]
        else:
            df_subtype = df_one_study[df_one_study[subtype] == 1]

        # Get the unique PatientIDs in the group
        unique_patient_ids = df_subtype["PatientID"].unique()

        # Check if there are enough patients for the test set
        if unique_patient_ids.size < 2 * test_patients_per_subtype:
            raise ScriptError(f"Not enough patients for subtype {subtype}: {unique_patient_ids.size}")

        # Shuffle the PatientIDs
        RGN.shuffle(unique_patient_ids)

        # Select test patients: only patients with one study
        subtype_test_patients_ids = unique_patient_ids[:test_patients_per_subtype]
        df_one_study = df_one_study[~df_one_study["PatientID"].isin(subtype_test_patients_ids)]
        test_patients_ids.extend(subtype_test_patients_ids)

        # Check if there are enough patients with one study for the test set
        if len(subtype_test_patients_ids) != test_patients_per_subtype:
            raise ScriptError(f"Not enough patients (1 study) for subtype {subtype}: {len(subtype_test_patients_ids)}")

    # Create a train/valid split
    all_patient_ids = df["PatientID"].unique()
    train_valid_candidates = [pid for pid in all_patient_ids if pid not in test_patients_ids]
    RGN.shuffle(train_valid_candidates)
    cutoff = int(len(train_valid_candidates) * train_percentage)
    train_patients_ids = train_valid_candidates[:cutoff]
    valid_patients_ids = train_valid_candidates[cutoff:]

    # Update the 'stage' column for the test patients
    mask = (df["PatientID"].isin(test_patients_ids)) & (df["stage"] == "placeholder-stage")
    df.loc[mask, "stage"] = "test"
    # Update the 'stage' column for the train patients
    mask = (df["PatientID"].isin(train_patients_ids)) & (df["stage"] == "placeholder-stage")
    df.loc[mask, "stage"] = "train"
    # Update the 'stage' column for the valid patients
    mask = (df["PatientID"].isin(valid_patients_ids)) & (df["stage"] == "placeholder-stage")
    df.loc[mask, "stage"] = "valid"

    # Check if the number of test patients is equal to the expected number
    expected_test_patients = test_patients_per_subtype * len(subtypes)
    df_test_patients = df[df["stage"] == "test"]["PatientID"]
    if df_test_patients.nunique() != expected_test_patients:
        raise ScriptError(
            "The number of test patients is not equal to the expected number: ",
            f"{expected_test_patients} != {df_test_patients.nunique()}",
        )

    # Sanity checks
    df_train = df[df["stage"] == "train"]
    df_valid = df[df["stage"] == "valid"]
    df_test = df[df["stage"] == "test"]

    if set(df_train["PatientID"].unique()) & set(df_valid["PatientID"].unique()):
        raise ScriptError("The train and valid sets have overlapping PatientIDs")
    if set(df_train["PatientID"].unique()) & set(df_test["PatientID"].unique()):
        raise ScriptError("The train and test sets have overlapping PatientIDs")
    if set(df_valid["PatientID"].unique()) & set(df_test["PatientID"].unique()):
        raise ScriptError("The valid and test sets have overlapping PatientIDs")

    for subtype in subtypes:
        df_subtype = df_test[df_test["any"] == 0] if subtype == "NO-IH" else df_test[df_test[subtype] == 1]
        if df_subtype["PatientID"].nunique() < test_patients_per_subtype:
            raise ScriptError(
                f"The number of test patients for subtype {subtype} lower than the expected number: "
                f"test_patients_per_subtype: {test_patients_per_subtype}, "
                f"num_patients: {df_subtype['PatientID'].nunique()}",
            )
        for patient_id in df_subtype["PatientID"].unique():
            num_studies = df_subtype[df_subtype["PatientID"] == patient_id]["StudyInstanceUID"].nunique()
            if num_studies != 1:
                raise ScriptError(
                    f"The number of studies for patient {patient_id} is not equal to 1: {num_studies}",
                )

    # Check if the stage column contains the expected values
    if set(df["stage"].unique()) != {"train", "valid", "test"}:
        raise ScriptError(f"The stage column does not contain the expected values: {df['stage'].unique()}")

    df = df.sort_values(by=["stage", "PatientID", "StudyInstanceUID", "order"])
    df = df.reset_index(drop=True)
    return df


def main() -> None:
    """Main function to run the script."""  # noqa: D401
    # Read raw RSNA 2019 train data
    logger.info("Reading raw RSNA 2019 train data")
    df = pd.read_csv(Path(RAW_DIR, "stage_2_train.csv"))

    # Flatten 6 rows to 1 row + 6 columns (Subtypes)
    logger.info("Flattening subtypes")
    df = flatten_subtypes(df)

    # Add Patient, Series and Study
    # SOPInstanceUID: Unique ID, and name of file in this case, https://stackoverflow.com/a/10179555
    # StudyID: Always empty
    # PatientID -> StudyInstanceUID -> SeriesInstanceUID -> SOPInstanceUID
    # sort index by PatientID, StudyInstanceUID
    logger.info("Adding PatientID, StudyInstanceUID and SeriesInstanceUID")
    df = add_ids_to_df(df=df)

    # Are there studies with more than one SeriesInstanceUID?
    group0 = df.groupby(["PatientID", "StudyInstanceUID"]).size().sort_values(ascending=False)
    group1 = df.groupby(["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]).size().sort_values(ascending=False)

    # Check if group0 and group1 are equal
    if not all(group0.to_numpy() == group1.to_numpy()):
        raise Exception("There are studies with more than one SeriesInstanceUID")

    # Group by PatientID, StudyInstanceUID and check how many unique SeriesInstanceUID there are
    maximum = df.groupby(["PatientID", "StudyInstanceUID"])["SeriesInstanceUID"].nunique().max()
    if maximum > 1:
        raise Exception(f"There are studies with more than one SeriesInstanceUID: {maximum}")

    # As there aren't studies with more than one SeriesInstanceUID, we only use
    # PatientID and StudyInstanceUID. Remove 'SeriesInstanceUID' column.
    df = df.drop(columns=["SeriesInstanceUID"])

    # Add order to the DataFrame
    logger.info("Adding order to the DataFrame")
    df = add_order_to_df(df)
    df = df.sort_values(by=["PatientID", "StudyInstanceUID", "order"])

    # Create the image dataset
    logger.info("Creating the image dataset")
    img_dataset_dict = create_img_dataset(df=df)
    invalid_patient_ids = []
    for sop in img_dataset_dict["corrupted_sops"]:  # only ["ID_6431af929"] in RSNA 2019
        # Get the PatientID from the SOPInstanceUID
        patient_id = df[df["SOPInstanceUID"] == sop]["PatientID"].to_numpy()[0]
        invalid_patient_ids.append(patient_id)
    invalid_patient_ids = set(invalid_patient_ids)
    logger.info(f"Invalid Patient IDs: {invalid_patient_ids}")
    for patient_id in invalid_patient_ids:
        prev_len_df = len(df)

        logger.info(f"Removing Patient ID: {patient_id}")
        sops_to_remove = df[df["PatientID"] == patient_id]["SOPInstanceUID"].to_numpy()
        for sop in sops_to_remove:
            path = Path(IMG_DATASET_DIR, f"{sop}.png")
            if not path.exists():
                continue
            logger.info(f"\t - Removing {path}")
            os.remove(path)
        df = df[df["PatientID"] != patient_id]
        if not len(df) == prev_len_df - len(sops_to_remove):
            raise ScriptError("The DataFrame length is not correct after removing the invalid Patient ID")

        if not len(df) == len(list(IMG_DATASET_DIR.glob("*"))):
            raise ScriptError("The DataFrame length is not equal to the number of images in the dataset")
    df = df.reset_index(drop=True)

    # train-valid-test split
    logger.info("Splitting the DataFrame into train, validation and test sets")
    df = train_valid_test_split(df=df, train_percentage=0.8, test_patients_per_subtype=5)

    # Save the DataFrame
    logger.info("Saving the DataFrame")
    df.to_csv(Path(PRE_DIR, "df.csv"), index=False)

    # Add extra information to the dataframe
    df = add_dcm_fields_to_df(df=df)

    # Save the DataFrame with extra information
    logger.info("Saving the DataFrame with extra information")
    df.to_csv(Path(PRE_DIR, "df_with_extra_info.csv"), index=False)


if __name__ == "__main__":
    import time

    t_start = time.time()
    logger.info("Starting preprocessing")
    main()
    logger.info("Finished preprocessing")
    t_end = time.time()
    logger.info(f"Time taken: {t_end - t_start:.2f} seconds")
