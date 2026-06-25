"""Select 50 stratified cases from the test set for the radiologist reader study.

Selects ~20 hemorrhage and ~30 normal triplets, maximizing patient diversity.
Outputs:
  - reader_study/cases.csv: ordered by case_id (1-50)
  - reader_study/cases_randomized.csv: same cases, randomized order for radiologists

Usage:
    uv run scripts/reader_study_select_cases.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Defaults -------------------------------------------------------------------
DEFAULT_HEMORRHAGE_LABELS_CSV = "results/tables/revision_hemorrhage_labels_2026-03-03.csv"
DEFAULT_OUTPUT_DIR = "reader_study"
DEFAULT_SEED = 42
DEFAULT_N_HEMORRHAGE = 20
DEFAULT_N_NORMAL = 30


def load_hemorrhage_labels(csv_path: Path) -> pd.DataFrame:
    """Load hemorrhage labels CSV.

    Returns a DataFrame with columns: patient_id, triplet_index, target_any, target_subtype.
    """
    df = pd.read_csv(csv_path)
    required = {"patient_id", "triplet_index", "target_any", "target_subtype"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Hemorrhage labels CSV missing columns: {missing}"
        raise ValueError(msg)
    return df


def classify_triplets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify each triplet as hemorrhage (target_any == 1) or normal (target_any == 0).

    Returns (hemorrhage_df, normal_df).
    """
    hemorrhage = df[df["target_any"] == 1].copy()
    normal = df[df["target_any"] == 0].copy()
    return hemorrhage, normal


def select_cases(
    hemorrhage_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    n_hemorrhage: int,
    n_normal: int,
    seed: int,
) -> pd.DataFrame:
    """Select cases maximizing patient diversity.

    Strategy: for each category, group by patient_id, shuffle patients,
    then pick one triplet per patient in round-robin until quota is met.
    If we run out of patients, cycle back and pick additional triplets.
    """
    rng = np.random.default_rng(seed)

    def _select_from_pool(pool_df: pd.DataFrame, n: int, label: str) -> list[dict]:
        """Select n triplets from pool, maximizing patient diversity."""
        patients = sorted(pool_df["patient_id"].unique())
        rng.shuffle(patients)

        # Group triplets by patient
        by_patient: dict[str, list[int]] = {}
        for pid in patients:
            indices = pool_df[pool_df["patient_id"] == pid]["triplet_index"].tolist()
            rng.shuffle(indices)
            by_patient[pid] = indices

        selected: list[dict] = []
        patient_cycle = list(patients)
        cycle_idx = 0

        while len(selected) < n:
            if cycle_idx >= len(patient_cycle):
                # All patients used at least once; rebuild list of patients that still have triplets
                patient_cycle = [p for p in patients if by_patient[p]]
                if not patient_cycle:
                    logger.warning(
                        f"Exhausted all {label} triplets after selecting {len(selected)}/{n}"
                    )
                    break
                rng.shuffle(patient_cycle)
                cycle_idx = 0

            pid = patient_cycle[cycle_idx]
            if by_patient[pid]:
                tidx = by_patient[pid].pop(0)
                row = pool_df[
                    (pool_df["patient_id"] == pid) & (pool_df["triplet_index"] == tidx)
                ].iloc[0]
                selected.append(
                    {
                        "patient_id": pid,
                        "triplet_index": tidx,
                        "hemorrhage_status": label,
                        "target_subtype": row["target_subtype"],
                    }
                )
            cycle_idx += 1

        return selected

    hem_cases = _select_from_pool(hemorrhage_df, n_hemorrhage, "hemorrhage")
    nor_cases = _select_from_pool(normal_df, n_normal, "normal")

    all_cases = hem_cases + nor_cases
    # Assign case_ids 1..N
    for i, case in enumerate(all_cases, start=1):
        case["case_id"] = i

    # Build slice_indices string: "k, k+1, k+2" from triplet_index
    for case in all_cases:
        t = case["triplet_index"]
        case["slice_indices"] = f"{t},{t + 1},{t + 2}"

    result = pd.DataFrame(all_cases)
    result = result[["case_id", "patient_id", "triplet_index", "hemorrhage_status", "target_subtype", "slice_indices"]]
    return result


def randomize_cases(cases_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create a randomized version of the cases for the radiologist."""
    rng = np.random.default_rng(seed + 1)  # Different seed to avoid correlation with selection
    shuffled = cases_df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    # Re-assign presentation_order (1-based)
    shuffled = shuffled.copy()
    shuffled.insert(0, "presentation_order", range(1, len(shuffled) + 1))
    return shuffled


def main() -> None:
    """Select cases for reader study."""
    parser = argparse.ArgumentParser(description="Select cases for radiologist reader study")
    parser.add_argument(
        "--hemorrhage-labels",
        type=Path,
        default=Path(DEFAULT_HEMORRHAGE_LABELS_CSV),
        help="Path to hemorrhage labels CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output directory for cases CSVs",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--n-hemorrhage", type=int, default=DEFAULT_N_HEMORRHAGE, help="Number of hemorrhage cases")
    parser.add_argument("--n-normal", type=int, default=DEFAULT_N_NORMAL, help="Number of normal cases")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading hemorrhage labels from {args.hemorrhage_labels}")
    labels_df = load_hemorrhage_labels(args.hemorrhage_labels)

    logger.info(
        f"Total triplets: {len(labels_df)} "
        f"({len(labels_df[labels_df['target_any'] == 1])} hemorrhage, "
        f"{len(labels_df[labels_df['target_any'] == 0])} normal)"
    )
    logger.info(f"Unique patients: {labels_df['patient_id'].nunique()}")

    # Classify
    hem_df, nor_df = classify_triplets(labels_df)
    logger.info(
        f"Hemorrhage patients: {hem_df['patient_id'].nunique()}, "
        f"Normal patients: {nor_df['patient_id'].nunique()}"
    )

    # Select
    cases = select_cases(hem_df, nor_df, args.n_hemorrhage, args.n_normal, args.seed)

    # Summary
    n_unique_patients = cases["patient_id"].nunique()
    logger.info(
        f"Selected {len(cases)} cases from {n_unique_patients} unique patients "
        f"({len(cases[cases['hemorrhage_status'] == 'hemorrhage'])} hemorrhage, "
        f"{len(cases[cases['hemorrhage_status'] == 'normal'])} normal)"
    )

    # Save ordered version
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "cases.csv"
    cases.to_csv(cases_path, index=False)
    logger.info(f"Saved ordered cases to {cases_path}")

    # Save randomized version
    randomized = randomize_cases(cases, args.seed)
    randomized_path = args.output_dir / "cases_randomized.csv"
    randomized.to_csv(randomized_path, index=False)
    logger.info(f"Saved randomized cases to {randomized_path}")

    # Print summary table
    logger.info("Case selection summary:")
    for _, row in cases.iterrows():
        logger.info(
            f"  Case {row['case_id']:2d}: patient={row['patient_id']}, "
            f"triplet={row['triplet_index']:2d}, "
            f"status={row['hemorrhage_status']}, "
            f"subtype={row['target_subtype']}"
        )


if __name__ == "__main__":
    main()
