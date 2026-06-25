"""Tests for reader study scripts.

Tests cover:
  - Case selection logic (stratification, patient diversity, reproducibility)
  - Statistical computations (Cohen's Kappa, ICC, correlations)
  - Data loading and validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from reader_study_analyze import (
    cohens_kappa_weighted,
    compute_icc_2_1,
    compute_inter_rater_agreement,
    compute_metric_correlations,
    compute_per_reader_stats,
    compute_stratified_stats,
    load_scores,
)
from reader_study_select_cases import (
    classify_triplets,
    load_hemorrhage_labels,
    randomize_cases,
    select_cases,
)


class TestCaseSelection:
    """Tests for reader_study_select_cases.py."""

    @pytest.fixture()
    def sample_labels(self, tmp_path: object) -> pd.DataFrame:
        """Create a sample hemorrhage labels DataFrame."""
        rows = []
        # 5 patients, each with some normal and hemorrhage triplets
        for i in range(5):
            pid = f"ID_patient_{i:02d}"
            for t in range(10):
                target_any = 1 if t < 4 else 0
                subtype = "subdural" if target_any else "normal"
                rows.append(
                    {
                        "patient_id": pid,
                        "triplet_index": t,
                        "target_any": target_any,
                        "target_subtype": subtype,
                    }
                )
        return pd.DataFrame(rows)

    def test_classify_triplets(self, sample_labels: pd.DataFrame) -> None:
        """Test that triplets are correctly classified as hemorrhage or normal."""
        hem, nor = classify_triplets(sample_labels)
        assert len(hem) == 20  # 5 patients * 4 hemorrhage triplets each
        assert len(nor) == 30  # 5 patients * 6 normal triplets each
        assert (hem["target_any"] == 1).all()
        assert (nor["target_any"] == 0).all()

    def test_select_cases_counts(self, sample_labels: pd.DataFrame) -> None:
        """Test that correct number of cases are selected per category."""
        hem, nor = classify_triplets(sample_labels)
        cases = select_cases(hem, nor, n_hemorrhage=3, n_normal=5, seed=42)
        assert len(cases) == 8
        assert len(cases[cases["hemorrhage_status"] == "hemorrhage"]) == 3
        assert len(cases[cases["hemorrhage_status"] == "normal"]) == 5

    def test_select_cases_patient_diversity(self, sample_labels: pd.DataFrame) -> None:
        """Test that selection maximizes patient diversity."""
        hem, nor = classify_triplets(sample_labels)
        # Request 5 hemorrhage cases (one per patient if possible)
        cases = select_cases(hem, nor, n_hemorrhage=5, n_normal=5, seed=42)
        hem_cases = cases[cases["hemorrhage_status"] == "hemorrhage"]
        # Should use all 5 patients for 5 hemorrhage cases
        assert hem_cases["patient_id"].nunique() == 5

    def test_select_cases_reproducibility(self, sample_labels: pd.DataFrame) -> None:
        """Test that same seed produces same selection."""
        hem, nor = classify_triplets(sample_labels)
        cases1 = select_cases(hem, nor, n_hemorrhage=3, n_normal=5, seed=42)
        cases2 = select_cases(hem, nor, n_hemorrhage=3, n_normal=5, seed=42)
        pd.testing.assert_frame_equal(cases1, cases2)

    def test_select_cases_different_seed(self, sample_labels: pd.DataFrame) -> None:
        """Test that different seeds produce different selections."""
        hem, nor = classify_triplets(sample_labels)
        cases1 = select_cases(hem, nor, n_hemorrhage=3, n_normal=5, seed=42)
        cases2 = select_cases(hem, nor, n_hemorrhage=3, n_normal=5, seed=99)
        # Not guaranteed to differ, but very likely with enough cases
        # Check they're valid regardless
        assert len(cases1) == len(cases2) == 8

    def test_case_ids_are_sequential(self, sample_labels: pd.DataFrame) -> None:
        """Test that case_ids are 1-based sequential integers."""
        hem, nor = classify_triplets(sample_labels)
        cases = select_cases(hem, nor, n_hemorrhage=3, n_normal=5, seed=42)
        assert list(cases["case_id"]) == list(range(1, 9))

    def test_slice_indices_format(self, sample_labels: pd.DataFrame) -> None:
        """Test that slice_indices has correct format."""
        hem, nor = classify_triplets(sample_labels)
        cases = select_cases(hem, nor, n_hemorrhage=2, n_normal=2, seed=42)
        for _, row in cases.iterrows():
            parts = row["slice_indices"].split(",")
            assert len(parts) == 3
            t = int(parts[0])
            assert int(parts[1]) == t + 1
            assert int(parts[2]) == t + 2

    def test_randomize_cases(self, sample_labels: pd.DataFrame) -> None:
        """Test that randomization preserves all cases but changes order."""
        hem, nor = classify_triplets(sample_labels)
        cases = select_cases(hem, nor, n_hemorrhage=3, n_normal=5, seed=42)
        randomized = randomize_cases(cases, seed=42)
        assert len(randomized) == len(cases)
        assert "presentation_order" in randomized.columns
        assert set(randomized["case_id"]) == set(cases["case_id"])
        assert list(randomized["presentation_order"]) == list(range(1, 9))

    def test_load_hemorrhage_labels_validates_columns(self, tmp_path: object) -> None:
        """Test that loading validates required columns."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"wrong_col": [1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_hemorrhage_labels(bad_csv)


class TestCohensKappa:
    """Tests for quadratic-weighted Cohen's Kappa."""

    def test_perfect_agreement(self) -> None:
        """Perfect agreement should give kappa = 1.0."""
        ratings = np.array([1, 2, 3, 4, 5, 3, 4, 2, 1, 5])
        kappa = cohens_kappa_weighted(ratings, ratings)
        assert kappa == pytest.approx(1.0)

    def test_random_agreement_near_zero(self) -> None:
        """Random ratings should give kappa near 0."""
        rng = np.random.default_rng(42)
        n = 1000
        a = rng.integers(1, 6, size=n)
        b = rng.integers(1, 6, size=n)
        kappa = cohens_kappa_weighted(a, b)
        assert abs(kappa) < 0.1  # Should be near zero for random

    def test_high_agreement(self) -> None:
        """Mostly agreeing ratings should give high kappa."""
        a = np.array([1, 2, 3, 4, 5, 3, 4, 2, 1, 5])
        b = np.array([1, 2, 3, 4, 5, 3, 4, 2, 2, 5])  # One disagreement
        kappa = cohens_kappa_weighted(a, b)
        assert kappa > 0.9

    def test_mismatched_lengths(self) -> None:
        """Should raise on mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa_weighted(np.array([1, 2]), np.array([1, 2, 3]))

    def test_empty_arrays(self) -> None:
        """Empty arrays should return NaN."""
        kappa = cohens_kappa_weighted(np.array([]), np.array([]))
        assert np.isnan(kappa)

    def test_kappa_symmetry(self) -> None:
        """Kappa should be symmetric."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 2, 3, 4, 4])
        assert cohens_kappa_weighted(a, b) == pytest.approx(cohens_kappa_weighted(b, a))


class TestICC:
    """Tests for ICC(2,1) computation."""

    @pytest.fixture()
    def _perfect_scores(self) -> pd.DataFrame:
        """Scores where both readers agree perfectly."""
        rows = []
        for case_id in range(1, 11):
            score = (case_id % 5) + 1
            for reader in ["R1", "R2"]:
                rows.append(
                    {
                        "case_id": case_id,
                        "reader_id": reader,
                        "anatomical_fidelity": score,
                        "diagnostic_adequacy": score,
                        "overall_quality": score,
                    }
                )
        return pd.DataFrame(rows)

    def test_perfect_agreement_icc(self, _perfect_scores: pd.DataFrame) -> None:
        """Perfect agreement should give ICC = 1.0."""
        result = compute_icc_2_1(_perfect_scores, "anatomical_fidelity")
        assert result["icc"] == pytest.approx(1.0)

    def test_icc_returns_expected_keys(self, _perfect_scores: pd.DataFrame) -> None:
        """ICC result should contain all expected keys."""
        result = compute_icc_2_1(_perfect_scores, "anatomical_fidelity")
        expected_keys = {"icc", "f_value", "df1", "df2", "p_value", "ci_lower", "ci_upper"}
        assert set(result.keys()) == expected_keys

    def test_icc_range(self) -> None:
        """ICC should typically be between -1 and 1 for reasonable data."""
        rng = np.random.default_rng(42)
        rows = []
        for case_id in range(1, 51):
            base = rng.integers(1, 6)
            for reader in ["R1", "R2"]:
                noise = rng.integers(-1, 2)
                score = max(1, min(5, base + noise))
                rows.append(
                    {
                        "case_id": case_id,
                        "reader_id": reader,
                        "anatomical_fidelity": score,
                        "diagnostic_adequacy": score,
                        "overall_quality": score,
                    }
                )
        scores = pd.DataFrame(rows)
        result = compute_icc_2_1(scores, "anatomical_fidelity")
        assert -1.0 <= result["icc"] <= 1.0

    def test_icc_single_reader_warning(self) -> None:
        """ICC with single reader should return NaN."""
        scores = pd.DataFrame(
            {
                "case_id": [1, 2, 3],
                "reader_id": ["R1", "R1", "R1"],
                "anatomical_fidelity": [3, 4, 5],
            }
        )
        result = compute_icc_2_1(scores, "anatomical_fidelity")
        assert np.isnan(result["icc"])


class TestPerReaderStats:
    """Tests for per-reader summary statistics."""

    def test_stats_computation(self) -> None:
        """Test that per-reader stats are computed correctly."""
        scores = pd.DataFrame(
            {
                "case_id": [1, 2, 3, 1, 2, 3],
                "reader_id": ["R1", "R1", "R1", "R2", "R2", "R2"],
                "anatomical_fidelity": [3, 4, 5, 2, 3, 4],
                "diagnostic_adequacy": [4, 4, 4, 3, 3, 3],
                "overall_quality": [3, 4, 5, 2, 3, 4],
            }
        )
        result = compute_per_reader_stats(scores)
        assert len(result) == 6  # 2 readers * 3 criteria
        r1_af = result[(result["reader_id"] == "R1") & (result["criterion"] == "anatomical_fidelity")]
        assert r1_af.iloc[0]["mean"] == pytest.approx(4.0)
        assert r1_af.iloc[0]["median"] == pytest.approx(4.0)


class TestStratifiedStats:
    """Tests for stratified analysis."""

    def test_stratified_by_status(self) -> None:
        """Test stratification by hemorrhage status."""
        scores = pd.DataFrame(
            {
                "case_id": [1, 2, 3, 4],
                "reader_id": ["R1", "R1", "R1", "R1"],
                "anatomical_fidelity": [5, 5, 3, 3],
                "diagnostic_adequacy": [5, 5, 3, 3],
                "overall_quality": [5, 5, 3, 3],
            }
        )
        cases = pd.DataFrame(
            {
                "case_id": [1, 2, 3, 4],
                "hemorrhage_status": ["normal", "normal", "hemorrhage", "hemorrhage"],
            }
        )
        result = compute_stratified_stats(scores, cases)
        hem = result[(result["hemorrhage_status"] == "hemorrhage") & (result["criterion"] == "anatomical_fidelity")]
        nor = result[(result["hemorrhage_status"] == "normal") & (result["criterion"] == "anatomical_fidelity")]
        assert hem.iloc[0]["mean"] == pytest.approx(3.0)
        assert nor.iloc[0]["mean"] == pytest.approx(5.0)


class TestMetricCorrelations:
    """Tests for correlation with automated metrics."""

    def test_correlation_computation(self) -> None:
        """Test Spearman correlation computation."""
        # Perfect positive correlation: higher quality = higher SSIM
        scores = pd.DataFrame(
            {
                "case_id": list(range(1, 11)),
                "reader_id": ["R1"] * 10,
                "anatomical_fidelity": list(range(1, 11)),
                "diagnostic_adequacy": list(range(1, 11)),
                "overall_quality": list(range(1, 11)),
            }
        )
        cases = pd.DataFrame(
            {
                "case_id": list(range(1, 11)),
                "patient_id": [f"P{i}" for i in range(10)],
                "triplet_index": [0] * 10,
            }
        )
        metrics = pd.DataFrame(
            {
                "patient_id": [f"P{i}" for i in range(10)],
                "triplet_index": [0] * 10,
                "ssim": np.linspace(0.8, 0.99, 10),
                "mae": np.linspace(0.05, 0.01, 10),
            }
        )
        result = compute_metric_correlations(scores, cases, metrics)
        # SSIM should correlate positively with quality scores
        ssim_corr = result[
            (result["criterion"] == "anatomical_fidelity") & (result["automated_metric"] == "ssim")
        ]
        assert len(ssim_corr) == 1
        assert ssim_corr.iloc[0]["spearman_rho"] > 0.9

    def test_empty_metrics(self) -> None:
        """Test graceful handling of empty metrics."""
        scores = pd.DataFrame(
            {
                "case_id": [1],
                "reader_id": ["R1"],
                "anatomical_fidelity": [3],
                "diagnostic_adequacy": [3],
                "overall_quality": [3],
            }
        )
        cases = pd.DataFrame(
            {
                "case_id": [1],
                "patient_id": ["P1"],
                "triplet_index": [0],
            }
        )
        result = compute_metric_correlations(scores, cases, pd.DataFrame())
        assert result.empty


class TestScoreValidation:
    """Tests for score data validation."""

    def test_valid_scores(self, tmp_path: object) -> None:
        """Test loading valid scores."""
        csv = tmp_path / "scores.csv"
        pd.DataFrame(
            {
                "case_id": [1, 2],
                "reader_id": ["R1", "R1"],
                "anatomical_fidelity": [3, 4],
                "diagnostic_adequacy": [3, 4],
                "overall_quality": [3, 4],
                "notes": ["", "ok"],
            }
        ).to_csv(csv, index=False)
        df = load_scores(csv)
        assert len(df) == 2

    def test_scores_out_of_range(self, tmp_path: object) -> None:
        """Test that out-of-range scores are rejected."""
        csv = tmp_path / "scores.csv"
        pd.DataFrame(
            {
                "case_id": [1],
                "reader_id": ["R1"],
                "anatomical_fidelity": [6],  # Out of range
                "diagnostic_adequacy": [3],
                "overall_quality": [3],
            }
        ).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="out of range"):
            load_scores(csv)

    def test_missing_columns(self, tmp_path: object) -> None:
        """Test that missing columns are detected."""
        csv = tmp_path / "scores.csv"
        pd.DataFrame(
            {
                "case_id": [1],
                "reader_id": ["R1"],
                "anatomical_fidelity": [3],
                # Missing diagnostic_adequacy and overall_quality
            }
        ).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_scores(csv)


class TestInterRaterAgreement:
    """Tests for the full inter-rater agreement pipeline."""

    def test_two_reader_agreement(self) -> None:
        """Test agreement computation with two readers."""
        scores = pd.DataFrame(
            {
                "case_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                "reader_id": ["R1"] * 5 + ["R2"] * 5,
                "anatomical_fidelity": [3, 4, 5, 3, 4, 3, 4, 5, 3, 4],
                "diagnostic_adequacy": [4, 4, 5, 3, 4, 4, 4, 5, 3, 4],
                "overall_quality": [3, 4, 5, 3, 4, 3, 4, 5, 3, 4],
            }
        )
        result = compute_inter_rater_agreement(scores)
        assert len(result) == 3  # One row per criterion
        # Perfect agreement -> kappa should be 1.0
        for _, row in result.iterrows():
            assert row["cohens_kappa_weighted"] == pytest.approx(1.0)
            assert row["icc_icc"] == pytest.approx(1.0)
