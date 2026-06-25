"""Tests for DICOM HU range export in export_test_interpolated_series.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


class TestDicomHUConstants:
    """Tests for HU windowing constants."""

    def test_hu_constants_defined(self) -> None:
        """Verify HU constants are defined correctly."""
        from export_test_interpolated_series import (
            HU_MAX,
            HU_MIN,
            HU_WINDOW_CENTER,
            HU_WINDOW_WIDTH,
        )

        # Brain tissue window parameters
        assert HU_WINDOW_CENTER == 44
        assert HU_WINDOW_WIDTH == 128

        # Derived range: [-20, 107] (128 discrete values)
        assert HU_MIN == -20  # 44 - 128//2
        assert HU_MAX == 107  # 44 + 128//2 - 1

    def test_hu_range_is_127(self) -> None:
        """Verify the HU range spans 127 HU (from -20 to 107 inclusive = 128 values)."""
        from export_test_interpolated_series import HU_MAX, HU_MIN

        assert HU_MAX - HU_MIN == 127


class TestDicomRescaleMapping:
    """Tests for pixel value to HU mapping."""

    def test_rescale_slope_calculation(self) -> None:
        """Verify RescaleSlope maps [0,255] to [HU_MIN, HU_MAX]."""
        from export_test_interpolated_series import HU_MAX, HU_MIN

        # RescaleSlope = (HU_MAX - HU_MIN) / 255
        expected_slope = (HU_MAX - HU_MIN) / 255.0
        assert expected_slope == pytest.approx(127.0 / 255.0)
        assert expected_slope == pytest.approx(0.49804, rel=1e-4)

    def test_pixel_0_maps_to_hu_min(self) -> None:
        """Verify pixel value 0 maps to HU_MIN (-20)."""
        from export_test_interpolated_series import HU_MAX, HU_MIN

        pixel_value = 0
        slope = (HU_MAX - HU_MIN) / 255.0
        intercept = HU_MIN
        hu_value = pixel_value * slope + intercept
        assert hu_value == pytest.approx(-20.0)

    def test_pixel_255_maps_to_hu_max(self) -> None:
        """Verify pixel value 255 maps to HU_MAX (107)."""
        from export_test_interpolated_series import HU_MAX, HU_MIN

        pixel_value = 255
        slope = (HU_MAX - HU_MIN) / 255.0
        intercept = HU_MIN
        hu_value = pixel_value * slope + intercept
        assert hu_value == pytest.approx(107.0)

    def test_pixel_127_maps_to_center(self) -> None:
        """Verify middle pixel value (127.5) maps close to window center."""
        from export_test_interpolated_series import HU_MAX, HU_MIN

        pixel_value = 127.5
        slope = (HU_MAX - HU_MIN) / 255.0
        intercept = HU_MIN
        hu_value = pixel_value * slope + intercept
        # Should be approximately at the midpoint of [-20, 107] = 43.5
        expected_center = (HU_MIN + HU_MAX) / 2.0  # 43.5
        assert hu_value == pytest.approx(expected_center, abs=0.5)


class TestMakeScDicomHU:
    """Tests for _make_sc_dicom function HU parameters."""

    @pytest.fixture
    def mock_template(self) -> MockDataset:
        """Create a minimal mock DICOM template."""

        class MockDataset:
            PatientID = "TEST001"
            PatientName = "Test^Patient"
            StudyInstanceUID = "1.2.3.4.5"

        return MockDataset()

    def test_make_sc_dicom_has_correct_rescale_intercept(
        self, mock_template: MockDataset
    ) -> None:
        """Verify _make_sc_dicom sets RescaleIntercept to HU_MIN."""
        from export_test_interpolated_series import HU_MIN, _make_sc_dicom

        pixel_u8 = np.zeros((64, 64), dtype=np.uint8)
        ds = _make_sc_dicom(
            template=mock_template,
            pixel_u8=pixel_u8,
            sop_instance_uid="1.2.3.4.5.6",
            series_instance_uid="1.2.3.4.5.7",
            instance_number=1,
            image_position_patient=None,
            derivation_description="Test",
            intensity_mode="hu",
        )
        assert ds.RescaleIntercept == pytest.approx(HU_MIN)

    def test_make_sc_dicom_has_correct_rescale_slope(
        self, mock_template: MockDataset
    ) -> None:
        """Verify _make_sc_dicom sets RescaleSlope for [0,255]->[HU_MIN,HU_MAX]."""
        from export_test_interpolated_series import HU_MAX, HU_MIN, _make_sc_dicom

        pixel_u8 = np.zeros((64, 64), dtype=np.uint8)
        ds = _make_sc_dicom(
            template=mock_template,
            pixel_u8=pixel_u8,
            sop_instance_uid="1.2.3.4.5.6",
            series_instance_uid="1.2.3.4.5.7",
            instance_number=1,
            image_position_patient=None,
            derivation_description="Test",
            intensity_mode="hu",
        )
        expected_slope = (HU_MAX - HU_MIN) / 255.0
        assert ds.RescaleSlope == pytest.approx(expected_slope)

    def test_make_sc_dicom_has_rescale_type_hu(
        self, mock_template: MockDataset
    ) -> None:
        """Verify _make_sc_dicom sets RescaleType to 'HU'."""
        from export_test_interpolated_series import _make_sc_dicom

        pixel_u8 = np.zeros((64, 64), dtype=np.uint8)
        ds = _make_sc_dicom(
            template=mock_template,
            pixel_u8=pixel_u8,
            sop_instance_uid="1.2.3.4.5.6",
            series_instance_uid="1.2.3.4.5.7",
            instance_number=1,
            image_position_patient=None,
            derivation_description="Test",
            intensity_mode="hu",
        )
        assert ds.RescaleType == "HU"

    def test_make_sc_dicom_has_correct_window_center(
        self, mock_template: MockDataset
    ) -> None:
        """Verify _make_sc_dicom sets WindowCenter to brain tissue value (44 HU)."""
        from export_test_interpolated_series import HU_WINDOW_CENTER, _make_sc_dicom

        pixel_u8 = np.zeros((64, 64), dtype=np.uint8)
        ds = _make_sc_dicom(
            template=mock_template,
            pixel_u8=pixel_u8,
            sop_instance_uid="1.2.3.4.5.6",
            series_instance_uid="1.2.3.4.5.7",
            instance_number=1,
            image_position_patient=None,
            derivation_description="Test",
            intensity_mode="hu",
        )
        assert ds.WindowCenter == pytest.approx(HU_WINDOW_CENTER)

    def test_make_sc_dicom_has_correct_window_width(
        self, mock_template: MockDataset
    ) -> None:
        """Verify _make_sc_dicom sets WindowWidth to brain tissue value (128 HU)."""
        from export_test_interpolated_series import HU_WINDOW_WIDTH, _make_sc_dicom

        pixel_u8 = np.zeros((64, 64), dtype=np.uint8)
        ds = _make_sc_dicom(
            template=mock_template,
            pixel_u8=pixel_u8,
            sop_instance_uid="1.2.3.4.5.6",
            series_instance_uid="1.2.3.4.5.7",
            instance_number=1,
            image_position_patient=None,
            derivation_description="Test",
            intensity_mode="hu",
        )
        assert ds.WindowWidth == pytest.approx(HU_WINDOW_WIDTH)


class TestEndToEndHUConversion:
    """End-to-end tests for HU conversion in exported DICOMs."""

    @pytest.fixture
    def mock_template(self) -> MockDataset:
        """Create a minimal mock DICOM template."""

        class MockDataset:
            PatientID = "TEST001"
            PatientName = "Test^Patient"
            StudyInstanceUID = "1.2.3.4.5"

        return MockDataset()

    def test_full_hu_range_preserved(self, mock_template: MockDataset) -> None:
        """Verify that pixel values [0,255] correctly convert to HU [-20,107]."""
        from export_test_interpolated_series import _make_sc_dicom

        # Create test image with known values
        pixel_u8 = np.array([[0, 127], [128, 255]], dtype=np.uint8)
        ds = _make_sc_dicom(
            template=mock_template,
            pixel_u8=pixel_u8,
            sop_instance_uid="1.2.3.4.5.6",
            series_instance_uid="1.2.3.4.5.7",
            instance_number=1,
            image_position_patient=None,
            derivation_description="Test",
            intensity_mode="hu",
        )

        # Convert pixels to HU using DICOM formula
        pixels = np.frombuffer(ds.PixelData, dtype=np.uint8).reshape(2, 2)
        hu_values = pixels * ds.RescaleSlope + ds.RescaleIntercept

        # Verify corner values
        assert hu_values[0, 0] == pytest.approx(-20.0)  # pixel 0
        assert hu_values[1, 1] == pytest.approx(107.0)  # pixel 255

        # Middle values should be around the center (43.5)
        assert hu_values[0, 1] == pytest.approx(43.25, abs=1.0)  # pixel 127
        assert hu_values[1, 0] == pytest.approx(43.75, abs=1.0)  # pixel 128
