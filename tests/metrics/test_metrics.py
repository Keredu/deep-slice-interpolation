"""Tests for metrics module."""

import pytest
import torch

from phd.metrics.compute import compute_all_metrics
from phd.metrics.correlation import ncc
from phd.metrics.error import _sobel_filter, gradient_mae, mae
from phd.metrics.quality import ms_ssim, psnr, ssim

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def identical_images() -> tuple[torch.Tensor, torch.Tensor]:
    """Return two identical images."""
    torch.manual_seed(42)
    img = torch.rand(2, 1, 256, 256)
    return img, img.clone()


@pytest.fixture
def different_images() -> tuple[torch.Tensor, torch.Tensor]:
    """Return two different images."""
    torch.manual_seed(42)
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256) * 0.5 + 0.25
    return pred, target


@pytest.fixture
def zeros_vs_ones() -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero image vs one image."""
    pred = torch.zeros(2, 1, 256, 256)
    target = torch.ones(2, 1, 256, 256)
    return pred, target


# ============================================================================
# Quality Metrics Tests
# ============================================================================


class TestSSIM:
    """Tests for SSIM metric."""

    def test_identical_images_perfect_score(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """SSIM of identical images should be 1.0."""
        pred, target = identical_images
        score = ssim(pred, target)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_different_images_lower_score(self, different_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """SSIM of different images should be less than 1.0."""
        pred, target = different_images
        score = ssim(pred, target)
        assert 0 < score < 1.0

    def test_returns_float(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """SSIM should return a Python float."""
        pred, target = identical_images
        score = ssim(pred, target)
        assert isinstance(score, float)


class TestMSSSIM:
    """Tests for MS-SSIM metric."""

    def test_identical_images_perfect_score(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """MS-SSIM of identical images should be 1.0."""
        pred, target = identical_images
        score = ms_ssim(pred, target)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_different_images_lower_score(self, different_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """MS-SSIM of different images should be less than 1.0."""
        pred, target = different_images
        score = ms_ssim(pred, target)
        assert 0 < score < 1.0

    def test_returns_float(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """MS-SSIM should return a Python float."""
        pred, target = identical_images
        score = ms_ssim(pred, target)
        assert isinstance(score, float)


class TestPSNR:
    """Tests for PSNR metric."""

    def test_identical_images_infinite(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """PSNR of identical images should be infinite."""
        pred, target = identical_images
        score = psnr(pred, target)
        assert score == float("inf")

    def test_different_images_finite(self, different_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """PSNR of different images should be finite."""
        pred, target = different_images
        score = psnr(pred, target)
        assert 0 < score < 100  # Reasonable range for typical images

    def test_zeros_vs_ones(self, zeros_vs_ones: tuple[torch.Tensor, torch.Tensor]) -> None:
        """PSNR between 0s and 1s should be 0 dB."""
        pred, target = zeros_vs_ones
        score = psnr(pred, target)
        # MSE = 1, PSNR = 10 * log10(1/1) = 0
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_returns_float(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """PSNR should return a Python float."""
        pred, target = identical_images
        score = psnr(pred, target)
        assert isinstance(score, float)


# ============================================================================
# Error Metrics Tests
# ============================================================================


class TestMAE:
    """Tests for MAE metric."""

    def test_identical_images_zero(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """MAE of identical images should be 0."""
        pred, target = identical_images
        score = mae(pred, target)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_zeros_vs_ones(self, zeros_vs_ones: tuple[torch.Tensor, torch.Tensor]) -> None:
        """MAE between 0s and 1s should be 1.0."""
        pred, target = zeros_vs_ones
        score = mae(pred, target)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_returns_float(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """MAE should return a Python float."""
        pred, target = identical_images
        score = mae(pred, target)
        assert isinstance(score, float)


class TestSobelFilter:
    """Tests for internal Sobel filter function."""

    def test_uniform_image_zero_gradient(self) -> None:
        """Uniform image should have near-zero gradients."""
        img = torch.ones(2, 1, 64, 64) * 0.5
        grad = _sobel_filter(img)
        # Output should be smaller due to convolution with valid padding
        assert grad.shape == (2, 1, 62, 62)
        assert grad.abs().max() < 1e-5

    def test_gradient_image_nonzero(self) -> None:
        """Image with gradient should have non-zero Sobel output."""
        # Create horizontal gradient image
        img = torch.zeros(2, 1, 64, 64)
        for i in range(64):
            img[:, :, :, i] = i / 63.0
        grad = _sobel_filter(img)
        assert grad.mean() > 0

    def test_multichannel_image(self) -> None:
        """Test Sobel filter on multi-channel image."""
        img = torch.rand(2, 3, 64, 64)
        grad = _sobel_filter(img)
        assert grad.shape == (2, 3, 62, 62)


class TestGradientMAE:
    """Tests for Gradient MAE metric."""

    def test_identical_images_zero(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Gradient MAE of identical images should be 0."""
        pred, target = identical_images
        score = gradient_mae(pred, target)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_different_images_nonzero(self, different_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Gradient MAE of different images should be non-zero."""
        pred, target = different_images
        score = gradient_mae(pred, target)
        assert score > 0

    def test_returns_float(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Gradient MAE should return a Python float."""
        pred, target = identical_images
        score = gradient_mae(pred, target)
        assert isinstance(score, float)


# ============================================================================
# Correlation Metrics Tests
# ============================================================================


class TestNCC:
    """Tests for NCC metric."""

    def test_identical_images_perfect_correlation(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """NCC of identical images should be 1.0."""
        pred, target = identical_images
        score = ncc(pred, target)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_inverted_images_negative_correlation(self) -> None:
        """NCC of inverted images should be -1.0."""
        torch.manual_seed(42)
        img = torch.rand(2, 1, 64, 64)
        inverted = 1.0 - img
        score = ncc(img, inverted)
        assert score == pytest.approx(-1.0, abs=1e-4)

    def test_uncorrelated_images(self) -> None:
        """NCC of uncorrelated images should be near 0."""
        torch.manual_seed(42)
        img1 = torch.rand(2, 1, 256, 256)
        torch.manual_seed(123)  # Different seed
        img2 = torch.rand(2, 1, 256, 256)
        score = ncc(img1, img2)
        # Should be close to 0 for random uncorrelated images
        assert -0.5 < score < 0.5

    def test_returns_float(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """NCC should return a Python float."""
        pred, target = identical_images
        score = ncc(pred, target)
        assert isinstance(score, float)


# ============================================================================
# Compute All Metrics Tests
# ============================================================================


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_returns_all_expected_keys(self, different_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Should return all expected metric keys."""
        pred, target = different_images
        metrics = compute_all_metrics(pred, target)

        expected_keys = {"ssim", "ms_ssim", "psnr", "mae", "gradient_mae", "ncc"}
        assert set(metrics.keys()) == expected_keys

    def test_all_values_are_floats(self, different_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """All metric values should be Python floats."""
        pred, target = different_images
        metrics = compute_all_metrics(pred, target)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float"

    def test_identical_images_expected_values(self, identical_images: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Identical images should have expected metric values."""
        pred, target = identical_images
        metrics = compute_all_metrics(pred, target)

        # Quality metrics should be high/perfect
        assert metrics["ssim"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["ms_ssim"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["psnr"] == float("inf")
        assert metrics["ncc"] == pytest.approx(1.0, abs=1e-4)

        # Error metrics should be 0
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["gradient_mae"] == pytest.approx(0.0, abs=1e-6)

    def test_multichannel_input(self) -> None:
        """Should work with multi-channel input."""
        torch.manual_seed(42)
        pred = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)
        metrics = compute_all_metrics(pred, target)

        assert "ssim" in metrics
        assert isinstance(metrics["ssim"], float)
