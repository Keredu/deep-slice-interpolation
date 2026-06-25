"""Comprehensive tests for loss functions in CT slice interpolation.

This module tests the loss functions in use:
- Combined losses (combined.py)
- CustomLoss integration (custom.py)
"""

import pytest
import torch

from phd.losses.combined import MSSSIMPlusL1Loss
from phd.losses.custom import CombinedSSIML1Loss, CustomLoss

# Type alias for tensor tuple fixtures
TensorPair = tuple[torch.Tensor, torch.Tensor]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def standard_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return standard input tensors (B=4, C=1, H=256, W=256)."""
    torch.manual_seed(42)
    pred = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    return pred, target


@pytest.fixture
def batch_sizes() -> list[int]:
    """Return different batch sizes to test."""
    return [1, 4, 16]


@pytest.fixture
def identical_images() -> tuple[torch.Tensor, torch.Tensor]:
    """Return identical prediction and target images."""
    torch.manual_seed(42)
    img = torch.rand(4, 1, 256, 256)
    return img.clone(), img.clone()


@pytest.fixture
def very_different_images() -> tuple[torch.Tensor, torch.Tensor]:
    """Return very different prediction and target images."""
    pred = torch.zeros(4, 1, 256, 256)
    target = torch.ones(4, 1, 256, 256)
    return pred, target


@pytest.fixture
def random_noise() -> tuple[torch.Tensor, torch.Tensor]:
    """Return random noise images."""
    torch.manual_seed(42)
    pred = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    return pred, target


# ============================================================================
# Combined Loss Tests
# ============================================================================


class TestMSSSIMPlusL1Loss:
    """Tests for MSSSIMPlusL1Loss."""

    def test_forward_basic(self, standard_input: TensorPair) -> None:
        """Test basic forward pass."""
        pred, target = standard_input
        loss_fn = MSSSIMPlusL1Loss()
        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        """Test with different batch sizes."""
        torch.manual_seed(42)
        pred = torch.rand(batch_size, 1, 256, 256)
        target = torch.rand(batch_size, 1, 256, 256)

        loss_fn = MSSSIMPlusL1Loss()
        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_backward(self, standard_input: TensorPair) -> None:
        """Test backward pass."""
        pred, target = standard_input
        pred.requires_grad = True

        loss_fn = MSSSIMPlusL1Loss()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.any(pred.grad != 0)

    def test_identical_images(self, identical_images: TensorPair) -> None:
        """Test that identical images give low loss."""
        pred, target = identical_images
        loss_fn = MSSSIMPlusL1Loss()
        loss = loss_fn(pred, target)

        assert loss.item() < 0.01

    def test_loss_non_negative(self, random_noise: TensorPair) -> None:
        """Test that loss is non-negative."""
        pred, target = random_noise
        loss_fn = MSSSIMPlusL1Loss()
        loss = loss_fn(pred, target)

        assert loss >= 0


class TestCombinedSSIML1Loss:
    """Tests for CombinedSSIML1Loss."""

    def test_forward_basic(self, standard_input: TensorPair) -> None:
        """Test basic forward pass."""
        pred, target = standard_input
        loss_fn = CombinedSSIML1Loss()
        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        """Test with different batch sizes."""
        torch.manual_seed(42)
        pred = torch.rand(batch_size, 1, 256, 256)
        target = torch.rand(batch_size, 1, 256, 256)

        loss_fn = CombinedSSIML1Loss()
        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_backward(self, standard_input: TensorPair) -> None:
        """Test backward pass."""
        pred, target = standard_input
        pred.requires_grad = True

        loss_fn = CombinedSSIML1Loss()
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.any(pred.grad != 0)

    def test_identical_images(self, identical_images: TensorPair) -> None:
        """Test that identical images give low loss."""
        pred, target = identical_images
        loss_fn = CombinedSSIML1Loss()
        loss = loss_fn(pred, target)

        assert loss.item() < 0.01

    def test_loss_non_negative(self, random_noise: TensorPair) -> None:
        """Test that loss is non-negative."""
        pred, target = random_noise
        loss_fn = CombinedSSIML1Loss()
        loss = loss_fn(pred, target)

        assert loss >= 0


# ============================================================================
# SSIM Stability Parameter Tests
# ============================================================================


class TestSSIMStabilityParams:
    """Tests for SSIM K and nonnegative_ssim parameters."""

    def test_ssim_custom_k(self, standard_input: TensorPair) -> None:
        """Test SSIM with custom K constants via CustomLoss."""
        pred, target = standard_input
        loss_fn = CustomLoss({
            "name": "ssim",
            "params": {"K": [0.01, 0.4], "nonnegative_ssim": True},
        })
        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss >= 0  # nonnegative_ssim should prevent negative loss

    def test_ssim_l1_custom_k(self, standard_input: TensorPair) -> None:
        """Test SSIM+L1 with stability params."""
        pred, target = standard_input
        loss_fn = CustomLoss({
            "name": "ssim+l1",
            "params": {
                "ssim_weight": 0.8, "l1_weight": 0.2,
                "K": [0.01, 0.4], "nonnegative_ssim": True,
            },
        })
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_msssim_l1_custom_k(self, standard_input: TensorPair) -> None:
        """Test MS-SSIM+L1 with stability params."""
        pred, target = standard_input
        loss_fn = CustomLoss({
            "name": "msssim+l1",
            "params": {
                "msssim_weight": 0.5, "l1_weight": 0.5,
                "K": [0.01, 0.4],
            },
        })
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_combined_ssim_l1_nonnegative(self, standard_input: TensorPair) -> None:
        """Test CombinedSSIML1Loss with nonnegative_ssim directly."""
        pred, target = standard_input
        loss_fn = CombinedSSIML1Loss(
            K=(0.01, 0.4), nonnegative_ssim=True,
        )
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_msssim_plus_l1_stability(self, standard_input: TensorPair) -> None:
        """Test MSSSIMPlusL1Loss with stability K param."""
        pred, target = standard_input
        loss_fn = MSSSIMPlusL1Loss(K=(0.01, 0.4))
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0


# ============================================================================
# CustomLoss Integration Tests
# ============================================================================


class TestCustomLoss:
    """Tests for CustomLoss config-based initialization."""

    @pytest.mark.parametrize(
        "loss_config",
        [
            {"name": "mse"},
            {"name": "l1"},
            {"name": "ssim"},
            {"name": "msssim"},
            {"name": "ssim+l1", "params": {"ssim_weight": 0.8, "l1_weight": 0.2}},
            {"name": "msssim+l1", "params": {"msssim_weight": 0.8, "l1_weight": 0.2}},
        ],
    )
    def test_create_all_loss_types(self, standard_input: TensorPair, loss_config: dict) -> None:
        """Test creating each loss type via config dict."""
        pred, target = standard_input

        loss_fn = CustomLoss(loss_config)
        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_unsupported_loss_raises(self) -> None:
        """Test that unsupported loss names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported loss function"):
            CustomLoss({"name": "unsupported_loss"})

    def test_mse_with_identical_images(self, identical_images: TensorPair) -> None:
        """Test MSE loss with identical images."""
        pred, target = identical_images
        loss_fn = CustomLoss({"name": "mse"})
        loss = loss_fn(pred, target)

        assert loss.item() < 1e-6

    def test_l1_with_identical_images(self, identical_images: TensorPair) -> None:
        """Test L1 loss with identical images."""
        pred, target = identical_images
        loss_fn = CustomLoss({"name": "l1"})
        loss = loss_fn(pred, target)

        assert loss.item() < 1e-6

    def test_ssim_with_identical_images(self, identical_images: TensorPair) -> None:
        """Test SSIM loss with identical images."""
        pred, target = identical_images
        loss_fn = CustomLoss({"name": "ssim"})
        loss = loss_fn(pred, target)

        assert loss.item() < 1e-4  # 1 - 1.0 should be ~0

    def test_ssim_params(self, standard_input: TensorPair) -> None:
        """Test SSIM with custom parameters."""
        pred, target = standard_input
        loss_config = {
            "name": "ssim",
            "params": {"data_range": 1.0, "channel": 1, "size_average": True},
        }
        loss_fn = CustomLoss(loss_config)
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)

    def test_combined_loss_params(self, standard_input: TensorPair) -> None:
        """Test combined loss with custom parameters."""
        pred, target = standard_input
        loss_config = {
            "name": "ssim+l1",
            "params": {"ssim_weight": 0.9, "l1_weight": 0.1},
        }
        loss_fn = CustomLoss(loss_config)
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)

    def test_case_insensitive_names(self, standard_input: TensorPair) -> None:
        """Test that loss names are case insensitive."""
        pred, target = standard_input

        for name in ["MSE", "Mse", "mSe"]:
            loss_fn = CustomLoss({"name": name})
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_ssim_l1_missing_weights_raises(self) -> None:
        """Test that ssim+l1 without weights raises ValueError."""
        with pytest.raises(ValueError, match=r"ssim\+l1 loss requires"):
            CustomLoss({"name": "ssim+l1", "params": {}})

    def test_msssim_l1_missing_weights_raises(self) -> None:
        """Test that msssim+l1 without weights raises ValueError."""
        with pytest.raises(ValueError, match=r"msssim\+l1 loss requires"):
            CustomLoss({"name": "msssim+l1", "params": {}})

    def test_backward_all_losses(self, standard_input: TensorPair) -> None:
        """Test backward pass works for all loss types."""
        pred, target = standard_input

        loss_configs = [
            {"name": "mse"},
            {"name": "l1"},
            {"name": "ssim"},
            {"name": "ssim+l1", "params": {"ssim_weight": 0.8, "l1_weight": 0.2}},
        ]

        for config in loss_configs:
            pred_grad = pred.clone().detach().requires_grad_(True)
            loss_fn = CustomLoss(config)
            loss = loss_fn(pred_grad, target)
            loss.backward()

            assert pred_grad.grad is not None, f"No gradient for {config['name']}"
