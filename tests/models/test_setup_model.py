"""Tests for setup_model factory function."""

import pytest
import torch

from phd.models.setup_model import setup_model


class TestSetupModel:
    """Tests for setup_model function."""

    def test_create_unet(self) -> None:
        """Test creating UNet model."""
        model = setup_model(
            in_channels=2,
            out_channels=1,
            pretrained=False,
            model_type="unet",
            encoder_name="resnet18",
        )

        x = torch.rand(1, 2, 256, 256)
        y = model(x)
        assert y.shape == (1, 1, 256, 256)

    def test_invalid_model_type_raises(self) -> None:
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            setup_model(
                in_channels=2,
                out_channels=1,
                pretrained=False,
                model_type="invalid_model",
            )

    def test_pretrained_unet(self) -> None:
        """Test UNet with pretrained encoder weights."""
        model = setup_model(
            in_channels=2,
            out_channels=1,
            pretrained=True,
            model_type="unet",
            encoder_name="resnet18",
        )

        assert model is not None
        x = torch.rand(1, 2, 256, 256)
        y = model(x)
        assert y.shape == (1, 1, 256, 256)
