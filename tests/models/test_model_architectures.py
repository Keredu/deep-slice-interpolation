"""Test all model architectures for consistent behavior."""

from pathlib import Path

import pytest
import torch

from phd.models.interpolation.unet import InterpolationUNet


class TestModelArchitectures:
    """Test all model architectures for shape consistency and interface."""

    def test_unet_shapes(self) -> None:
        """Test U-Net forward pass shapes."""
        in_channels = 2
        out_channels = 1
        batch_size = 2
        sizes = [256, 512]

        model = InterpolationUNet(
            encoder_name="tu-tf_efficientnetv2_s",
            encoder_weights=None,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        model.eval()

        for size in sizes:
            with torch.no_grad():
                x = torch.randn(batch_size, in_channels, size, size)
                y = model(x)
                assert y.shape == (batch_size, out_channels, size, size), (
                    f"Expected shape ({batch_size}, {out_channels}, {size}, {size}), got {y.shape}"
                )

    def test_models_are_callable(self) -> None:
        """Verify all models can be instantiated and called."""
        in_channels = 2
        out_channels = 1

        model = InterpolationUNet(
            encoder_name="tu-tf_efficientnetv2_s",
            encoder_weights=None,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        x = torch.randn(1, in_channels, 256, 256)

        model.eval()
        with torch.no_grad():
            output = model(x)
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4


class TestModelValidation:
    """Test model validation and error handling."""

    def test_unet_invalid_encoder_name(self) -> None:
        """Test that invalid encoder name raises KeyError."""
        with pytest.raises(KeyError):
            InterpolationUNet(
                encoder_name="invalid_encoder",
                encoder_weights=None,
                in_channels=2,
                out_channels=1,
            )


class TestModelCheckpointLoading:
    """Test model checkpoint loading."""

    def test_unet_load_from_weights(self, tmp_path: Path) -> None:
        """Test loading U-Net from checkpoint."""
        model = InterpolationUNet(
            encoder_name="tu-tf_efficientnetv2_s",
            encoder_weights=None,
            in_channels=2,
            out_channels=1,
        )

        checkpoint = {
            "config": {
                "encoder_name": "tu-tf_efficientnetv2_s",
                "encoder_weights": None,
                "in_channels": 2,
                "out_channels": 1,
            },
            "model_state_dict": model.state_dict(),
        }

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        loaded_model = InterpolationUNet.load_from_weights(str(checkpoint_path))
        loaded_model.eval()
        with torch.no_grad():
            x = torch.randn(1, 2, 256, 256)
            y = loaded_model(x)
            assert y.shape == (1, 1, 256, 256)

    def test_unet_load_from_weights_with_overrides(self, tmp_path: Path) -> None:
        """Test loading U-Net with config overrides."""
        model = InterpolationUNet(
            encoder_name="tu-tf_efficientnetv2_s",
            encoder_weights=None,
            in_channels=2,
            out_channels=1,
        )

        checkpoint = {
            "config": {
                "encoder_name": "tu-tf_efficientnetv2_s",
                "encoder_weights": "imagenet",  # Will be overridden
                "in_channels": 2,
                "out_channels": 1,
            },
            "model_state_dict": model.state_dict(),
        }

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        loaded_model = InterpolationUNet.load_from_weights(str(checkpoint_path), encoder_weights=None)
        loaded_model.eval()
        with torch.no_grad():
            x = torch.randn(1, 2, 256, 256)
            y = loaded_model(x)
            assert y.shape == (1, 1, 256, 256)
