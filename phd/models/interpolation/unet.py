"""U-Net model for image interpolation using segmentation_models_pytorch."""

import segmentation_models_pytorch as smp
import torch
from torch import nn


class InterpolationUNet(nn.Module):
    """U-Net with EfficientNetV2 encoder for image interpolation.

    Uses segmentation_models_pytorch adapted for regression task.
    Skip connections from encoder to decoder preserve fine spatial details,
    making this architecture particularly suitable for medical image tasks
    and reconstruction problems like CT slice interpolation.
    """

    def __init__(
        self,
        encoder_name: str = "tu-tf_efficientnetv2_s",
        encoder_weights: str = "imagenet",
        in_channels: int = 2,
        out_channels: int = 1,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ) -> None:
        """Initialize U-Net model.

        Args:
            encoder_name: Encoder backbone from timm (via smp)
                         prefix 'tu-' uses timm-universal encoders
            encoder_weights: Pretrained weights ("imagenet" or None)
            in_channels: Number of input channels
            out_channels: Number of output channels
            decoder_channels: Channel counts for each decoder block
        """
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.decoder_channels = decoder_channels

        # Create U-Net model adapted for regression
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            decoder_channels=decoder_channels,
            activation=None,  # No activation for regression (not segmentation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        return self.model(x)

    @classmethod
    def load_from_weights(cls, path: str, **kwargs) -> "InterpolationUNet":
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            **kwargs: Override config values

        Returns:
            Loaded model instance
        """
        state_dict = torch.load(path, weights_only=False)
        for key, value in kwargs.items():
            state_dict["config"][key] = value

        model = cls(
            encoder_name=state_dict["config"]["encoder_name"],
            encoder_weights=state_dict["config"].get("encoder_weights", "imagenet"),
            in_channels=state_dict["config"]["in_channels"],
            out_channels=state_dict["config"]["out_channels"],
            decoder_channels=state_dict["config"].get(
                "decoder_channels",
                (256, 128, 64, 32, 16),
            ),
        )
        model.load_state_dict(state_dict["model_state_dict"], strict=True)
        return model
