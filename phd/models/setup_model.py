from torch import nn


def setup_model(
    in_channels: int,
    out_channels: int,
    pretrained: bool,
    model_type: str,
    encoder_name: str | None = None,
) -> nn.Module:
    """Set up the interpolation model based on CONFIG.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        pretrained (bool): Whether to use pretrained weights
        model_type (str): Type of model ('unet')
        encoder_name (str | None): Encoder name for U-Net (e.g., 'tu-tf_efficientnetv2_s')

    Returns:
        nn.Module: Interpolation model instance

    Raises:
        ValueError: If unknown model_type specified
    """
    if model_type == "unet":
        from phd.models.interpolation.unet import InterpolationUNet

        return InterpolationUNet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    raise ValueError(f"Unknown model_type: {model_type}. Must be 'unet'")
