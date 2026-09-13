"""
Reference:
https://pytorch.org/vision/main/models/efficientnet.html
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)
from torchvision.models.efficientnet import EfficientNet


def _load_pretrained_first_conv(
    model: EfficientNet, n_channels: int, efficientnet_type: str = "efficientnet_v2_s"
) -> None:
    if efficientnet_type == "efficientnet_v2_s":
        pretrained = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    elif efficientnet_type == "efficientnet_v2_m":
        pretrained = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    elif efficientnet_type == "efficientnet_v2_l":
        pretrained = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown EfficientNet type: {efficientnet_type}")

    # First conv is at features[0][0]
    w_rgb = pretrained.features[0][0].weight  # [out_channels, 3, k, k]

    with torch.no_grad():
        if n_channels == 1:
            model.features[0][0].weight.copy_(w_rgb.mean(dim=1, keepdim=True))
        elif n_channels > 3:
            repeat = n_channels - 3
            extra = w_rgb[:, :repeat, :, :]
            w_new = torch.cat([w_rgb, extra], dim=1)
            model.features[0][0].weight.copy_(w_new)
        else:  # 2ch
            model.features[0][0].weight[:, :n_channels].copy_(w_rgb[:, :n_channels])


def _replace_first_conv(model: EfficientNet, n_channels: int) -> None:
    if n_channels != 3:
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            n_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )


def _set_classifier(
    model: EfficientNet, num_classes: int, use_identity: bool, dropout: float = 0.2
) -> None:
    if use_identity:
        model.classifier = nn.Identity()
    else:
        # classifier is nn.Sequential(Dropout, Linear)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )


def build_efficientnet_v2_s_backbone(
    n_channels: int = 3,
    num_classes: int = 1000,
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    use_identity: bool = False,
) -> tuple[EfficientNet, int]:
    model = efficientnet_v2_s(weights=weights)
    out_features = model.classifier[1].in_features  # 1280
    if n_channels != 3:
        _replace_first_conv(model, n_channels)
        _load_pretrained_first_conv(model, n_channels, "efficientnet_v2_s")
    _set_classifier(model, num_classes, use_identity)
    return model, out_features


def build_efficientnet_v2_m_backbone(
    n_channels: int = 3,
    num_classes: int = 1000,
    weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1,
    use_identity: bool = False,
) -> tuple[EfficientNet, int]:
    model = efficientnet_v2_m(weights=weights)
    out_features = model.classifier[1].in_features  # 1280
    if n_channels != 3:
        _replace_first_conv(model, n_channels)
        _load_pretrained_first_conv(model, n_channels, "efficientnet_v2_m")
    _set_classifier(model, num_classes, use_identity)
    return model, out_features


def build_efficientnet_v2_l_backbone(
    n_channels: int = 3,
    num_classes: int = 1000,
    weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1,
    use_identity: bool = False,
) -> tuple[EfficientNet, int]:
    model = efficientnet_v2_l(weights=weights)
    out_features = model.classifier[1].in_features  # 1280
    if n_channels != 3:
        _replace_first_conv(model, n_channels)
        _load_pretrained_first_conv(model, n_channels, "efficientnet_v2_l")
    _set_classifier(model, num_classes, use_identity)
    return model, out_features


if __name__ == "__main__":
    model, out_features = build_efficientnet_v2_s_backbone(
        n_channels=1, num_classes=512, use_identity=True
    )
    print(model)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    print(y.shape)
    print(out_features)
