from torch import nn
from torchvision.models.efficientnet import (
    EfficientNet,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
)
from models import (
    UNet,
    EventEgoHandsV1,
    build_efficientnet_v2_s_backbone,
    build_efficientnet_v2_m_backbone,
    build_efficientnet_v2_l_backbone,
)
from utils import SegmentationConfig, HandEstimationConfig


def get_segmentation_model(cfg: SegmentationConfig):
    if cfg.model == "UNet":
        return UNet(n_channels=cfg.n_channels, n_classes=cfg.n_classes)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


def get_hand_model(cfg: HandEstimationConfig):
    if cfg.model == "EventEgoHandsV1":
        return EventEgoHandsV1(
            mano_hand_model_path=cfg.mano_hand_model_path, n_pose_params=cfg.mano_cmps
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


def get_hand_model_v2(cfg: HandEstimationConfig, device):
    """Build EventEgoHandsV2 from a config. Shared by v2 train / test / vis."""
    # Imported here because EventEgoHandsV2 pulls get_spatial_backbone back out
    # of this module.
    from models.EventEgoHandsV2 import EventEgoHandsV2

    return EventEgoHandsV2(
        backbone_name=cfg.model,
        output_mano_dim=cfg.output_mano_dim,
        n_channels=cfg.n_channels,
        mano_hand_model_path=cfg.mano_hand_model_path,
        is_weights=getattr(cfg, "is_weights", True),
        crop_size=(cfg.output_height, cfg.output_width),
        num_pose_coeffs=cfg.mano_cmps,
        use_pose_pca=cfg.mano_use_pose_pca,
        device=device,
        bbox_expand_ratio=getattr(cfg, "bbox_expand_ratio", 1.0),
        use_mask=getattr(cfg, "use_mask", True),
        mask_dilation_kernel_size=getattr(cfg, "mask_dilation_kernel_size", 5),
        mask_dilation_iterations=getattr(cfg, "mask_dilation_iterations", 2),
        # Attention parameters
        use_cross_attention=getattr(cfg, "use_cross_attention", True),
        use_self_attention=getattr(cfg, "use_self_attention", True),
        attention_num_heads=getattr(cfg, "attention_num_heads", 4),
        attention_dropout=getattr(cfg, "attention_dropout", 0.1),
        # Spatial attention parameters
        use_positional_encoding=getattr(cfg, "use_positional_encoding", True),
        num_attention_layers=getattr(cfg, "num_attention_layers", 1),
        # V2 specific parameters
        use_learnable_pos_encoding=getattr(cfg, "use_learnable_pos_encoding", True),
        pos_encoding_base_size=getattr(cfg, "pos_encoding_base_size", 14),
        add_pos_encoding_every_layer=getattr(cfg, "add_pos_encoding_every_layer", True),
        use_shared_backbone=getattr(cfg, "use_shared_backbone", False),
        attention_order=getattr(cfg, "attention_order", "self_first"),
    ).to(device)


def get_spatial_backbone(name: str = "EfficientNetV2_S", n_channels: int = 3, is_weights: bool = True) -> tuple[nn.Module, int]:
    """
    Returns backbone that outputs spatial feature maps (before global average pooling).
    Output shape: [B, C, H, W] instead of [B, C]

    For EfficientNetV2-S with 224x224 input: [B, 1280, 7, 7]

    Args:
        name: Backbone name (e.g., "EfficientNetV2_S")
        n_channels: Number of input channels
        is_weights: Whether to use pretrained weights

    Returns:
        Tuple of (spatial_backbone, out_features)
    """
    backbone, out_features = get_backbone(
        name=name, n_channels=n_channels, use_identity=True, is_weights=is_weights
    )

    if "EfficientNet" in name or "efficientnet" in name:
        # Use features only (without classifier)
        return backbone.features, out_features
    else:
        raise ValueError(f"Spatial backbone not supported for: {name}")


def get_backbone(name: str = "EfficientNetV2_S", n_channels: int = 3, use_identity: bool = True, num_classes: int = 1000, is_weights: bool = True) -> tuple[EfficientNet, int]:
    if name == "efficientnet_v2_s" or name == "EfficientNetV2_S":
        return build_efficientnet_v2_s_backbone(n_channels=n_channels, use_identity=use_identity, num_classes=num_classes, weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if is_weights else None)
    elif name == "efficientnet_v2_m" or name == "EfficientNetV2_M":
        return build_efficientnet_v2_m_backbone(n_channels=n_channels, use_identity=use_identity, num_classes=num_classes, weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if is_weights else None)
    elif name == "efficientnet_v2_l" or name == "EfficientNetV2_L":
        return build_efficientnet_v2_l_backbone(n_channels=n_channels, use_identity=use_identity, num_classes=num_classes, weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1 if is_weights else None)
    else:
        raise ValueError(f"Unknown model: {name}")
