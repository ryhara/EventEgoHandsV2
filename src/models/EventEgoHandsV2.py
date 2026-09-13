import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from factory.get_model import get_spatial_backbone
from models.MANOHandModelCustom import MANOHandModelCustom


class MANODecoder(nn.Module):
    def __init__(
        self,
        n_pose_params=15,  # 15 is hot3d, 6 is Ev2Hands pretrained model
        n_shape_params=10,
        device: str = "cuda",
    ):
        super(MANODecoder, self).__init__()

        self.n_pose_params = n_pose_params
        self.n_shape_params = n_shape_params
        self.n_mano_params = n_pose_params + n_shape_params
        self.n_global_orient_params = 3
        self.n_transl_params = 3
        self.device = device

    def forward(
        self, x, key: str, mano_hand_model: MANOHandModelCustom,
        use_default_shape: bool = False,
    ):  # key: "left" or "right"
        batch_size = x.shape[0]
        # global_orient: 0-2
        global_orient = x[:, : self.n_global_orient_params]
        # hand_pose: 3-17
        hand_pose = x[
            :,
            self.n_global_orient_params : self.n_global_orient_params
            + self.n_pose_params,
        ]
        # betas: 18-27
        betas = x[
            :, self.n_global_orient_params + self.n_pose_params : -self.n_transl_params
        ]
        # transl: 28-30
        transl = x[:, -self.n_transl_params :]

        mano_args = {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": betas,
            "transl": transl,
        }

        mano_outs = dict()

        global_xfrom = torch.cat(
            [
                global_orient,
                transl,
            ],
            dim=1,
        ).to(self.device)
        if use_default_shape:
            shape_params = torch.zeros_like(betas[0]).to(self.device)
        else:
            shape_params = betas[0].to(self.device)
        joint_angles = hand_pose.to(self.device)
        is_right_hand = torch.tensor([key == "right"] * batch_size).to(self.device)

        vertices, hand_landmarks = mano_hand_model.forward_kinematics(
            shape_params=shape_params,
            joint_angles=joint_angles,
            global_xfrom=global_xfrom,
            is_right_hand=is_right_hand,
        )
        mano_outs["global_orient"] = global_orient
        mano_outs["vertices"] = vertices.to(device=x.device)
        mano_outs["j3d"] = hand_landmarks.to(device=x.device)

        mano_outs.update(mano_args)

        if key == "right":
            hand_triangles = mano_hand_model.mano_layer_right.faces
        else:
            hand_triangles = mano_hand_model.mano_layer_left.faces

        mano_outs["faces"] = torch.from_numpy(hand_triangles).to(
            dtype=torch.int32, device=x.device
        )

        return mano_outs

def dilation_masks_gpu(masks: torch.Tensor, kernel_size: int = 5, iterations: int = 1) -> torch.Tensor:
    """
    GPU-accelerated dilation using max pooling.

    Args:
        masks: [N, H, W] or [N, 1, H, W] binary masks
        kernel_size: Size of dilation kernel
        iterations: Number of dilation iterations

    Returns:
        Dilated masks with same shape as input
    """
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    padding = kernel_size // 2
    result = masks

    for _ in range(iterations):
        result = F.max_pool2d(result, kernel_size=kernel_size, stride=1, padding=padding)

    if squeeze_output:
        result = result.squeeze(1)

    return result


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to replace Global Average Pooling.
    Uses a learnable query vector to compute spatial attention weights.
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Feature dimension (channels)
        """
        super(AttentionPooling, self).__init__()
        self.embed_dim = embed_dim

        # Learnable query vector for attention
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial feature map

        Returns:
            [B, C] pooled feature vector
        """
        B, C, H, W = x.shape

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        x_seq = x.flatten(2).permute(0, 2, 1)  # [B, N, C] where N = H*W

        # Expand query for batch: [1, 1, C] -> [B, 1, C]
        query = self.query.expand(B, -1, -1)

        # Compute attention scores: [B, 1, C] @ [B, C, N] -> [B, 1, N]
        attn_scores = torch.bmm(query, x_seq.transpose(1, 2)) * self.scale

        # Softmax over spatial positions
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, N]

        # Weighted sum: [B, 1, N] @ [B, N, C] -> [B, 1, C]
        pooled = torch.bmm(attn_weights, x_seq)

        # Remove the extra dimension: [B, 1, C] -> [B, C]
        return pooled.squeeze(1)


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for spatial feature maps.
    (Legacy - kept for backward compatibility)
    """

    def __init__(self, embed_dim: int, max_h: int = 64, max_w: int = 64):
        super(PositionalEncoding2D, self).__init__()
        self.embed_dim = embed_dim

        # Create positional encoding matrix
        pe = torch.zeros(embed_dim, max_h, max_w)

        d_model = embed_dim // 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()

        # Height encoding
        pe[0:d_model:2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(-1, -1, max_w)
        )

        # Width encoding
        pe[d_model::2, :, :] = (
            torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)
        )
        pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(-1, max_h, -1)
        )

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            [B, C, H, W] feature map with positional encoding added
        """
        B, C, H, W = x.shape
        return x + self.pe[:C, :H, :W].unsqueeze(0)


class LearnablePositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for spatial feature maps.
    Supports dynamic size through interpolation.
    """

    def __init__(self, embed_dim: int, base_h: int = 14, base_w: int = 14):
        """
        Args:
            embed_dim: Feature dimension (channels)
            base_h: Base height for positional encoding (will be interpolated if different)
            base_w: Base width for positional encoding (will be interpolated if different)
        """
        super(LearnablePositionalEncoding2D, self).__init__()
        self.embed_dim = embed_dim
        self.base_h = base_h
        self.base_w = base_w

        # Learnable positional encoding: [1, C, H, W]
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, base_h, base_w))

        # Initialize with truncated normal (following ViT)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            [B, C, H, W] feature map with positional encoding added
        """
        B, C, H, W = x.shape

        # Get positional encoding (with interpolation if needed)
        pos_encoding = self._get_pos_encoding(H, W, x.device, x.dtype)

        return x + pos_encoding

    def _get_pos_encoding(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get positional encoding, interpolating if size differs from base.
        """
        if H == self.base_h and W == self.base_w:
            return self.pos_embed

        # Interpolate to target size using bicubic interpolation
        pos_encoding = F.interpolate(
            self.pos_embed,
            size=(H, W),
            mode='bicubic',
            align_corners=False,
        )

        return pos_encoding

    def get_encoding_for_seq(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get positional encoding in sequence format [1, H*W, C].
        Useful for adding to sequence directly.
        """
        pos_encoding = self._get_pos_encoding(H, W, device, dtype)
        # [1, C, H, W] -> [1, H*W, C]
        return pos_encoding.flatten(2).permute(0, 2, 1)


class SpatialSelfAttentionBlock(nn.Module):
    """
    Self-Attention block for spatial feature maps.
    Each spatial position attends to all other positions.
    Optimized to accept sequence input directly when chaining multiple layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_learnable_pos_encoding: bool = True,
        pos_encoding_base_size: int = 14,
    ):
        super(SpatialSelfAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        self.use_learnable_pos_encoding = use_learnable_pos_encoding

        if use_positional_encoding:
            if use_learnable_pos_encoding:
                self.pos_encoding = LearnablePositionalEncoding2D(
                    embed_dim, base_h=pos_encoding_base_size, base_w=pos_encoding_base_size
                )
            else:
                self.pos_encoding = PositionalEncoding2D(embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial feature map

        Returns:
            [B, C, H, W] refined spatial feature map

        Note: Uses Pre-Norm structure (LayerNorm before attention/FFN)
        for better gradient flow in deep networks.
        """
        B, C, H, W = x.shape

        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        x_seq = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Pre-Norm Self-attention with residual
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out

        # Pre-Norm FFN with residual
        x_norm = self.norm2(x_seq)
        ffn_out = self.ffn(x_norm)
        x_seq = x_seq + ffn_out

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        return x_seq.permute(0, 2, 1).view(B, C, H, W)

    def forward_seq(
        self,
        x_seq: torch.Tensor,
        spatial_shape: Tuple[int, int],
        add_pos_encoding: bool = True
    ) -> torch.Tensor:
        """
        Optimized forward pass that operates on sequence directly.

        Args:
            x_seq: [B, H*W, C] sequence
            spatial_shape: (H, W) tuple for positional encoding
            add_pos_encoding: Whether to add positional encoding

        Returns:
            [B, H*W, C] refined sequence

        Note: Uses Pre-Norm structure (LayerNorm before attention/FFN)
        for better gradient flow in deep networks.
        """
        B, N, C = x_seq.shape
        H, W = spatial_shape

        # Add positional encoding if needed
        if add_pos_encoding and self.use_positional_encoding:
            # Reshape to spatial, add encoding, reshape back
            x_spatial = x_seq.permute(0, 2, 1).view(B, C, H, W)
            x_spatial = self.pos_encoding(x_spatial)
            x_seq = x_spatial.flatten(2).permute(0, 2, 1)

        # Pre-Norm Self-attention with residual
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out

        # Pre-Norm FFN with residual
        x_norm = self.norm2(x_seq)
        ffn_out = self.ffn(x_norm)
        x_seq = x_seq + ffn_out

        return x_seq


class SpatialCrossAttentionBlock(nn.Module):
    """
    Cross-Attention block for spatial feature maps between left and right hands.
    Each spatial position in one hand attends to all positions in the other hand.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_learnable_pos_encoding: bool = True,
        pos_encoding_base_size: int = 14,
    ):
        super(SpatialCrossAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        self.use_learnable_pos_encoding = use_learnable_pos_encoding

        if use_positional_encoding:
            if use_learnable_pos_encoding:
                # Separate positional encoding for left and right hands
                # to learn hand-specific spatial patterns (e.g. mirrored thumb positions)
                self.pos_encoding_left = LearnablePositionalEncoding2D(
                    embed_dim, base_h=pos_encoding_base_size, base_w=pos_encoding_base_size
                )
                self.pos_encoding_right = LearnablePositionalEncoding2D(
                    embed_dim, base_h=pos_encoding_base_size, base_w=pos_encoding_base_size
                )
            else:
                self.pos_encoding_left = PositionalEncoding2D(embed_dim)
                self.pos_encoding_right = PositionalEncoding2D(embed_dim)

        # Cross attention: left attends to right
        self.cross_attn_l2r = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Cross attention: right attends to left
        self.cross_attn_r2l = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_left1 = nn.LayerNorm(embed_dim)
        self.norm_right1 = nn.LayerNorm(embed_dim)

        self.ffn_left = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_right = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm_left2 = nn.LayerNorm(embed_dim)
        self.norm_right2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        left_feat: torch.Tensor,
        right_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention between left and right hand spatial features.

        Args:
            left_feat: [B, C, H, W] left hand spatial feature map
            right_feat: [B, C, H, W] right hand spatial feature map

        Returns:
            Tuple of (refined_left_feat, refined_right_feat), each [B, C, H, W]

        Note: Uses Pre-Norm structure (LayerNorm before attention/FFN)
        for better gradient flow in deep networks.
        """
        B, C, H, W = left_feat.shape

        # Add positional encoding (separate for left and right)
        if self.use_positional_encoding:
            left_feat = self.pos_encoding_left(left_feat)
            right_feat = self.pos_encoding_right(right_feat)

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        left_seq = left_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        right_seq = right_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Pre-Norm Cross-attention: Left attends to right (query=left, key/value=right)
        left_norm = self.norm_left1(left_seq)
        right_norm = self.norm_right1(right_seq)
        left_attn_out, _ = self.cross_attn_l2r(left_norm, right_norm, right_norm)
        left_out = left_seq + left_attn_out

        # Pre-Norm Cross-attention: Right attends to left (query=right, key/value=left)
        right_attn_out, _ = self.cross_attn_r2l(right_norm, left_norm, left_norm)
        right_out = right_seq + right_attn_out

        # Pre-Norm FFN with residual
        left_out = left_out + self.ffn_left(self.norm_left2(left_out))
        right_out = right_out + self.ffn_right(self.norm_right2(right_out))

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        left_out = left_out.permute(0, 2, 1).view(B, C, H, W)
        right_out = right_out.permute(0, 2, 1).view(B, C, H, W)

        return left_out, right_out

    def forward_seq(
        self,
        left_seq: torch.Tensor,
        right_seq: torch.Tensor,
        spatial_shape: Tuple[int, int],
        add_pos_encoding: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass that operates on sequences directly.

        Args:
            left_seq: [B, H*W, C] left sequence
            right_seq: [B, H*W, C] right sequence
            spatial_shape: (H, W) tuple for positional encoding
            add_pos_encoding: Whether to add positional encoding

        Returns:
            Tuple of (refined_left_seq, refined_right_seq), each [B, H*W, C]

        Note: Uses Pre-Norm structure (LayerNorm before attention/FFN)
        for better gradient flow in deep networks.
        """
        B, N, C = left_seq.shape
        H, W = spatial_shape

        # Add positional encoding if needed (separate for left and right)
        if add_pos_encoding and self.use_positional_encoding:
            left_spatial = left_seq.permute(0, 2, 1).view(B, C, H, W)
            right_spatial = right_seq.permute(0, 2, 1).view(B, C, H, W)
            left_spatial = self.pos_encoding_left(left_spatial)
            right_spatial = self.pos_encoding_right(right_spatial)
            left_seq = left_spatial.flatten(2).permute(0, 2, 1)
            right_seq = right_spatial.flatten(2).permute(0, 2, 1)

        # Pre-Norm Cross-attention: Left attends to right (query=left, key/value=right)
        left_norm = self.norm_left1(left_seq)
        right_norm = self.norm_right1(right_seq)
        left_attn_out, _ = self.cross_attn_l2r(left_norm, right_norm, right_norm)
        left_out = left_seq + left_attn_out

        # Pre-Norm Cross-attention: Right attends to left (query=right, key/value=left)
        right_attn_out, _ = self.cross_attn_r2l(right_norm, left_norm, left_norm)
        right_out = right_seq + right_attn_out

        # Pre-Norm FFN with residual
        left_out = left_out + self.ffn_left(self.norm_left2(left_out))
        right_out = right_out + self.ffn_right(self.norm_right2(right_out))

        return left_out, right_out


class EventEgoHandsV2Base(nn.Module):
    """
    Hand pose estimation model with YOLO-based cropping and spatial attention mechanisms.
    V2 version with enhanced features:
    - Attention Pooling instead of Global Average Pooling
    - Learnable 2D Positional Encoding with interpolation support
    - Pre-Norm structure for better gradient flow
    - Shared positional encoding for cross-attention

    - When both hands are detected: applies both Cross-Attention and Self-Attention
      based on attention_order ("cross_first" or "self_first")
    - When only one hand is detected: applies Spatial Self-Attention to that hand's features
    - Both attention mechanisms can be independently enabled/disabled via config

    Attention order (when both hands are detected and both attention types enabled):
    - "cross_first": Cross-Attention -> Self-Attention (default)
    - "self_first": Self-Attention -> Cross-Attention

    Optimized version with:
    - GPU-accelerated crop extraction using grid_sample
    - Reduced CPU-GPU data transfers
    - Optimized attention layer chaining
    """

    def __init__(
        self,
        backbone_name: str = "ResNet18",
        output_mano_dim: int = 31 * 2,
        n_channels: int = 2,
        mano_hand_model_path: str = None,
        is_weights: bool = True,
        crop_size: Tuple[int, int] = (224, 224),
        num_pose_coeffs: int = 45,
        use_pose_pca: bool = False,
        device: str = "cuda",
        bbox_expand_ratio: float = 1.0,
        use_mask: bool = False,
        mask_dilation_kernel_size: int = 5,
        mask_dilation_iterations: int = 2,
        # Attention parameters
        use_cross_attention: bool = True,
        use_self_attention: bool = True,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_learnable_pos_encoding: bool = True,  # Use learnable positional encoding
        pos_encoding_base_size: int = 14,  # Base size for positional encoding (interpolated if different)
        add_pos_encoding_every_layer: bool = True,  # Add positional encoding at every attention layer
        num_attention_layers: int = 1,
        use_shared_backbone: bool = False,  # Use shared backbone for left and right hands
        attention_order: str = "self_first",  # "cross_first" or "self_first"
    ):
        super(EventEgoHandsV2Base, self).__init__()

        self.output_mano_dim = output_mano_dim
        self.crop_size = crop_size
        self.n_channels = n_channels
        self.bbox_expand_ratio = bbox_expand_ratio
        self.use_mask = use_mask
        self.mask_dilation_kernel_size = mask_dilation_kernel_size
        self.mask_dilation_iterations = mask_dilation_iterations
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention
        self.backbone_name = backbone_name
        self.num_attention_layers = num_attention_layers
        self.use_shared_backbone = use_shared_backbone
        self.attention_order = attention_order
        self.add_pos_encoding_every_layer = add_pos_encoding_every_layer

        if attention_order not in ["cross_first", "self_first"]:
            raise ValueError(f"attention_order must be 'cross_first' or 'self_first', got {attention_order}")

        # Spatial backbones (output feature maps)
        if use_shared_backbone:
            # Shared backbone for both hands
            self.shared_backbone, out_features = get_spatial_backbone(
                name=backbone_name, n_channels=n_channels, is_weights=is_weights
            )
            self.left_out_features = out_features
            self.right_out_features = out_features
        else:
            # Separate backbones for left and right hands
            self.left_backbone, self.left_out_features = get_spatial_backbone(
                name=backbone_name, n_channels=n_channels, is_weights=is_weights
            )
            self.right_backbone, self.right_out_features = get_spatial_backbone(
                name=backbone_name, n_channels=n_channels, is_weights=is_weights
            )
        # Attention pooling to convert spatial features to vectors
        self.use_attention_pooling = use_cross_attention or use_self_attention
        if self.use_attention_pooling:
            self.left_attn_pool = AttentionPooling(self.left_out_features)
            self.right_attn_pool = AttentionPooling(self.right_out_features)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        # Feature dimension for attention modules
        feature_dim = self.left_out_features

        # Initialize spatial attention modules
        if self.use_cross_attention:
            self.spatial_cross_attention_layers = nn.ModuleList(
                [
                    SpatialCrossAttentionBlock(
                        embed_dim=feature_dim,
                        num_heads=attention_num_heads,
                        dropout=attention_dropout,
                        use_positional_encoding=use_positional_encoding,
                        use_learnable_pos_encoding=use_learnable_pos_encoding,
                        pos_encoding_base_size=pos_encoding_base_size,
                    )
                    for _ in range(num_attention_layers)
                ]
            )

        if self.use_self_attention:
            self.spatial_self_attention_left_layers = nn.ModuleList(
                [
                    SpatialSelfAttentionBlock(
                        embed_dim=feature_dim,
                        num_heads=attention_num_heads,
                        dropout=attention_dropout,
                        use_positional_encoding=use_positional_encoding,
                        use_learnable_pos_encoding=use_learnable_pos_encoding,
                        pos_encoding_base_size=pos_encoding_base_size,
                    )
                    for _ in range(num_attention_layers)
                ]
            )
            self.spatial_self_attention_right_layers = nn.ModuleList(
                [
                    SpatialSelfAttentionBlock(
                        embed_dim=feature_dim,
                        num_heads=attention_num_heads,
                        dropout=attention_dropout,
                        use_positional_encoding=use_positional_encoding,
                        use_learnable_pos_encoding=use_learnable_pos_encoding,
                        pos_encoding_base_size=pos_encoding_base_size,
                    )
                    for _ in range(num_attention_layers)
                ]
            )

        # MLP heads for MANO parameter prediction
        self.left_mlp = nn.Sequential(
            nn.Linear(self.left_out_features, self.output_mano_dim // 2),
        )
        self.right_mlp = nn.Sequential(
            nn.Linear(self.right_out_features, self.output_mano_dim // 2),
        )

        # MANO decoder and hand model
        self.mano_decoder = MANODecoder(
            n_pose_params=num_pose_coeffs,
            n_shape_params=10,
            device=device,
        )
        self.mano_hand_model = MANOHandModelCustom(
            mano_model_files_dir=mano_hand_model_path,
            num_pose_coeffs=num_pose_coeffs,
            use_pose_pca=use_pose_pca,
            device=device,
        )
        self.mano_hand_model.to(device)

    def _create_crop_grid(
        self,
        bboxes: torch.Tensor,
        img_h: int,
        img_w: int,
        crop_h: int,
        crop_w: int,
    ) -> torch.Tensor:
        """
        Create sampling grid for grid_sample from bounding boxes.

        Args:
            bboxes: [N, 4] tensor of (x1, y1, x2, y2) in pixel coordinates
            img_h, img_w: Original image dimensions
            crop_h, crop_w: Output crop dimensions

        Returns:
            grid: [N, crop_h, crop_w, 2] sampling grid in [-1, 1] normalized coordinates
        """
        N = bboxes.shape[0]
        device = bboxes.device

        # Normalize bbox coordinates to [-1, 1]
        x1 = (bboxes[:, 0] / img_w) * 2 - 1  # [N]
        y1 = (bboxes[:, 1] / img_h) * 2 - 1
        x2 = (bboxes[:, 2] / img_w) * 2 - 1
        y2 = (bboxes[:, 3] / img_h) * 2 - 1

        # Create base grid in [0, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, crop_h, device=device),
            torch.linspace(0, 1, crop_w, device=device),
            indexing='ij'
        )

        # Expand for batch: [crop_h, crop_w] -> [N, crop_h, crop_w]
        grid_x = grid_x.unsqueeze(0).expand(N, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(N, -1, -1)

        # Scale and shift grid to bbox coordinates
        # grid_x_scaled = x1 + (x2 - x1) * grid_x
        x1 = x1.view(N, 1, 1)
        x2 = x2.view(N, 1, 1)
        y1 = y1.view(N, 1, 1)
        y2 = y2.view(N, 1, 1)

        sample_x = x1 + (x2 - x1) * grid_x
        sample_y = y1 + (y2 - y1) * grid_y

        # Stack to create grid: [N, crop_h, crop_w, 2]
        grid = torch.stack([sample_x, sample_y], dim=-1)

        return grid

    def _expand_bbox_tensor(
        self,
        bboxes: torch.Tensor,
        img_h: int,
        img_w: int,
        expand_ratio: float,
    ) -> torch.Tensor:
        """
        Expand bounding boxes by the given ratio while keeping center (vectorized).

        Args:
            bboxes: [N, 4] tensor of (x1, y1, x2, y2)
            img_h, img_w: Image dimensions for clamping
            expand_ratio: Expansion ratio

        Returns:
            Expanded bboxes: [N, 4]
        """
        if expand_ratio == 1.0:
            return bboxes

        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        new_w = w * expand_ratio
        new_h = h * expand_ratio

        new_x1 = torch.clamp(cx - new_w / 2, min=0)
        new_y1 = torch.clamp(cy - new_h / 2, min=0)
        new_x2 = torch.clamp(cx + new_w / 2, max=img_w)
        new_y2 = torch.clamp(cy + new_h / 2, max=img_h)

        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

    def _get_mask_bboxes_batched(
        self,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate bounding boxes from masks (GPU-accelerated).

        Args:
            masks: [N, H, W] binary masks

        Returns:
            bboxes: [N, 4] bounding boxes (x1, y1, x2, y2)
            valid: [N] boolean tensor indicating valid masks
        """
        N, H, W = masks.shape
        device = masks.device

        # Create coordinate grids
        y_coords = torch.arange(H, device=device).view(1, H, 1).expand(N, H, W)
        x_coords = torch.arange(W, device=device).view(1, 1, W).expand(N, H, W)

        # Mask coordinates
        mask_binary = (masks > 0.5)

        # For empty masks, set large/small values that will result in invalid bbox
        large_val = H + W

        # Min/max with masking
        y_masked = torch.where(mask_binary, y_coords.float(), torch.tensor(large_val, dtype=torch.float32, device=device))
        x_masked = torch.where(mask_binary, x_coords.float(), torch.tensor(large_val, dtype=torch.float32, device=device))

        y1 = y_masked.view(N, -1).min(dim=1)[0]
        x1 = x_masked.view(N, -1).min(dim=1)[0]

        y_masked_max = torch.where(mask_binary, y_coords.float(), torch.tensor(-1.0, device=device))
        x_masked_max = torch.where(mask_binary, x_coords.float(), torch.tensor(-1.0, device=device))

        y2 = y_masked_max.view(N, -1).max(dim=1)[0]
        x2 = x_masked_max.view(N, -1).max(dim=1)[0]

        # Check validity (mask has at least one pixel)
        valid = mask_binary.view(N, -1).any(dim=1)

        bboxes = torch.stack([x1, y1, x2, y2], dim=1)

        return bboxes, valid

    def extract_hand_crops(
        self,
        lnes: torch.Tensor,
        yolo_results,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Extract and resize hand crops from YOLO detection results.
        Optimized with GPU-accelerated grid_sample.

        Args:
            lnes: [batch_size, 2, H, W] LNES tensor
            yolo_results: List of YOLO prediction results (one per batch item)

        Returns:
            left_crops, left_valid_mask, right_crops, right_valid_mask, left_bboxes, right_bboxes
        """
        batch_size = lnes.shape[0]
        device = lnes.device
        crop_h, crop_w = self.crop_size
        img_h, img_w = lnes.shape[2], lnes.shape[3]

        left_crops = torch.zeros(
            batch_size, self.n_channels, crop_h, crop_w, device=device, dtype=lnes.dtype
        )
        right_crops = torch.zeros(
            batch_size, self.n_channels, crop_h, crop_w, device=device, dtype=lnes.dtype
        )

        left_valid_mask = torch.zeros(batch_size, device=device, dtype=torch.float32)
        right_valid_mask = torch.zeros(batch_size, device=device, dtype=torch.float32)

        left_bboxes = torch.zeros(batch_size, 4, device=device, dtype=torch.float32)
        right_bboxes = torch.zeros(batch_size, 4, device=device, dtype=torch.float32)

        # Collect all detections for batch processing
        left_batch_indices = []
        left_det_bboxes = []
        left_det_masks = []

        right_batch_indices = []
        right_det_bboxes = []
        right_det_masks = []

        for batch_idx, results in enumerate(yolo_results):
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy  # Keep on GPU if available
                classes = results.boxes.cls

                # Move to target device if needed
                if boxes.device != device:
                    boxes = boxes.to(device)
                    classes = classes.to(device)

                masks = None
                if (
                    self.use_mask
                    and hasattr(results, "masks")
                    and results.masks is not None
                    and len(results.masks) > 0
                ):
                    masks = results.masks.data
                    if masks.device != device:
                        masks = masks.to(device)
                    # Resize masks if needed
                    if masks.shape[-2:] != (img_h, img_w):
                        masks = F.interpolate(
                            masks.unsqueeze(1).float(),
                            size=(img_h, img_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)

                for det_idx in range(len(boxes)):
                    cls = int(classes[det_idx].item())
                    box = boxes[det_idx]

                    if cls == 0:  # Left hand
                        left_batch_indices.append(batch_idx)
                        left_det_bboxes.append(box)
                        if masks is not None:
                            left_det_masks.append(masks[det_idx])
                        else:
                            left_det_masks.append(None)
                    elif cls == 1:  # Right hand
                        right_batch_indices.append(batch_idx)
                        right_det_bboxes.append(box)
                        if masks is not None:
                            right_det_masks.append(masks[det_idx])
                        else:
                            right_det_masks.append(None)

        # Process left hand detections
        if left_batch_indices:
            left_crops, left_valid_mask, left_bboxes = self._process_hand_crops(
                lnes, left_batch_indices, left_det_bboxes, left_det_masks,
                left_crops, left_valid_mask, left_bboxes,
                img_h, img_w, crop_h, crop_w
            )

        # Process right hand detections
        if right_batch_indices:
            right_crops, right_valid_mask, right_bboxes = self._process_hand_crops(
                lnes, right_batch_indices, right_det_bboxes, right_det_masks,
                right_crops, right_valid_mask, right_bboxes,
                img_h, img_w, crop_h, crop_w
            )

        return (
            left_crops,
            left_valid_mask,
            right_crops,
            right_valid_mask,
            left_bboxes,
            right_bboxes,
        )

    def _process_hand_crops(
        self,
        lnes: torch.Tensor,
        batch_indices: List[int],
        det_bboxes: List[torch.Tensor],
        det_masks: List[Optional[torch.Tensor]],
        crops: torch.Tensor,
        valid_mask: torch.Tensor,
        bboxes_out: torch.Tensor,
        img_h: int,
        img_w: int,
        crop_h: int,
        crop_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process hand crops for one hand type (left or right).
        """
        device = lnes.device

        # Stack bboxes
        stacked_bboxes = torch.stack(det_bboxes, dim=0)  # [n_dets, 4]

        # Process masks if available
        has_masks = self.use_mask and any(m is not None for m in det_masks)

        if has_masks:
            # Stack masks (use zero mask for None)
            mask_list = []
            for m in det_masks:
                if m is not None:
                    mask_list.append(m)
                else:
                    mask_list.append(torch.zeros(img_h, img_w, device=device))
            stacked_masks = torch.stack(mask_list, dim=0)  # [n_dets, H, W]

            # Apply dilation on GPU
            dilated_masks = dilation_masks_gpu(
                stacked_masks,
                kernel_size=self.mask_dilation_kernel_size,
                iterations=self.mask_dilation_iterations
            )

            # Get bboxes from masks
            mask_bboxes, mask_valid = self._get_mask_bboxes_batched(dilated_masks)

            # Use mask bbox where valid, otherwise expand original bbox
            expanded_bboxes = self._expand_bbox_tensor(
                stacked_bboxes, img_h, img_w, self.bbox_expand_ratio
            )
            final_bboxes = torch.where(
                mask_valid.unsqueeze(1).expand(-1, 4),
                mask_bboxes,
                expanded_bboxes
            )
        else:
            dilated_masks = None
            final_bboxes = self._expand_bbox_tensor(
                stacked_bboxes, img_h, img_w, self.bbox_expand_ratio
            )

        # Create sampling grid for all detections
        grid = self._create_crop_grid(final_bboxes, img_h, img_w, crop_h, crop_w)

        # Gather input images for each detection
        batch_idx_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
        input_imgs = lnes[batch_idx_tensor]  # [n_dets, C, H, W]

        # Apply grid_sample
        cropped = F.grid_sample(
            input_imgs,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [n_dets, C, crop_h, crop_w]

        # Apply mask if available
        if has_masks and dilated_masks is not None:
            # Crop the masks using same grid
            mask_cropped = F.grid_sample(
                dilated_masks.unsqueeze(1),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            ).squeeze(1)  # [n_dets, crop_h, crop_w]

            # Apply mask
            cropped = cropped * (mask_cropped > 0.5).unsqueeze(1).float()

        # Scatter results back to output tensors
        for i, batch_idx in enumerate(batch_indices):
            crops[batch_idx] = cropped[i]
            valid_mask[batch_idx] = 1.0
            bboxes_out[batch_idx] = final_bboxes[i]

        return crops, valid_mask, bboxes_out

    def forward(
        self,
        lnes: torch.Tensor,
        yolo_results,
    ):
        """
        Forward pass with YOLO-based cropping and spatial attention mechanisms.

        Args:
            lnes: [batch_size, 2, H, W] LNES tensor
            yolo_results: List of YOLO prediction results (one per batch item)

        Returns:
            outputs: Dictionary with left/right MANO outputs and validity masks
        """
        (
            left_crops,
            left_valid_mask,
            right_crops,
            right_valid_mask,
            left_bboxes,
            right_bboxes,
        ) = self.extract_hand_crops(lnes, yolo_results)


        x_left_feat, x_right_feat = self._forward_spatial_attention(
            left_crops,
            right_crops,
            left_valid_mask,
            right_valid_mask,
        )

        # Apply MLP to get MANO parameters
        left_mano_parameters = self.left_mlp(x_left_feat)
        left_outputs = self.mano_decoder(
            left_mano_parameters, "left", self.mano_hand_model
        )

        right_mano_parameters = self.right_mlp(x_right_feat)
        right_outputs = self.mano_decoder(
            right_mano_parameters, "right", self.mano_hand_model
        )

        outputs = {
            "left": left_outputs,
            "right": right_outputs,
            "left_valid": left_valid_mask,
            "right_valid": right_valid_mask,
            "left_bboxes": left_bboxes,
            "right_bboxes": right_bboxes,
        }
        return outputs

    def _forward_spatial_attention(
        self,
        left_crops: torch.Tensor,
        right_crops: torch.Tensor,
        left_valid: torch.Tensor,
        right_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using spatial feature maps for attention.
        """
        # Extract spatial feature maps (before GAP)
        if self.use_shared_backbone:
            left_spatial = self.shared_backbone(left_crops)  # [B, C, H, W]
            right_spatial = self.shared_backbone(right_crops)  # [B, C, H, W]
        else:
            left_spatial = self.left_backbone(left_crops)  # [B, C, H, W]
            right_spatial = self.right_backbone(right_crops)  # [B, C, H, W]

        # Apply spatial attention
        left_spatial, right_spatial = self._apply_spatial_attention(
            left_spatial,
            right_spatial,
            left_valid,
            right_valid,
        )

        # Attention pooling to get feature vectors
        if self.use_attention_pooling:
            left_feat = self.left_attn_pool(left_spatial)
            right_feat = self.right_attn_pool(right_spatial)
        else:
            left_feat = self.gap(left_spatial).flatten(1)
            right_feat = self.gap(right_spatial).flatten(1)

        return left_feat, right_feat

    def _apply_spatial_attention(
        self,
        left_spatial: torch.Tensor,
        right_spatial: torch.Tensor,
        left_valid: torch.Tensor,
        right_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention mechanisms based on hand detection status.
        Optimized to avoid unnecessary clone operations.

        When both hands are detected and both attention types are enabled:
        - attention_order="cross_first": Cross-Attention -> Self-Attention
        - attention_order="self_first": Self-Attention -> Cross-Attention
        """
        B, C, H, W = left_spatial.shape
        spatial_shape = (H, W)

        both_valid = (left_valid > 0.5) & (right_valid > 0.5)
        only_left_valid = (left_valid > 0.5) & (right_valid <= 0.5)
        only_right_valid = (left_valid <= 0.5) & (right_valid > 0.5)

        # Check if any attention will be applied
        apply_cross = self.use_cross_attention and both_valid.any()
        apply_self_both = self.use_self_attention and both_valid.any()
        apply_self_left = self.use_self_attention and only_left_valid.any()
        apply_self_right = self.use_self_attention and only_right_valid.any()

        # Only clone if we need to modify
        if apply_cross or apply_self_both or apply_self_left:
            out_left = left_spatial.clone()
        else:
            out_left = left_spatial

        if apply_cross or apply_self_both or apply_self_right:
            out_right = right_spatial.clone()
        else:
            out_right = right_spatial

        # Process samples where both hands are detected
        if both_valid.any() and (apply_cross or apply_self_both):
            both_indices = both_valid.nonzero(as_tuple=True)[0]
            left_both = left_spatial[both_indices]
            right_both = right_spatial[both_indices]

            if self.attention_order == "cross_first":
                # Cross-Attention first, then Self-Attention
                if apply_cross:
                    left_both, right_both = self._apply_cross_attention_layers(
                        left_both, right_both, spatial_shape, C, H, W
                    )
                if apply_self_both:
                    left_both = self._apply_self_attention_layers(
                        left_both, spatial_shape, C, H, W, hand="left"
                    )
                    right_both = self._apply_self_attention_layers(
                        right_both, spatial_shape, C, H, W, hand="right"
                    )
            else:  # self_first
                # Self-Attention first, then Cross-Attention
                if apply_self_both:
                    left_both = self._apply_self_attention_layers(
                        left_both, spatial_shape, C, H, W, hand="left"
                    )
                    right_both = self._apply_self_attention_layers(
                        right_both, spatial_shape, C, H, W, hand="right"
                    )
                if apply_cross:
                    left_both, right_both = self._apply_cross_attention_layers(
                        left_both, right_both, spatial_shape, C, H, W
                    )

            out_left[both_indices] = left_both
            out_right[both_indices] = right_both

        # Self-Attention for samples where only left hand is detected
        if apply_self_left:
            left_only_indices = only_left_valid.nonzero(as_tuple=True)[0]
            left_only = left_spatial[left_only_indices]
            left_only = self._apply_self_attention_layers(
                left_only, spatial_shape, C, H, W, hand="left"
            )
            out_left[left_only_indices] = left_only

        # Self-Attention for samples where only right hand is detected
        if apply_self_right:
            right_only_indices = only_right_valid.nonzero(as_tuple=True)[0]
            right_only = right_spatial[right_only_indices]
            right_only = self._apply_self_attention_layers(
                right_only, spatial_shape, C, H, W, hand="right"
            )
            out_right[right_only_indices] = right_only

        return out_left, out_right

    def _apply_cross_attention_layers(
        self,
        left_feat: torch.Tensor,
        right_feat: torch.Tensor,
        spatial_shape: Tuple[int, int],
        C: int,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention layers to left and right features.
        """
        if self.num_attention_layers > 1:
            # Convert to sequence once
            left_seq = left_feat.flatten(2).permute(0, 2, 1)
            right_seq = right_feat.flatten(2).permute(0, 2, 1)

            for i, layer in enumerate(self.spatial_cross_attention_layers):
                # Add positional encoding at every layer if configured, otherwise only first
                add_pos = self.add_pos_encoding_every_layer or (i == 0)
                left_seq, right_seq = layer.forward_seq(
                    left_seq, right_seq, spatial_shape,
                    add_pos_encoding=add_pos
                )

            # Convert back to spatial
            left_feat = left_seq.permute(0, 2, 1).view(-1, C, H, W)
            right_feat = right_seq.permute(0, 2, 1).view(-1, C, H, W)
        else:
            for layer in self.spatial_cross_attention_layers:
                left_feat, right_feat = layer(left_feat, right_feat)

        return left_feat, right_feat

    def _apply_self_attention_layers(
        self,
        feat: torch.Tensor,
        spatial_shape: Tuple[int, int],
        C: int,
        H: int,
        W: int,
        hand: str,
    ) -> torch.Tensor:
        """
        Apply self-attention layers to features.

        Args:
            feat: [N, C, H, W] spatial features
            spatial_shape: (H, W) tuple
            C, H, W: feature dimensions
            hand: "left" or "right" to select the appropriate layers
        """
        layers = (
            self.spatial_self_attention_left_layers
            if hand == "left"
            else self.spatial_self_attention_right_layers
        )

        if self.num_attention_layers > 1:
            feat_seq = feat.flatten(2).permute(0, 2, 1)

            for i, layer in enumerate(layers):
                # Add positional encoding at every layer if configured, otherwise only first
                add_pos = self.add_pos_encoding_every_layer or (i == 0)
                feat_seq = layer.forward_seq(
                    feat_seq, spatial_shape,
                    add_pos_encoding=add_pos
                )

            feat = feat_seq.permute(0, 2, 1).view(-1, C, H, W)
        else:
            for layer in layers:
                feat = layer(feat)

        return feat



class EventEgoHandsV2(EventEgoHandsV2Base):
    """
    Hand pose estimation model that applies YOLO segmentation masks to the full image
    instead of cropping. The masked full-size image is passed to the backbone.

    Inherits from EventEgoHandsV2Base and overrides:
    - extract_hand_masked_images(): replaces extract_hand_crops()
    - forward(): uses masked images instead of crops, does not output bboxes
    """

    def extract_hand_masked_images(
        self,
        lnes: torch.Tensor,
        yolo_results,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract hand-masked full images from YOLO detection results.
        Instead of cropping, applies segmentation masks to zero out non-hand regions.

        Args:
            lnes: [batch_size, 2, H, W] LNES tensor
            yolo_results: List of YOLO prediction results (one per batch item)

        Returns:
            left_masked: [batch_size, 2, H, W] masked images for left hand
            left_valid_mask: [batch_size] validity mask
            right_masked: [batch_size, 2, H, W] masked images for right hand
            right_valid_mask: [batch_size] validity mask
        """
        batch_size = lnes.shape[0]
        device = lnes.device
        img_h, img_w = lnes.shape[2], lnes.shape[3]

        left_masked = torch.zeros_like(lnes)
        right_masked = torch.zeros_like(lnes)

        left_valid_mask = torch.zeros(batch_size, device=device, dtype=torch.float32)
        right_valid_mask = torch.zeros(batch_size, device=device, dtype=torch.float32)

        # Collect detections per batch item
        left_batch_indices = []
        left_det_masks = []
        left_det_bboxes = []

        right_batch_indices = []
        right_det_masks = []
        right_det_bboxes = []

        for batch_idx, results in enumerate(yolo_results):
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy
                classes = results.boxes.cls

                if boxes.device != device:
                    boxes = boxes.to(device)
                    classes = classes.to(device)

                masks = None
                if (
                    hasattr(results, "masks")
                    and results.masks is not None
                    and len(results.masks) > 0
                ):
                    masks = results.masks.data
                    if masks.device != device:
                        masks = masks.to(device)
                    # Resize masks if needed
                    if masks.shape[-2:] != (img_h, img_w):
                        masks = F.interpolate(
                            masks.unsqueeze(1).float(),
                            size=(img_h, img_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)

                for det_idx in range(len(boxes)):
                    cls = int(classes[det_idx].item())
                    box = boxes[det_idx]

                    if cls == 0:  # Left hand
                        if batch_idx not in left_batch_indices:
                            left_batch_indices.append(batch_idx)
                            if masks is not None:
                                left_det_masks.append(masks[det_idx])
                            else:
                                left_det_masks.append(None)
                            left_det_bboxes.append(box)
                    elif cls == 1:  # Right hand
                        if batch_idx not in right_batch_indices:
                            right_batch_indices.append(batch_idx)
                            if masks is not None:
                                right_det_masks.append(masks[det_idx])
                            else:
                                right_det_masks.append(None)
                            right_det_bboxes.append(box)

        # Process left hand
        if left_batch_indices:
            left_masked, left_valid_mask = self._apply_masks_to_images(
                lnes,
                left_batch_indices,
                left_det_masks,
                left_det_bboxes,
                left_masked,
                left_valid_mask,
                img_h,
                img_w,
            )

        # Process right hand
        if right_batch_indices:
            right_masked, right_valid_mask = self._apply_masks_to_images(
                lnes,
                right_batch_indices,
                right_det_masks,
                right_det_bboxes,
                right_masked,
                right_valid_mask,
                img_h,
                img_w,
            )

        return left_masked, left_valid_mask, right_masked, right_valid_mask

    def _apply_masks_to_images(
        self,
        lnes: torch.Tensor,
        batch_indices: List[int],
        det_masks: List[Optional[torch.Tensor]],
        det_bboxes: List[torch.Tensor],
        masked_out: torch.Tensor,
        valid_mask: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply segmentation masks (or bbox-derived rectangular masks) to full images.

        Args:
            lnes: [B, C, H, W] full LNES images
            batch_indices: list of batch indices with detections
            det_masks: list of segmentation masks (or None for bbox fallback)
            det_bboxes: list of bounding boxes [4] tensors
            masked_out: [B, C, H, W] output tensor to fill
            valid_mask: [B] validity tensor to fill
            img_h, img_w: image dimensions

        Returns:
            masked_out, valid_mask (updated)
        """
        device = lnes.device
        n_dets = len(batch_indices)

        # Build mask tensor for all detections
        mask_list = []
        for i in range(n_dets):
            if det_masks[i] is not None:
                mask_list.append(det_masks[i])
            else:
                # Fallback: create rectangular mask from bbox
                rect_mask = torch.zeros(img_h, img_w, device=device)
                box = det_bboxes[i]
                x1 = max(0, int(box[0].item()))
                y1 = max(0, int(box[1].item()))
                x2 = min(img_w, int(box[2].item()))
                y2 = min(img_h, int(box[3].item()))
                rect_mask[y1:y2, x1:x2] = 1.0
                mask_list.append(rect_mask)

        stacked_masks = torch.stack(mask_list, dim=0)  # [n_dets, H, W]

        # Apply dilation
        dilated_masks = dilation_masks_gpu(
            stacked_masks,
            kernel_size=self.mask_dilation_kernel_size,
            iterations=self.mask_dilation_iterations,
        )

        # Apply binary mask to full images
        binary_masks = (dilated_masks > 0.5).unsqueeze(1).float()  # [n_dets, 1, H, W]

        batch_idx_tensor = torch.tensor(
            batch_indices, device=device, dtype=torch.long
        )
        input_imgs = lnes[batch_idx_tensor]  # [n_dets, C, H, W]

        masked_imgs = input_imgs * binary_masks  # [n_dets, C, H, W]

        # Scatter back
        for i, batch_idx in enumerate(batch_indices):
            masked_out[batch_idx] = masked_imgs[i]
            valid_mask[batch_idx] = 1.0

        return masked_out, valid_mask

    def forward(
        self,
        lnes: torch.Tensor,
        yolo_results,
    ):
        """
        Forward pass with YOLO mask-based full image input.

        Args:
            lnes: [batch_size, 2, H, W] LNES tensor
            yolo_results: List of YOLO prediction results (one per batch item)

        Returns:
            outputs: Dictionary with left/right MANO outputs and validity masks.
                     Does not include bboxes (loss uses original coordinates).
        """
        (
            left_masked,
            left_valid_mask,
            right_masked,
            right_valid_mask,
        ) = self.extract_hand_masked_images(lnes, yolo_results)

        x_left_feat, x_right_feat = self._forward_spatial_attention(
            left_masked,
            right_masked,
            left_valid_mask,
            right_valid_mask,
        )

        # Apply MLP to get MANO parameters
        left_mano_parameters = self.left_mlp(x_left_feat)
        left_outputs = self.mano_decoder(
            left_mano_parameters, "left", self.mano_hand_model
        )

        right_mano_parameters = self.right_mlp(x_right_feat)
        right_outputs = self.mano_decoder(
            right_mano_parameters, "right", self.mano_hand_model
        )

        outputs = {
            "left": left_outputs,
            "right": right_outputs,
            "left_valid": left_valid_mask,
            "right_valid": right_valid_mask,
        }
        return outputs
