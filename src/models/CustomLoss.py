import torch
import torch.nn.functional as F
from torch import nn
import collections
from typing import Optional, Tuple
from utils.config import HandEstimationConfig

from models.MANOHandModelCustom import MANOHandModelCustom


# Wrist joint index in HOT3D format (mano_joint_mapping[5] = 0 = wrist)
WRIST_INDEX = 5

# Fingertip indices in HOT3D format (0-4: thumb, index, middle, ring, pinky)
FINGERTIP_INDICES = [0, 1, 2, 3, 4]

# Bone connections in HOT3D format: (child, parent)
HOT3D_BONE_CONNECTIONS = [
    (0, 6), (6, 7), (7, 5),                   # Thumb (3 bones)
    (1, 8), (8, 9), (9, 10), (10, 5),         # Index (4 bones)
    (2, 11), (11, 12), (12, 13), (13, 5),     # Middle (4 bones)
    (3, 14), (14, 15), (15, 16), (16, 5),     # Ring (4 bones)
    (4, 17), (17, 18), (18, 19), (19, 5),     # Pinky (4 bones)
]

# MoCap 16-joint order:
# 0-2:index, 3-5:middle, 6-8:pinky, 9-11:ring, 12-14:thumb, 15:wrist
# HOT3D20 -> MoCap16 direct mapping for common joints (excluding Thumb CMC).
# MoCap16 order:
# 0-2:index, 3-5:middle, 6-8:pinky, 9-11:ring, 12:thumb_cmc, 13:thumb_mcp, 14:thumb_ip, 15:wrist
HOT3D20_TO_MOCAP16_COMMON_WITHOUT_CMC_INDICES = [
    8, 9, 10,      # index
    11, 12, 13,    # middle
    17, 18, 19,    # pinky
    14, 15, 16,    # ring
    6, 7,          # thumb mcp/ip
    5,             # wrist
]
# MoCap16 -> Common15 mapping (exclude Thumb CMC index 12)
MOCAP16_TO_COMMON15_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]
COMMON15_WRIST_INDEX = 14
COMMON15_FINGERTIP_INDICES = [13, 2, 5, 11, 8]  # thumb(IP), index, middle, ring, pinky
COMMON15_BONE_CONNECTIONS = [
    (12, 14), (13, 12),  # Thumb: MCP -> Wrist, IP -> MCP
    (0, 14), (1, 0), (2, 1),  # Index
    (3, 14), (4, 3), (5, 4),  # Middle
    (9, 14), (10, 9), (11, 10),  # Ring
    (6, 14), (7, 6), (8, 7),  # Pinky
]
MOCAP16_ALLOWED_LOSSES = [
    "joints_3d",
    "joints_3d_relative",
    "inter_j3d",
    "fingertip_3d",
    "bone_length",
]

# Available loss types
AVAILABLE_LOSSES = [
    "vertices",
    "joints_3d",
    "joints_3d_relative",
    "joints_2d",
    "hand_pose",
    "global_orient",
    "shape",
    "transl",
    "inter_j3d",
    "inter_shape",
    "inter_global_orient",
    "inter_transl",
    "fingertip_3d",
    "bone_length",
]


class CustomLoss(nn.Module):
    def __init__(
        self,
        device,
        cfg: HandEstimationConfig,
        hand_pose_dim: int = 15,
        mano_hand_model_path: str = None,
        crop_size: Tuple[int, int] = (224, 224),
        use_yolo_valid_mask: bool = False,
    ) -> None:
        super(CustomLoss, self).__init__()
        self.device = device
        self.hand_pose_dim = hand_pose_dim
        self.crop_size = crop_size
        self.scale = 1000.0
        self.use_yolo_valid_mask = use_yolo_valid_mask
        self.annotation_source = getattr(cfg, "annotation_source", "mano")

        # Get enabled losses from config
        self.enabled_losses = cfg.train.losses

        # Validate enabled losses
        for loss_name in self.enabled_losses:
            if loss_name not in AVAILABLE_LOSSES:
                raise ValueError(
                    f"Loss '{loss_name}' is not available. "
                    f"Available losses: {AVAILABLE_LOSSES}"
                )

        if str(self.annotation_source).lower() == "mocap":
            filtered_losses = [l for l in self.enabled_losses if l in MOCAP16_ALLOWED_LOSSES]
            dropped_losses = [l for l in self.enabled_losses if l not in MOCAP16_ALLOWED_LOSSES]
            if len(dropped_losses) > 0:
                print(
                    "[CustomLoss] annotation_source=mocap: disabled unsupported losses: "
                    f"{dropped_losses}"
                )
            self.enabled_losses = filtered_losses

        # Loss ratios
        self.loss_vertices_ratio = getattr(cfg.train, "loss_vertices_ratio", 1.0)
        self.loss_j3d_ratio = getattr(cfg.train, "loss_j3d_ratio", 1.0)
        self.loss_j3d_relative_ratio = getattr(cfg.train, "loss_j3d_relative_ratio", 1.0)
        self.loss_j2d_ratio = getattr(cfg.train, "loss_j2d_ratio", 1.0)
        self.loss_hand_pose_ratio = getattr(cfg.train, "loss_hand_pose_ratio", 1.0)
        self.loss_global_orient_ratio = getattr(cfg.train, "loss_global_orient_ratio", 1.0)
        self.loss_shape_ratio = getattr(cfg.train, "loss_shape_ratio", 1.0)
        self.loss_transl_ratio = getattr(cfg.train, "loss_transl_ratio", 1.0)
        self.loss_inter_j3d_ratio = getattr(cfg.train, "loss_inter_j3d_ratio", 1.0)
        self.loss_inter_shape_ratio = getattr(cfg.train, "loss_inter_shape_ratio", 1.0)
        self.loss_inter_global_orient_ratio = getattr(cfg.train, "loss_inter_global_orient_ratio", 1.0)
        self.loss_inter_transl_ratio = getattr(cfg.train, "loss_inter_transl_ratio", 1.0)
        self.loss_fingertip_3d_ratio = getattr(cfg.train, "loss_fingertip_3d_ratio", 1.0)
        self.loss_bone_length_ratio = getattr(cfg.train, "loss_bone_length_ratio", 1.0)

        # Bone connections tensor for efficient computation
        self.bone_connections = torch.tensor(HOT3D_BONE_CONNECTIONS, dtype=torch.long)
        self.hot3d20_to_mocap16_common_without_cmc_indices = torch.tensor(
            HOT3D20_TO_MOCAP16_COMMON_WITHOUT_CMC_INDICES, dtype=torch.long
        )
        self.mocap16_to_common15_indices = torch.tensor(
            MOCAP16_TO_COMMON15_INDICES, dtype=torch.long
        )
        self.common15_bone_connections = torch.tensor(
            COMMON15_BONE_CONNECTIONS, dtype=torch.long
        )

        self.focal_length = cfg.focal_length

        # Initialize MANO model for PCA conversion if hand_pose_dim is 45
        self.mano_model_custom = None
        if hand_pose_dim == 45 and mano_hand_model_path is not None:
            self.mano_model_custom = MANOHandModelCustom(
                mano_model_files_dir=mano_hand_model_path,
                num_pose_coeffs=15,
                use_pose_pca=True,
                device=device,
            )

    def is_loss_enabled(self, loss_name: str) -> bool:
        """Check if a loss is enabled in config."""
        return loss_name in self.enabled_losses

    def _is_mocap16_batch(self, targets) -> bool:
        """Check whether batch annotation format is mocap16."""
        annotation_format = targets.get("annotation_format", "mano20")
        if isinstance(annotation_format, list):
            if len(annotation_format) == 0:
                return False
            return str(annotation_format[0]) == "mocap16"
        return str(annotation_format) == "mocap16"

    def _to_mocap_common15_joints(self, pred_j3d: torch.Tensor) -> torch.Tensor:
        """Convert predicted HOT3D20 joints to MoCap common15 order."""
        indices = self.hot3d20_to_mocap16_common_without_cmc_indices.to(pred_j3d.device)
        return pred_j3d[:, indices, :]

    def _mocap16_gt_to_common15(self, gt_j3d: torch.Tensor) -> torch.Tensor:
        """Convert MoCap16 GT joints to common15 order by dropping Thumb CMC."""
        indices = self.mocap16_to_common15_indices.to(gt_j3d.device)
        return gt_j3d[:, indices, :]

    def compute_bone_lengths(
        self, joints_3d: torch.Tensor, bone_connections: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute bone lengths from 3D joint positions.

        Args:
            joints_3d: [B, N, 3] tensor of 3D joint positions

        Returns:
            bone_lengths: [B, num_bones] tensor of bone lengths
        """
        if bone_connections is None:
            bone_connections = self.bone_connections
        bone_connections = bone_connections.to(joints_3d.device)
        child_pos = joints_3d[:, bone_connections[:, 0], :]
        parent_pos = joints_3d[:, bone_connections[:, 1], :]
        bone_lengths = torch.norm(child_pos - parent_pos, dim=-1)
        return bone_lengths

    def transform_original_to_crop(self, joints_2d_original, bbox, crop_size):
        """
        Transform 2D joints from original image coordinate system to crop coordinate system.

        Args:
            joints_2d_original: [B, N, 2] tensor of 2D joint positions in original image coordinates
            bbox: [B, 4] tensor with [x1, y1, x2, y2] bounding box in original image
            crop_size: (crop_h, crop_w) size of the crop

        Returns:
            joints_2d_crop: [B, N, 2] tensor of 2D joint positions in crop coordinates
        """
        batch_size = joints_2d_original.shape[0]
        crop_h, crop_w = crop_size

        joints_2d_crop = torch.zeros_like(joints_2d_original)

        for b in range(batch_size):
            x1, y1, x2, y2 = bbox[b]

            # Calculate scale factors
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Avoid division by zero
            if bbox_w <= 0 or bbox_h <= 0:
                # Invalid bbox, keep joints as is (will be masked out by valid flag)
                joints_2d_crop[b] = joints_2d_original[b]
                continue

            scale_x = crop_w / bbox_w
            scale_y = crop_h / bbox_h

            # Transform: translate then scale
            # (original - bbox_top_left) * scale
            joints_2d_crop[b, :, 0] = (joints_2d_original[b, :, 0] - x1) * scale_x
            joints_2d_crop[b, :, 1] = (joints_2d_original[b, :, 1] - y1) * scale_y

        return joints_2d_crop

    def perspective_projection(
        self,
        points: torch.Tensor,
        translation: torch.Tensor,
        focal_length: torch.Tensor,
        camera_center: Optional[torch.Tensor] = None,
        rotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the perspective projection of a set of 3D points.
        Args:
            points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
            translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
            focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
            camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
            rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
        Returns:
            torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
        """
        batch_size = points.shape[0]
        if rotation is None:
            rotation = (
                torch.eye(3, device=points.device, dtype=points.dtype)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
        if camera_center is None:
            camera_center = torch.zeros(
                batch_size, 2, device=points.device, dtype=points.dtype
            )
        # Populate intrinsic camera matrix K.
        K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
        K[:, 0, 0] = focal_length[:, 0]
        K[:, 1, 1] = focal_length[:, 1]
        K[:, 2, 2] = 1.0
        K[:, :-1, -1] = camera_center

        # Transform points
        points = torch.einsum("bij,bkj->bki", rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion with safety check for zero depth
        z_values = points[:, :, -1].unsqueeze(-1)
        # Prevent division by zero or very small values
        z_values_safe = torch.where(
            torch.abs(z_values) < 1e-6,
            torch.sign(z_values) * 1e-6 + (z_values == 0).float() * 1e-6,
            z_values,
        )
        projected_points = points / z_values_safe

        # Apply camera intrinsics
        projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

        return projected_points[:, :, :-1]

    def index_losss(self, loss_fn, outs, targets, indices):
        indices = indices.int()

        if indices.sum() == 0:
            return 0

        loss = loss_fn(outs, targets, reduction="none")
        per_sample_loss = loss.view(loss.size(0), -1).mean(dim=1)

        return per_sample_loss[indices].mean()

    # def index_losss(self, loss_fn, outs, targets, indices):
    #     indices = indices.int()

    #     if indices.sum() == 0:
    #         return 0

    #     loss = loss_fn(outs, targets, reduction="none")
    #     B = loss.shape[0]
    #     loss = loss.reshape(B, -1)
    #     D = loss.shape[1]
    #     indices = indices[:, None].repeat(1, D)

    #     index_loss = loss * indices
    #     index_mean_loss = index_loss.sum() / indices.sum()

    #     return index_mean_loss

    def forward(self, outs, targets):
        losses = collections.defaultdict(float)
        gt_hand_poses = targets["gt_hand_poses"]
        gt_joints = targets["gt_joints"]
        is_mocap16 = self._is_mocap16_batch(targets)

        for hand_type in ["left", "right"]:
            gt_hand_poses[hand_type]["vertices"] = gt_joints[hand_type]["vertices_3d"]
            gt_hand_poses[hand_type]["j3d"] = gt_joints[hand_type]["joints_3d"]

        # Interacting hand losses (bilateral constraints)
        interacting_indices = torch.sum(gt_hand_poses["handedness"], 1) == 2

        if (not is_mocap16) and self.is_loss_enabled("inter_shape"):
            losses["loss_inter_shape"] = (
                self.index_losss(
                    F.mse_loss,
                    outs["left"]["betas"],
                    outs["right"]["betas"],
                    interacting_indices,
                )
                * self.loss_inter_shape_ratio
            )

        if (not is_mocap16) and self.is_loss_enabled("inter_global_orient"):
            losses["loss_inter_global_orient"] = (
                self.index_losss(
                    F.mse_loss,
                    outs["left"]["global_orient"] - outs["right"]["global_orient"],
                    (
                        gt_hand_poses["left"]["global_orient"]
                        - gt_hand_poses["right"]["global_orient"]
                    ),
                    interacting_indices,
                )
                * self.loss_inter_global_orient_ratio
            )

        if (not is_mocap16) and self.is_loss_enabled("inter_transl"):
            losses["loss_inter_transl"] = (
                self.index_losss(
                    F.mse_loss,
                    outs["left"]["transl"] - outs["right"]["transl"],
                    (gt_hand_poses["left"]["trans"] - gt_hand_poses["right"]["trans"]),
                    interacting_indices,
                )
                * self.loss_inter_transl_ratio
            )

        if self.is_loss_enabled("inter_j3d"):
            left_pred_j3d = outs["left"]["j3d"]
            right_pred_j3d = outs["right"]["j3d"]
            left_gt_j3d = gt_hand_poses["left"]["j3d"]
            right_gt_j3d = gt_hand_poses["right"]["j3d"]
            if is_mocap16:
                left_pred_j3d = self._to_mocap_common15_joints(left_pred_j3d)
                right_pred_j3d = self._to_mocap_common15_joints(right_pred_j3d)
                left_gt_j3d = self._mocap16_gt_to_common15(left_gt_j3d)
                right_gt_j3d = self._mocap16_gt_to_common15(right_gt_j3d)

            losses["loss_inter_j3d"] = (
                self.index_losss(
                    F.mse_loss,
                    (left_pred_j3d - right_pred_j3d),
                    (left_gt_j3d - right_gt_j3d),
                    interacting_indices,
                )
                * self.loss_inter_j3d_ratio
            )

        # Per-hand losses
        for hand_type in ["left", "right"]:
            indices = gt_hand_poses[hand_type]["valid"]
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(indices).to(self.device)

            # Apply YOLO valid mask if enabled
            if self.use_yolo_valid_mask and f"{hand_type}_valid" in outs:
                yolo_valid = outs[f"{hand_type}_valid"]  # [batch_size]
                # Combine GT valid and YOLO valid (both must be valid)
                indices = indices * yolo_valid

            pred_j3d = outs[hand_type]["j3d"]
            gt_j3d = gt_hand_poses[hand_type]["j3d"]
            if is_mocap16:
                pred_j3d = self._to_mocap_common15_joints(pred_j3d)
                gt_j3d = self._mocap16_gt_to_common15(gt_j3d)

            # Vertices loss
            if (not is_mocap16) and self.is_loss_enabled("vertices"):
                losses["loss_vertices"] += (
                    self.index_losss(
                        F.l1_loss,
                        outs[hand_type]["vertices"] * self.scale,
                        gt_hand_poses[hand_type]["vertices"] * self.scale,
                        indices,
                    )
                    * self.loss_vertices_ratio
                )

            # Global orientation loss
            if (not is_mocap16) and self.is_loss_enabled("global_orient"):
                losses["loss_global_orient"] += (
                    self.index_losss(
                        F.mse_loss,
                        outs[hand_type]["global_orient"],
                        gt_hand_poses[hand_type]["global_orient"],
                        indices,
                    )
                    * self.loss_global_orient_ratio
                )

            # Hand pose loss with PCA conversion if needed
            if (not is_mocap16) and self.is_loss_enabled("hand_pose"):
                pred_hand_pose = outs[hand_type]["hand_pose"]
                gt_hand_pose = gt_hand_poses[hand_type]["hand_pose"]

                # If output is 45D and GT is 15D, convert GT to 45D using PCA
                if (
                    self.hand_pose_dim == 45
                    and self.mano_model_custom is not None
                    and gt_hand_pose.shape[-1] == 15
                ):
                    # Determine hand side
                    is_right_hand = hand_type == "right"

                    # Convert GT from PCA (15D) to full (45D)
                    gt_hand_pose_full = (
                        self.mano_model_custom.convert_hand_pose_pca_to_full(
                            hand_pose_pca=gt_hand_pose,
                            is_right_hand=is_right_hand,
                        )
                    )
                    gt_hand_pose = gt_hand_pose_full

                # Ensure GT matches the expected dimension
                if gt_hand_pose.shape[-1] > self.hand_pose_dim:
                    gt_hand_pose = gt_hand_pose[:, : self.hand_pose_dim]

                losses["loss_hand_pose"] += (
                    self.index_losss(
                        F.mse_loss,
                        pred_hand_pose,
                        gt_hand_pose,
                        indices,
                    )
                    * self.loss_hand_pose_ratio
                )

            # 3D joints loss
            if self.is_loss_enabled("joints_3d"):
                losses["loss_j3d"] += (
                    self.index_losss(
                        F.l1_loss,
                        pred_j3d * self.scale,
                        gt_j3d * self.scale,
                        indices,
                    )
                    * self.loss_j3d_ratio
                )

            # 3D joints relative loss (wrist-relative)
            if self.is_loss_enabled("joints_3d_relative"):
                # Get wrist position and compute relative positions
                wrist_index = COMMON15_WRIST_INDEX if is_mocap16 else WRIST_INDEX
                pred_wrist = pred_j3d[:, wrist_index : wrist_index + 1, :]
                gt_wrist = gt_j3d[:, wrist_index : wrist_index + 1, :]

                pred_j3d_relative = pred_j3d - pred_wrist
                gt_j3d_relative = gt_j3d - gt_wrist

                losses["loss_j3d_relative"] += (
                    self.index_losss(
                        F.l1_loss,
                        pred_j3d_relative * self.scale,
                        gt_j3d_relative * self.scale,
                        indices,
                    )
                    * self.loss_j3d_relative_ratio
                )

            # 2D projection loss
            if (
                (not is_mocap16)
                and
                self.is_loss_enabled("joints_2d")
                and "joints_2d" in gt_joints[hand_type]
            ):
                # Project 3D joints to 2D
                batch_size = outs[hand_type]["j3d"].shape[0]
                focal_length_tensor = torch.tensor(
                    [[self.focal_length, self.focal_length]], device=self.device
                ).repeat(batch_size, 1)

                # Get translation from model output
                translation = outs[hand_type]["transl"]  # [B, 3]

                # Project joints to 2D
                pred_j2d = self.perspective_projection(
                    points=outs[hand_type]["j3d"],  # [B, N, 3]
                    translation=translation,  # [B, 3]
                    focal_length=focal_length_tensor,  # [B, 2]
                )

                # Get ground truth 2D joints (in original image coordinates)
                gt_j2d_original = gt_joints[hand_type]["joints_2d"]  # [B, N, 2]

                # Get bbox information from model output
                bbox_key = f"{hand_type}_bboxes"
                if bbox_key in outs:
                    bboxes = outs[bbox_key]  # [B, 4]
                    # Transform GT joints to crop coordinate system
                    gt_j2d = self.transform_original_to_crop(
                        gt_j2d_original, bboxes, self.crop_size
                    )
                else:
                    # Fallback: use original coordinates (for backward compatibility)
                    gt_j2d = gt_j2d_original

                # Check for NaN or Inf in projection result
                if torch.isnan(pred_j2d).any() or torch.isinf(pred_j2d).any():
                    # Skip this loss if projection failed
                    print(
                        f"Warning: Invalid projection for {hand_type} hand. "
                        f"NaN: {torch.isnan(pred_j2d).any()}, Inf: {torch.isinf(pred_j2d).any()}"
                    )
                    continue

                j2d_loss = self.index_losss(
                    F.l1_loss,
                    pred_j2d,
                    gt_j2d,
                    indices,
                )

                # Check if loss is valid (not 0, None, NaN, or Inf)
                if j2d_loss != 0:
                    # Convert to tensor if needed for checking
                    if isinstance(j2d_loss, torch.Tensor):
                        loss_tensor = j2d_loss
                    else:
                        loss_tensor = torch.tensor(j2d_loss, device=self.device)

                    if not torch.isnan(loss_tensor) and not torch.isinf(loss_tensor):
                        losses["loss_j2d"] += j2d_loss * self.loss_j2d_ratio

            # Shape (betas) loss
            if (not is_mocap16) and self.is_loss_enabled("shape"):
                losses["loss_shape"] += (
                    self.index_losss(
                        F.mse_loss,
                        outs[hand_type]["betas"],
                        gt_hand_poses[hand_type]["shape"],
                        indices,
                    )
                    * self.loss_shape_ratio
                )

            # Translation loss
            if (not is_mocap16) and self.is_loss_enabled("transl"):
                losses["loss_transl"] += (
                    self.index_losss(
                        F.mse_loss,
                        outs[hand_type]["transl"],
                        gt_hand_poses[hand_type]["trans"],
                        indices,
                    )
                    * self.loss_transl_ratio
                )

            # Fingertip 3D loss (higher weight on fingertip positions)
            if self.is_loss_enabled("fingertip_3d"):
                fingertip_indices = (
                    COMMON15_FINGERTIP_INDICES if is_mocap16 else FINGERTIP_INDICES
                )
                pred_fingertips = pred_j3d[:, fingertip_indices, :]
                gt_fingertips = gt_j3d[:, fingertip_indices, :]

                losses["loss_fingertip_3d"] += (
                    self.index_losss(
                        F.l1_loss,
                        pred_fingertips * self.scale,
                        gt_fingertips * self.scale,
                        indices,
                    )
                    * self.loss_fingertip_3d_ratio
                )

            # Bone length loss (consistency of bone lengths)
            if self.is_loss_enabled("bone_length"):
                if is_mocap16:
                    pred_bone_lengths = self.compute_bone_lengths(
                        pred_j3d, self.common15_bone_connections
                    )
                    gt_bone_lengths = self.compute_bone_lengths(
                        gt_j3d, self.common15_bone_connections
                    )
                else:
                    pred_bone_lengths = self.compute_bone_lengths(pred_j3d)
                    gt_bone_lengths = self.compute_bone_lengths(gt_j3d)

                losses["loss_bone_length"] += (
                    self.index_losss(
                        F.l1_loss,
                        pred_bone_lengths * self.scale,
                        gt_bone_lengths * self.scale,
                        indices,
                    )
                    * self.loss_bone_length_ratio
                )

        return losses
