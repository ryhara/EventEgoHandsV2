import collections
import torch
import torch.nn.functional as F
from torch import nn
import sys
import numpy as np
import numpy.typing as npt
from typing import Sequence

# from mesh_intersection.loss import DistanceFieldPenetrationLoss
# from mesh_intersection.bvh_search_tree import BVH

from utils.config import MANO_CMPS, OUTPUT_WIDTH, OUTPUT_HEIGHT
from utils.projection import PROJECTION_MATRIX, opengl_projection_transform

sys.path.append("../../hot3d/hot3d")
from hot3d.hot3d.data_loaders.mano_layer import MANOHandModel


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def rotmat_to_rot6d(x):
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d


# class CollisionLoss:
#     def __init__(self, device) -> None:
#         max_collisions = 16
#         self.search_tree = BVH(max_collisions=max_collisions)

#         self.collision_weight = 1e2
#         print(f"collision_weight: {self.collision_weight}")

#         sigma = 0.5
#         point2plane = False
#         self.pen_distance = DistanceFieldPenetrationLoss(
#             sigma=sigma,
#             point2plane=point2plane,
#             vectorized=True,
#             penalize_outside=False,
#         )

#         self.device = device

#     def __call__(self, outs):
#         B, V, C = outs["left"]["vertices"].shape

#         verts = list()
#         faces = list()
#         for i in range(B):
#             hand_verts = torch.cat(
#                 [outs["left"]["vertices"][i], outs["right"]["vertices"][i]]
#             )
#             hand_faces = torch.cat([outs["left"]["faces"], outs["right"]["faces"] + V])

#             verts.append(hand_verts[None, ...])
#             faces.append(hand_faces[None, ...])

#         verts_tensor = torch.cat(verts, dim=0)
#         face_tensor = torch.cat(faces, dim=0)

#         triangles = verts_tensor.view([-1, 3])[face_tensor]

#         with torch.no_grad():
#             collision_idxs = self.search_tree(triangles)

#         loss = self.pen_distance(triangles, collision_idxs)
#         indices = torch.nonzero(loss)

#         if len(indices):
#             loss = loss[indices].mean() * self.collision_weight
#         else:
#             loss = 0

#         return loss


def undistort(params: Sequence[float], p: npt.NDArray) -> npt.NDArray:
    k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = params  # pylint: disable=unpacking-non-sequence
    # radial component
    r2 = (p * p).sum(axis=-1, keepdims=True)
    r2 = np.clip(r2, -(np.pi**2), np.pi**2)

    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r4 * r4
    r10 = r4 * r6
    r12 = r6 * r6
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
    uv = p * radial

    # tangential component
    x, y = uv[..., 0], uv[..., 1]
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    x += 2 * p2 * xy + p1 * (r2 + 2 * x2)
    y += 2 * p1 * xy + p2 * (r2 + 2 * y2)

    # thin prism
    r4 = r2 * r2
    x += s1 * r2 + s2 * r4
    y += s3 * r2 + s4 * r4

    return np.stack((x, y), axis=-1)


class BaseLoss(nn.Module):
    def __init__(self, device, mano_hand_model_path=None) -> None:
        super(BaseLoss, self).__init__()
        self.device = device
        # self.collision_loss = CollisionLoss(device)

        self.previous_out = dict()
        self.PROJECTION_MATRIX = torch.tensor(PROJECTION_MATRIX).to(device).float()
        if mano_hand_model_path:
            self.mano_hand_model = MANOHandModel(mano_hand_model_path)

    def quant_error_loss(self, inp, target):
        c1 = target[:, 0]
        v1 = target[:, 1:]
        c2 = inp[:, 0]
        v2 = inp[:, 1:]

        tmp = torch.clamp(c1 * c2 + torch.sum(v1 * v2, 1), -1, 1)
        cosine_similarity = torch.abs(tmp)

        cosine_distance = 1 - cosine_similarity

        return cosine_distance.mean(-1)

    def index_losss(self, loss_fn, outs, targets, indices):
        indices = indices.int()

        if indices.sum() == 0:
            return 0

        loss = loss_fn(outs, targets, reduction="none")
        B = loss.shape[0]
        loss = loss.reshape(B, -1)
        D = loss.shape[1]
        indices = indices[:, None].repeat(1, D)

        index_loss = loss * indices
        index_mean_loss = index_loss.sum() / indices.sum()

        return index_mean_loss

    def forward(self, outs, targets, losses=None):
        # targets["mano_gt"] = torch.mean(targets["mano_gt"])

        # if targets["mano_gt"]:
        return self.forward_mano_data(outs, targets, losses)

        # return self.forward_non_mano_data(outs, targets["gt_hand_poses"], losses)

    def forward_mano_data(self, outs, targets, losses=None):
        if losses is None:
            losses = collections.defaultdict(float)
        gt_hand_poses = targets["gt_hand_poses"]
        gt_joints = targets["gt_joints"]
        # gt_device_pose = targets["gt_device_pose"]

        # world_camera_quat = gt_device_pose["world_camera_pose"]["quat"].cpu().numpy()
        # world_camera_transl = (
        #     gt_device_pose["world_camera_pose"]["translation"].cpu().numpy()
        # )

        # intrinsics
        # projection_params = (
        #     gt_device_pose["intrinsics"]["projection_params"][0].cpu().numpy()
        # )
        # projection_params = projection_params[3:]
        # c = tuple([512 / 2, 512 / 2])
        # f = (150, 150)
        # height_scale = OUTPUT_HEIGHT / 512
        # width_scale = OUTPUT_WIDTH / 512

        # device_pose = SE3.from_quat_and_translation(
        #     world_camera_quat[:, 0],
        #     np.array(world_camera_quat[:, 1:]),
        #     np.array(world_camera_transl),
        # )[0]

        for hand_type in ["left", "right"]:
            # gt from mano
            # cpu = torch.device("cpu")
            # transl = gt_hand_poses[hand_type]["trans"].cpu().numpy()
            # quat = gt_hand_poses[hand_type]["quat"].cpu().numpy()
            # world_hand_pose = SE3.from_quat_and_translation(
            # quat[:, 0],
            # quat[:, 1:],
            # transl
            # )
            # camera_hand_pose = device_pose.inverse() @ world_hand_pose
            # quat_and_translation = camera_hand_pose.to_quat_and_translation()
            # camera_quat = torch.tensor(quat_and_translation[:, :4], dtype=torch.float32).to(cpu)
            # camera_trans = torch.tensor(quat_and_translation[:, 4:], dtype=torch.float32).to(cpu)
            # global_orient = quaternion_to_axis_angle(camera_quat).to(cpu)

            # global_xfrom = torch.cat(
            #     [
            #         global_orient,
            #         camera_trans,
            #     ],
            #     dim=1,
            # ).to(cpu)
            # shape_params = gt_hand_poses[hand_type]["shape"][0].to(cpu)
            # joint_angles = gt_hand_poses[hand_type]["hand_pose"].to(cpu)
            # batch_size = joint_angles.shape[0]
            # is_right_hand = torch.tensor([hand_type == "right"] * batch_size).to(cpu)

            # vertices, hand_landmarks = self.mano_hand_model.forward_kinematics(
            #     shape_params=shape_params,
            #     joint_angles=joint_angles,
            #     global_xfrom=global_xfrom,
            #     is_right_hand=is_right_hand,
            # )

            # gt_hand_poses[hand_type]["global_orient"] = global_orient.to(device=self.device, dtype=torch.float32)
            # gt_hand_poses[hand_type]["trans"] = camera_trans.to(device=self.device, dtype=torch.float32)
            # gt_hand_poses[hand_type]["j3d"] = hand_landmarks.to(device=self.device, dtype=torch.float32)
            # gt_hand_poses[hand_type]["vertices"] = vertices.to(device=self.device, dtype=torch.float32)

            # gt
            # gt_hand_poses[hand_type]["j3d"] = gt_hand_poses[hand_type]["hand_landmarks"]
            # gt_hand_poses[hand_type]["faces"] = gt_hand_poses[hand_type]["triangles"]
            gt_hand_poses[hand_type]["vertices"] = gt_joints[hand_type]["vertices_3d"]
            gt_hand_poses[hand_type]["j3d"] = gt_joints[hand_type]["joints_3d"]
            # gt_hand_poses[hand_type]["j2d"] = gt_joints[hand_type]["joints_2d"].to(
            #     device=self.device
            # )
            # gt_hand_poses[hand_type]["j2d"][:, :, 0] *= width_scale
            # gt_hand_poses[hand_type]["j2d"][:, :, 1] *= height_scale
            # outs
            # hand_landmarks_2d = np.array(
            #     (
            #         (outs[hand_type]["j3d"][:, :, 0] / outs[hand_type]["j3d"][:, :, 2])
            #         .detach()
            #         .cpu()
            #         .numpy(),
            #         (outs[hand_type]["j3d"][:, :, 1] / outs[hand_type]["j3d"][:, :, 2])
            #         .detach()
            #         .cpu()
            #         .numpy(),
            #     )
            # ).transpose(1, 2, 0)
            # q = undistort(projection_params, hand_landmarks_2d)
            # result = q * f + c
            # mask = (result[:, :, 0] >= 0) & (result[:, :, 0] < 512) & (result[:, :, 1] >= 0) & (result[:, :, 1] < 512)
            # result = np.where(mask[:, :, None], result, np.nan)

            # result[:, :, 0] *= width_scale
            # result[:, :, 1] *= height_scale

            # outs[hand_type]["j2d"] = torch.tensor(result, dtype=torch.float32).to(
            #     device=self.device
            # )

        # losses["loss_interpen"] += self.collision_loss(outs)

        interacting_indices = torch.sum(gt_hand_poses["handedness"], 1) == 2

        # losses["loss_inter_shape"] += self.index_losss(
        #     F.mse_loss,
        #     outs["left"]["betas"],
        #     outs["right"]["betas"],
        #     interacting_indices,
        # )

        # losses["loss_inter_global_orient"] += (
        #     self.index_losss(
        #         F.mse_loss,
        #         outs["left"]["global_orient"] - outs["right"]["global_orient"],
        #         (gt_hand_poses["left"]["global_orient"] - gt_hand_poses["right"]["global_orient"]),
        #         interacting_indices,
        #     )
        # )

        # losses["loss_inter_transl"] += (
        #     self.index_losss(
        #         F.mse_loss,
        #         outs["left"]["transl"] - outs["right"]["transl"],
        #         (gt_hand_poses["left"]["trans"] - gt_hand_poses["right"]["trans"]),
        #         interacting_indices,
        #     )
        #     # * 100
        # )

        losses["loss_inter_j3d"] += (
            self.index_losss(
                F.mse_loss,
                (outs["left"]["j3d"] - outs["right"]["j3d"]),
                (gt_hand_poses["left"]["j3d"] - gt_hand_poses["right"]["j3d"]),
                interacting_indices,
            )
        )

        for hand_type in ["left", "right"]:
            indices = gt_hand_poses[hand_type]["valid"]
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(indices).to(self.device)

            losses["loss_vertices"] += (
                self.index_losss(
                F.l1_loss,
                outs[hand_type]["vertices"],
                gt_hand_poses[hand_type]["vertices"],
                indices,
                )
            )

            # losses["loss_global_orient"] += (
            #     self.index_losss(
            #         F.mse_loss,
            #         outs[hand_type]["global_orient"],
            #         gt_hand_poses[hand_type]["global_orient"],
            #         indices,
            #     )
            #     * 10
            # )

            gt_hand_poses[hand_type]["hand_pose"] = gt_hand_poses[hand_type][
                "hand_pose"
            ][:, :MANO_CMPS]
            losses["loss_hand_pose"] += (
                self.index_losss(
                    F.mse_loss,
                    outs[hand_type]["hand_pose"],
                    gt_hand_poses[hand_type]["hand_pose"],
                    indices,
                )
                * 20
            )

            # Ev2Hands
            # losses["loss_rj3d"] += (
            #     self.index_losss(
            #         F.l1_loss,
            #         (
            #             outs[hand_type]["j3d"][:, 1:, :]
            #             - outs[hand_type]["j3d"][:, :1, :]
            #         )
            #         * 1000,
            #         (
            #             gt_hand_poses[hand_type]["j3d"][:, 1:, :]
            #             - gt_hand_poses[hand_type]["j3d"][:, :1, :]
            #         )
            #         * 1000,
            #         indices,
            #     )
            #     * 0.01
            # )

            # losses["loss_rj3d"] += (
            #     self.index_losss(
            #         F.mse_loss,
            #         (
            #             outs[hand_type]["j3d"][:, :, :]
            #             - outs[hand_type]["j3d"][:, 5:6, :]
            #         )
            #         * 1000,
            #         (
            #             gt_hand_poses[hand_type]["j3d"][:, :, :]
            #             - gt_hand_poses[hand_type]["j3d"][:, 5:6, :]
            #         )
            #         * 1000,
            #         indices,
            #     )
            #     * 0.01
            # )

            losses["loss_j3d"] += (
                self.index_losss(
                    F.l1_loss,
                    outs[hand_type]["j3d"] * 1000,
                    gt_hand_poses[hand_type]["j3d"] * 1000,
                    indices,
                )
                * 0.1
            )

            # j2d includes nan values
            # losses["loss_j2d"] += (
            #     self.index_losss(
            #         F.mse_loss,
            #         torch.nan_to_num(outs[hand_type]["j2d"], nan=0.0),
            #         torch.nan_to_num(gt_hand_poses[hand_type]["j2d"], nan=0.0),
            #         indices,
            #     )
            #     * 0.0001
            # )

            losses["loss_shape"] += (
                self.index_losss(
                    F.mse_loss,
                    outs[hand_type]["betas"],
                    gt_hand_poses[hand_type]["shape"],
                    indices,
                )
                * 20
            )
            # losses["loss_transl"] += (
            #     self.index_losss(
            #         F.mse_loss,
            #         outs[hand_type]["transl"],
            #         gt_hand_poses[hand_type]["trans"],
            #         indices,
            #     )
            #     * 10
            # )

            # === L1 Loss ===
            # lambda_l1 = 0.01
            # l1_reg = torch.sum(torch.abs(outs[hand_type]["hand_pose"]))
            # losses["loss_hand_pose"] += lambda_l1 * l1_reg

            # === L2 Loss ===
            # lambda_l2 = 0.01
            # l2_reg = torch.sum(torch.square(outs[hand_type]["hand_pose"]))
            # losses["loss_hand_pose"] += lambda_l2 * l2_reg

            # losses["regularizer_loss"] += 0.1 * self.index_losss(
            #     F.mse_loss, outs[hand_type]["betas"], outs[hand_type]["betas"], indices
            # )
            # losses["regularizer_loss"] += self.index_losss(
            #     F.mse_loss,
            #     outs[hand_type]["hand_pose"],
            #     outs[hand_type]["hand_pose"],
            #     indices,
            # )

        # losses['loss_class_logits'] = F.cross_entropy(outs['class_logits'], targets['class_logits'], weight=torch.tensor([1, 30, 30, 10]).float().to(self.device),
        #                                               ignore_index=0)

        return losses

    def forward_non_mano_data(self, outs, targets, losses=None):
        if losses is None:
            losses = collections.defaultdict(float)

        for hand_type in ["left", "right"]:
            outs[hand_type]["faces"] = torch.tensor(self.hands[hand_type].faces).to(
                self.device
            )
            outs[hand_type]["j2d"] = opengl_projection_transform(
                self.PROJECTION_MATRIX,
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT,
                outs[hand_type]["j3d"] * 1000,
            )

        losses["loss_interpen"] += self.collision_loss(outs)

        interacting_indices = torch.sum(targets["handedness"], 1) == 2

        losses["loss_inter_shape"] += (
            self.index_losss(
                F.mse_loss,
                outs["left"]["betas"],
                outs["right"]["betas"],
                interacting_indices,
            )
            * 1e3
        )

        losses["loss_inter_j3d"] += self.index_losss(
            F.l1_loss,
            (outs["left"]["j3d"] - outs["right"]["j3d"]) * 1000,
            (targets["left"]["j3d"] - targets["right"]["j3d"]) * 1000,
            interacting_indices,
        )

        for hand_type in ["left", "right"]:
            indices = targets[hand_type]["valid"]

            # losses["regularizer_loss"] += (
            #     torch.mean(outs[hand_type]["betas"] ** 2) * 1e3
            # )
            # losses["regularizer_loss"] += torch.mean(outs[hand_type]["hand_pose"] ** 2)

            # losses["regularizer_loss"] *= 0.025

            losses["loss_rj3d"] += (
                self.index_losss(
                    F.l1_loss,
                    (
                        outs[hand_type]["j3d"][:, 1:, :]
                        - outs[hand_type]["j3d"][:, 5:6, :]
                    )
                    * 1000,
                    (
                        targets[hand_type]["j3d"][:, 1:, :]
                        - targets[hand_type]["j3d"][:, 5:6, :]
                    )
                    * 1000,
                    indices,
                )
                * 10
            )
            losses["loss_j2d"] += self.index_losss(
                F.mse_loss,
                outs[hand_type]["j2d"],
                targets[hand_type]["j2d"][..., :2],
                indices,
            )

        return losses
