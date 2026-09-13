"""
Reference:
https://github.com/Chris10M/Ev2Hands/blob/main/src/Ev2Hands/model/TEHNet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append("../../hot3d/hot3d")
sys.path.append("..")

from models.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation,
)
from hot3d.hot3d.data_loaders.mano_layer import MANOHandModel
from hot3d.hot3d.data_loaders.pytorch3d_rotation.rotation_conversions import (
    quaternion_to_axis_angle,
)


os.environ["ERPC"] = "1"


class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()

    def forward(self, key, value, query):
        query = query.permute(0, 2, 1)
        N, KC = key.shape[:2]
        key = key.view(N, KC, -1)

        N, KC = value.shape[:2]
        value = value.view(N, KC, -1)

        sim_map = torch.bmm(key, mat2=query)
        sim_map = (KC**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=1)

        context = torch.bmm(sim_map, value)

        return context


class MANORegressor(nn.Module):
    def __init__(
        self,
        n_inp_features=256,
        n_pose_params=6,  # 15 is hot3d, 6 is Ev2Hands pretrained model
        n_shape_params=10,
        key="right",
    ):
        super(MANORegressor, self).__init__()

        normal_channel = True

        if normal_channel:
            additional_channel = n_inp_features
        else:
            additional_channel = 0

        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstractionMsg(
            128,
            [0.4, 0.8],
            [64, 128],
            additional_channel,
            [[128, 128, 256], [128, 196, 256]],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,
            mlp=[256, 512],
            group_all=True,
        )

        self.n_pose_params = n_pose_params
        self.n_mano_params = n_pose_params + n_shape_params
        self.n_global_orient_params = 3
        self.n_transl_params = 3

        self.mano_regressor = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(
                1024,
                self.n_global_orient_params + self.n_mano_params + self.n_transl_params,
            ),
        )

        self.key = key

    def J3dtoJ2d(self, j3d, scale):
        B, N = j3d.shape[:2]
        device = j3d.device

        j2d = torch.zeros(B, N, 2, device=device)
        j2d[:, :, 0] = scale[:, :, 0] * j3d[:, :, 0]
        j2d[:, :, 1] = scale[:, :, 1] * j3d[:, :, 1]

        return j2d

    def forward(self, xyz, features, mano_hand_model: MANOHandModel):
        batch_size = xyz.shape[0]

        l0_xyz = xyz
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l2_xyz = l2_xyz.squeeze(-1)
        l2_points = l2_points.squeeze(-1)

        mano_params = self.mano_regressor(l2_points)

        global_orient = mano_params[:, : self.n_global_orient_params]
        hand_pose = mano_params[
            :,
            self.n_global_orient_params : self.n_global_orient_params
            + self.n_pose_params,
        ]
        betas = mano_params[
            :, self.n_global_orient_params + self.n_pose_params : -self.n_transl_params
        ]
        transl = mano_params[:, -self.n_transl_params :]

        mano_args = {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": betas,
            "transl": transl,
        }

        mano_outs = dict()

        cpu = torch.device("cpu")
        global_xfrom = torch.cat(
            [
                global_orient,
                transl,
            ],
            dim=1,
        ).to(cpu)
        shape_params = betas[0].to(cpu)
        joint_angles = hand_pose.to(cpu)
        is_right_hand = torch.tensor([self.key == "right"] * batch_size).to(cpu)

        vertices, hand_landmarks = mano_hand_model.forward_kinematics(
            shape_params=shape_params,
            joint_angles=joint_angles,
            global_xfrom=global_xfrom,
            is_right_hand=is_right_hand,
        )
        mano_outs["global_orient"] = global_orient
        mano_outs["vertices"] = vertices.to(device=xyz.device)
        mano_outs["j3d"] = hand_landmarks.to(device=xyz.device)

        mano_outs.update(mano_args)

        if self.key == "right":
            hand_triangles = mano_hand_model.mano_layer_right.faces
        else:
            hand_triangles = mano_hand_model.mano_layer_left.faces

        mano_outs["faces"] = torch.from_numpy(hand_triangles).to(
            dtype=torch.int32, device=xyz.device
        )

        return mano_outs


class QueryMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_features),
        )

    def forward(self, x):
        return self.net(x)


class MANOPramsMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_features),
        )

    def forward(self, x):
        return self.net(x)

class MANODecoder(nn.Module):
    def __init__(
        self,
        n_pose_params=15,
        n_shape_params=10,
        n_transl_params=3,
        n_quat_params=4,
        key="right",
    ):
        super().__init__()
        self.key = key
        self.n_pose_params = n_pose_params
        self.n_shape_params = n_shape_params
        self.n_transl_params = n_transl_params
        self.n_quat_params = n_quat_params
        self.n_global_orient_params = 3

        self.sa1 = PointNetSetAbstractionMsg(
            128,
            [0.4, 0.8],
            [64, 128],
            256,
            [[128, 128, 256], [128, 196, 256]],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,
            mlp=[256, 512],
            group_all=True,
        )

        # self.mano_decoder = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(0.3),
        #     nn.Linear(
        #         1024,
        #         self.n_global_orient_params + self.n_pose_params + self.n_shape_params + self.n_transl_params,
        #     ),
        # )

        self.pose_decoder = MANOPramsMLP(512, n_pose_params)
        self.shape_decoder = MANOPramsMLP(512, n_shape_params)
        self.quat_decoder = MANOPramsMLP(512, n_quat_params)
        self.transl_decoder = MANOPramsMLP(512, n_transl_params)

    def format_output(
        self,
        pose,
        shape,
        quat,
        transl,
        batch_size,
        device,
        mano_hand_model: MANOHandModel,
    ):
        mano_args = {
            "quat": quat,
            "hand_pose": pose,
            "betas": shape,
            "transl": transl,
        }

        mano_outs = dict()

        cpu = torch.device("cpu")
        global_xfrom = torch.cat(
            [
                quaternion_to_axis_angle(quaternions=quat),
                transl,
            ],
            dim=1,
        ).to(cpu)
        shape_params = shape[0].to(cpu)
        joint_angles = pose.to(cpu)
        is_right_hand = torch.tensor([self.key == "right"] * batch_size).to(cpu)

        vertices, hand_landmarks = mano_hand_model.forward_kinematics(
            shape_params=shape_params,
            joint_angles=joint_angles,
            global_xfrom=global_xfrom,
            is_right_hand=is_right_hand,
        )
        mano_outs["global_orient"] = quaternion_to_axis_angle(quat)
        mano_outs["vertices"] = vertices.to(device=device)
        mano_outs["j3d"] = hand_landmarks.to(device=device)

        mano_outs.update(mano_args)

        if self.key == "right":
            hand_triangles = mano_hand_model.mano_layer_right.faces
        else:
            hand_triangles = mano_hand_model.mano_layer_left.faces

        mano_outs["faces"] = torch.from_numpy(hand_triangles).to(
            dtype=torch.int32, device=device
        )

        return mano_outs

    def format_output_global_orient(self, pose, shape, global_orient, transl, batch_size, device, mano_hand_model: MANOHandModel):
        mano_args = {
            "global_orient": global_orient,
            "hand_pose": pose,
            "betas": shape,
            "transl": transl,
        }

        mano_outs = dict()

        cpu = torch.device("cpu")
        global_xfrom = torch.cat(
            [
                global_orient,
                transl,
            ],
            dim=1,
        ).to(cpu)
        shape_params = shape[0].to(cpu)
        joint_angles = pose.to(cpu)
        is_right_hand = torch.tensor([self.key == "right"] * batch_size).to(cpu)

        vertices, hand_landmarks = mano_hand_model.forward_kinematics(
            shape_params=shape_params,
            joint_angles=joint_angles,
            global_xfrom=global_xfrom,
            is_right_hand=is_right_hand,
        )
        mano_outs["global_orient"] = global_orient
        mano_outs["vertices"] = vertices.to(device=device)
        mano_outs["j3d"] = hand_landmarks.to(device=device)

        mano_outs.update(mano_args)

        if self.key == "right":
            hand_triangles = mano_hand_model.mano_layer_right.faces
        else:
            hand_triangles = mano_hand_model.mano_layer_left.faces

        mano_outs["faces"] = torch.from_numpy(hand_triangles).to(
            dtype=torch.int32, device=device
        )

        return mano_outs

    def forward(self, xyz, features, mano_hand_model: MANOHandModel):
        batch_size = xyz.shape[0]

        l0_xyz = xyz
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l2_xyz = l2_xyz.squeeze(-1)
        l2_points = l2_points.squeeze(-1)

        x = l2_points

        pose = self.pose_decoder(x)
        shape = self.shape_decoder(x)
        quat = self.quat_decoder(x)
        transl = self.transl_decoder(x)

        mano_outs = self.format_output(
            pose, shape, quat, transl, batch_size, x.device, mano_hand_model
        )

        return mano_outs


class EventEgoHandsV1(nn.Module):
    def __init__(self, n_pose_params=15, num_classes=4, mano_hand_model_path=None, mode="cross"):
        super(EventEgoHandsV1, self).__init__()

        normal_channel = True

        if normal_channel:
            additional_channel = 1 + int(os.getenv("ERPC", 0))
        else:
            additional_channel = 0

        self.mode = mode

        self.sa1 = PointNetSetAbstractionMsg(
            512,
            [0.1, 0.2, 0.4],
            [32, 64, 128],
            3 + additional_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128,
            [0.4, 0.8],
            [64, 128],
            128 + 128 + 64,
            [[128, 128, 256], [128, 196, 256]],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 256])

        self.left_query_conv = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 3 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, 3, 1, 3 // 2),
            nn.BatchNorm1d(256),
        )

        self.right_query_conv = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 3 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, 3, 1, 3 // 2),
            nn.BatchNorm1d(256),
        )
        # self.left_query_conv = QueryMLP(256, 256)
        # self.right_query_conv = QueryMLP(256, 256)

        # self.attention_block = AttentionBlock()
        self.left_attention = AttentionBlock()
        self.right_attention = AttentionBlock()
        # self.left_attention = nn.MultiheadAttention(embed_dim=256, num_heads=2)
        # self.right_attention = nn.MultiheadAttention(embed_dim=256, num_heads=2)

        self.mano_hand_model = MANOHandModel(mano_model_files_dir=mano_hand_model_path)
        # self.left_decoder = MANODecoder(key="left")
        # self.right_decoder = MANODecoder(key="right")
        self.left_mano_regressor = MANORegressor(
            n_pose_params=n_pose_params, key="left"
        )
        self.right_mano_regressor = MANORegressor(
            n_pose_params=n_pose_params, key="right"
        )

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        # PointNet++ Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # PointNet++ Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # point_features: [Batch Size, feature size, number of points]
        point_features = l0_points

        # query: [Batch Size, feature size, number of points]
        left_query = self.left_query_conv(point_features)
        left_query = left_query.permute(2, 0, 1)
        right_query = self.right_query_conv(point_features)
        right_query = right_query.permute(2, 0, 1)
        # features: [number of points, Batch Size, feature size]
        if self.mode == "cross":
            left_hand_features = self.left_attention(left_query, right_query, right_query)
            right_hand_features = self.right_attention(right_query, left_query, left_query)
        else:
            left_hand_features = self.left_attention(left_query, left_query, left_query)
            right_hand_features = self.right_attention(right_query, right_query, right_query)

        left_hand_features = left_hand_features.permute(1, 2, 0)
        right_hand_features = right_hand_features.permute(1, 2, 0)

        # left = self.left_decoder(l0_xyz, left_hand_features, self.mano_hand_model)
        # right = self.right_decoder(l0_xyz, right_hand_features, self.mano_hand_model)
        left = self.left_mano_regressor(l0_xyz, left_hand_features, self.mano_hand_model)
        right = self.right_mano_regressor(l0_xyz, right_hand_features, self.mano_hand_model)

        return {"left": left, "right": right}


if __name__ == "__main__":
    model = EventEgoHandsV1(
        mano_hand_model_path="src/mano/models",
        n_pose_params=15,
    )
    # parameter count
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.rand(16, 5, 2048)
    output = model(x)

    # 計算グラフの可視化
    # left = output["left"]
    # betas = left["vertices"]
    # dot = make_dot(betas, params=dict(model.named_parameters()))
    # dot.render("model_graph", format="png", cleanup=True)
