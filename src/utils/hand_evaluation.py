import torch
import numpy as np


"""
# MPJPE ↓
Mean Per Joint Position Error (MPJPE) は関節点の推定座標と正解座標の距離（単位は主にmm）を全ての関節点およびデータで平均する
2D / 3D

# PA-MPJPE ↓
PA-MPJPE は手首を起点とした手のポーズでプロクラステス アライメントを実行した後、平均 3D 関節距離を計算します。

# 3DPCK ↑
Percentage of Correct 3D Keypoints (3D PCK)
関節点の推定座標と正解座標の距離が設定した閾値よりも小さいときにその関節点の推定を正しいものとし、推定が正しく行われた割合をその評価値とします。

https://engineering.dena.com/blog/2019/12/cv-papers-19-3d-human-pose-estimation/

# PCKh@0.5 ↑
PCKh@0.5 は，推定関節位置と正解関節位置の距離が対象の頭部の 50%の大きさの閾値以下に存在する割合である．

# AUC
横軸Error Threshold(0mm ~ 100mm), 縦軸PCK, AUC(Area under the curve)は曲線下面積 AUCが1に近いほど性能が良い
縦軸relative 3D PCK, relative root PCKスコアを用いる
"""

# unityの単位を調べてmm二直す
# manopthの返り値はmmになっている


def calc_MPJPE3d(pred, target, device, mode="relative", pred_key="hand_landmarks", coordinate="world"):
    # https://github.com/zhaoweixi/GraFormer/blob/main/common/loss.py
    left_pred = pred["left"][pred_key].to(device)
    right_pred = pred["right"][pred_key].to(device)


    if coordinate == "camera":
        left_target = target["left"]["joints_3d"].to(device)
        right_target = target["right"]["joints_3d"].to(device)
    else:
        left_target = target["left"]["hand_landmarks"].to(device)
        right_target = target["right"]["hand_landmarks"].to(device)

    if mode == "relative":
        left_pred = left_pred - left_pred[:, 5:6, :]
        right_pred = right_pred - right_pred[:, 5:6, :]
        left_target = left_target - left_target[:, 5:6, :]
        right_target = right_target - right_target[:, 5:6, :]
    elif mode == "manopth_relative":
        left_pred = left_pred - left_pred[:, :1, :]
        right_pred = right_pred - right_pred[:, :1, :]
        left_target = left_target - left_target[:, :1, :]
        right_target = right_target - right_target[:, :1, :]

    mpjpe_left = torch.norm(left_pred - left_target, dim=len(left_target.shape) - 1)
    mpjpe_right = torch.norm(
        right_pred - right_target, dim=len(right_target.shape) - 1
    )

    mean_mpjpe = (mpjpe_left + mpjpe_right) / 2

    mean_mpjpe = torch.mean(mean_mpjpe)

    return mean_mpjpe

def calc_MPVPE3d(pred, target, device, mode="relative", pred_key="vertices", coordinate="world"):
    left_pred = pred["left"][pred_key].to(device)
    right_pred = pred["right"][pred_key].to(device)


    if coordinate == "camera":
        left_target = target["left"]["vertices_3d"].to(device)
        right_target = target["right"]["vertices_3d"].to(device)
        left_target_j3d = target["left"]["joints_3d"].to(device)
        right_target_j3d = target["right"]["joints_3d"].to(device)
        left_pred_j3d = pred["left"]["j3d"].to(device)
        right_pred_j3d = pred["right"]["j3d"].to(device)
    else:
        left_target = target["left"]["vertices"].to(device)
        right_target = target["right"]["vertices"].to(device)
        left_target_j3d = target["left"]["hand_landmarks"].to(device)
        right_target_j3d = target["right"]["hand_landmarks"].to(device)
        left_pred_j3d = pred["left"]["j3d"].to(device)
        right_pred_j3d = pred["right"]["j3d"].to(device)

    if mode == "relative":
        left_pred = left_pred - left_pred_j3d[:, 5:6, :]
        right_pred = right_pred - right_pred_j3d[:, 5:6, :]
        left_target = left_target - left_target_j3d[:, 5:6, :]
        right_target = right_target - right_target_j3d[:, 5:6, :]
    elif mode == "manopth_relative":
        left_pred = left_pred - left_pred_j3d[:, :1, :]
        right_pred = right_pred - right_pred_j3d[:, :1, :]
        left_target = left_target - left_target_j3d [:, :1, :]
        right_target = right_target - right_target_j3d[:, :1, :]

    mpvpe_left = torch.norm(left_pred - left_target, dim=len(left_target.shape) - 1)
    mpvpe_right = torch.norm(right_pred - right_target, dim=len(right_target.shape) - 1)

    mean_mpvpe = (mpvpe_left + mpvpe_right) / 2

    mean_mpvpe = torch.mean(mean_mpvpe)

    return mean_mpvpe

def pa_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    Referece: https://github.com/zhaoweixi/GraFormer/blob/main/common/loss.py
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)


def calc_PA_MPJPE(pred, target, device, mode="relative", pred_key="hand_landmarks", coordinate="world"):
    left_pred = pred["left"][pred_key].cpu().numpy()
    right_pred = pred["right"][pred_key].cpu().numpy()

    if coordinate == "camera":
        left_target = target["left"]["joints_3d"].cpu().numpy()
        right_target = target["right"]["joints_3d"].cpu().numpy()
    else:
        left_target = target["left"]["hand_landmarks"].cpu().numpy()
        right_target = target["right"]["hand_landmarks"].cpu().numpy()

    if mode == "relative":
        left_pred = left_pred - left_pred[:, 5:6, :]
        right_pred = right_pred - right_pred[:, 5:6, :]
        left_target = left_target - left_target[:, 5:6, :]
        right_target = right_target - right_target[:, 5:6, :]
    elif mode == "manopth_relative":
        left_pred = left_pred - left_pred[:, :1, :]
        right_pred = right_pred - right_pred[:, :1, :]
        left_target = left_target - left_target[:, :1, :]
        right_target = right_target - right_target[:, :1, :]

    pa_mpjpe_left = pa_mpjpe(left_pred, left_target)
    pa_mpjpe_right = pa_mpjpe(right_pred, right_target)

    mean_pa_mpjpe = (pa_mpjpe_left + pa_mpjpe_right) / 2

    mean_pa_mpjpe = np.mean(mean_pa_mpjpe)

    return mean_pa_mpjpe
