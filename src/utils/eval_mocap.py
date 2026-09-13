"""
Evaluation functions for Mocap 16-joint annotations.

Mocap (16 joints) and HOT3D (20 joints) share only 15 common joints.
Excluded: HOT3D fingertips (indices 0-4), Mocap Thumb CMC (index 12).

Mocap 16 joint layout:
  15: Wrist
   0-2: Index (MCP, PIP, DIP)
   3-5: Middle (MCP, PIP, DIP)
   6-8: Pinky (MCP, PIP, DIP)
   9-11: Ring (MCP, PIP, DIP)
  12: Thumb CMC (no HOT3D equivalent)
  13-14: Thumb (MCP, IP)

HOT3D 20 joint layout:
   5: Wrist
   0-4: Fingertips (Thumb, Index, Middle, Ring, Pinky)
   6-7: Thumb (MCP, IP)
   8-10: Index (MCP, PIP, DIP)
  11-13: Middle (MCP, PIP, DIP)
  14-16: Ring (MCP, PIP, DIP)
  17-19: Pinky (MCP, PIP, DIP)

Common 15 joint layout (after mapping):
   0-2: Index (MCP, PIP, DIP)
   3-5: Middle (MCP, PIP, DIP)
   6-8: Pinky (MCP, PIP, DIP)
   9-11: Ring (MCP, PIP, DIP)
  12: Thumb MCP
  13: Thumb IP
  14: Wrist
"""

import numpy as np
import sklearn.metrics as skmetrics
import torch

NUM_COMMON_JOINTS = 15

# Indices to select from Mocap 16 joints (exclude index 12: Thumb CMC)
MOCAP_COMMON_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]

# Corresponding indices to select from HOT3D 20 joints
HOT3D_COMMON_INDICES = [8, 9, 10, 11, 12, 13, 17, 18, 19, 14, 15, 16, 6, 7, 5]

# Wrist position in the 15 common joints
COMMON_WRIST_INDEX = 14


def map_pred_to_common(joints_hot3d: torch.Tensor) -> torch.Tensor:
    """Select 15 common joints from HOT3D 20-joint predictions.

    Works with any leading dimensions: (20, 3), (B, 20, 3), (2, 20, 3), etc.
    """
    return joints_hot3d[..., HOT3D_COMMON_INDICES, :]


def map_gt_to_common(joints_mocap: torch.Tensor) -> torch.Tensor:
    """Select 15 common joints from Mocap 16-joint GT.

    Works with any leading dimensions: (16, 3), (B, 16, 3), (2, 16, 3), etc.
    """
    return joints_mocap[..., MOCAP_COMMON_INDICES, :]


# ---------------------------------------------------------------------------
# MPJPE
# ---------------------------------------------------------------------------


def calc_MPJPE3d_mocap(pred, target, device, mode="relative", pred_key="j3d"):
    """MPJPE on 15 common joints between HOT3D prediction and Mocap GT."""
    left_pred = map_pred_to_common(pred["left"][pred_key].to(device))
    right_pred = map_pred_to_common(pred["right"][pred_key].to(device))
    left_target = map_gt_to_common(target["left"]["joints_3d"].to(device))
    right_target = map_gt_to_common(target["right"]["joints_3d"].to(device))

    if mode == "relative":
        w = COMMON_WRIST_INDEX
        left_pred = left_pred - left_pred[:, w : w + 1, :]
        right_pred = right_pred - right_pred[:, w : w + 1, :]
        left_target = left_target - left_target[:, w : w + 1, :]
        right_target = right_target - right_target[:, w : w + 1, :]

    mpjpe_left = torch.norm(left_pred - left_target, dim=-1)
    mpjpe_right = torch.norm(right_pred - right_target, dim=-1)

    return torch.mean((mpjpe_left + mpjpe_right) / 2)


# ---------------------------------------------------------------------------
# PA-MPJPE (Procrustes-aligned)
# ---------------------------------------------------------------------------


def _procrustes_align(predicted, target):
    """Procrustes alignment (rigid + scale). Returns per-joint errors."""
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

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.linalg.norm(predicted_aligned - target, axis=-1)


def calc_PA_MPJPE_mocap(pred, target, device, mode="relative", pred_key="j3d"):
    """PA-MPJPE on 15 common joints between HOT3D prediction and Mocap GT."""
    left_pred = map_pred_to_common(pred["left"][pred_key].to(device)).cpu().numpy()
    right_pred = map_pred_to_common(pred["right"][pred_key].to(device)).cpu().numpy()
    left_target = map_gt_to_common(target["left"]["joints_3d"].to(device)).cpu().numpy()
    right_target = map_gt_to_common(
        target["right"]["joints_3d"].to(device)
    ).cpu().numpy()

    if mode == "relative":
        w = COMMON_WRIST_INDEX
        left_pred = left_pred - left_pred[:, w : w + 1, :]
        right_pred = right_pred - right_pred[:, w : w + 1, :]
        left_target = left_target - left_target[:, w : w + 1, :]
        right_target = right_target - right_target[:, w : w + 1, :]

    pa_left = _procrustes_align(left_pred, left_target)
    pa_right = _procrustes_align(right_pred, right_target)

    return np.mean((pa_left + pa_right) / 2)


# ---------------------------------------------------------------------------
# PCK / AUC helpers
# ---------------------------------------------------------------------------


def _pck_curve(dists, num_steps, dist_max_mm):
    """Compute PCK curve from distance tensor."""
    pck = np.zeros(num_steps + 1)
    for s in range(num_steps + 1):
        dist_s = (dist_max_mm / num_steps) * s
        pck[s] = (dists < dist_s).float().mean()
    return pck


def _apply_valid_mask(dists, valid_mask, num_joints_per_hand):
    """Mask out invalid hand joints. Returns None if all invalid."""
    if valid_mask is None:
        return dists
    mask = torch.cat(
        [
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[0],
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[1],
        ]
    )
    valid_indices = mask > 0.5
    if valid_indices.sum() == 0:
        return None
    return dists[valid_indices]


def get_auc(pck3d):
    """Compute AUC from PCK curve."""
    auc = skmetrics.auc(range(pck3d.shape[0]), pck3d) / pck3d.shape[0]
    return round(auc, 3)


# ---------------------------------------------------------------------------
# PCK frame-level functions (15 common joints)
# ---------------------------------------------------------------------------


def absolute_pck3d_frame_mocap(
    joints_pred, joints_gt, num_steps=100, dist_max_mm=100, valid_mask=None
):
    """Absolute PCK3D for 15 common joints.

    Args:
        joints_pred: (2, 20, 3) HOT3D prediction [left, right]
        joints_gt: (2, 16, 3) Mocap GT [left, right]
    """
    pred = map_pred_to_common(joints_pred)
    gt = map_gt_to_common(joints_gt)

    pred_flat = torch.cat([pred[0], pred[1]], 0)
    gt_flat = torch.cat([gt[0], gt[1]], 0)

    dists = torch.norm(pred_flat - gt_flat, p=2, dim=1)
    dists = _apply_valid_mask(dists, valid_mask, NUM_COMMON_JOINTS)
    if dists is None:
        return np.zeros(num_steps + 1)

    return _pck_curve(dists, num_steps, dist_max_mm)


def relative_pck3d_frame_mocap(
    joints_pred, joints_gt, num_steps=100, dist_max_mm=100, valid_mask=None
):
    """Wrist-relative PCK3D for 15 common joints.

    Args:
        joints_pred: (2, 20, 3) HOT3D prediction [left, right]
        joints_gt: (2, 16, 3) Mocap GT [left, right]
    """
    pred = map_pred_to_common(joints_pred)
    gt = map_gt_to_common(joints_gt)

    w = COMMON_WRIST_INDEX
    pred = pred - pred[:, w : w + 1, :]
    gt = gt - gt[:, w : w + 1, :]

    pred_flat = torch.cat([pred[0], pred[1]], 0)
    gt_flat = torch.cat([gt[0], gt[1]], 0)

    dists = torch.norm(pred_flat - gt_flat, p=2, dim=1)
    dists = _apply_valid_mask(dists, valid_mask, NUM_COMMON_JOINTS)
    if dists is None:
        return np.zeros(num_steps + 1)

    return _pck_curve(dists, num_steps, dist_max_mm)


def right_root_relative_pck3d_frame_mocap(
    joints_pred, joints_gt, num_steps=100, dist_max_mm=100, valid_mask=None
):
    """Right-wrist-relative PCK3D for 15 common joints.

    Args:
        joints_pred: (2, 20, 3) HOT3D prediction [left, right]
        joints_gt: (2, 16, 3) Mocap GT [left, right]
    """
    pred = map_pred_to_common(joints_pred)
    gt = map_gt_to_common(joints_gt)

    w = COMMON_WRIST_INDEX
    pred = pred - pred[1:, w : w + 1, :]
    gt = gt - gt[1:, w : w + 1, :]

    pred_flat = torch.cat([pred[0], pred[1]], 0)
    gt_flat = torch.cat([gt[0], gt[1]], 0)

    dists = torch.norm(pred_flat - gt_flat, p=2, dim=1)
    dists = _apply_valid_mask(dists, valid_mask, NUM_COMMON_JOINTS)
    if dists is None:
        return np.zeros(num_steps + 1)

    return _pck_curve(dists, num_steps, dist_max_mm)


def mepj_frame_mocap(joints_pred, joints_gt, valid_mask=None):
    """Mean End-Point Joint error for 15 common joints (wrist-relative).

    Args:
        joints_pred: (2, 20, 3) HOT3D prediction [left, right]
        joints_gt: (2, 16, 3) Mocap GT [left, right]
    """
    pred = map_pred_to_common(joints_pred)
    gt = map_gt_to_common(joints_gt)

    w = COMMON_WRIST_INDEX
    pred = pred - pred[:, w : w + 1, :]
    gt = gt - gt[:, w : w + 1, :]

    pred_flat = torch.cat([pred[0], pred[1]], 0)
    gt_flat = torch.cat([gt[0], gt[1]], 0)

    dists = torch.norm(pred_flat - gt_flat, p=2, dim=1)
    dists = _apply_valid_mask(dists, valid_mask, NUM_COMMON_JOINTS)
    if dists is None:
        return torch.tensor(0.0)

    return dists.mean()


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------


def evaluate_joints_real_mocap(j3d_pred, j3d_gts, num_steps, valid_mask=None):
    """Evaluate 3D joint predictions against Mocap GT using 15 common joints.

    Args:
        j3d_pred: (2, 20, 3) HOT3D prediction [left, right] in mm
        j3d_gts: (N, 2, 16, 3) Mocap GT candidates in mm
        num_steps: Number of steps for PCK computation
        valid_mask: Optional (2,) tensor [left_valid, right_valid]

    Returns:
        Dictionary with evaluation metrics.
    """
    dist_max_mm = 100

    # Find best GT candidate by right-root-relative AUC
    aucs = []
    for idx, j3d_gt in enumerate(j3d_gts):
        right_root_pck = right_root_relative_pck3d_frame_mocap(
            j3d_pred, j3d_gt, num_steps=num_steps,
            dist_max_mm=dist_max_mm, valid_mask=valid_mask,
        )
        aucs.append(get_auc(right_root_pck))

    best_idx = np.argmax(aucs)
    j3d_gt = j3d_gts[best_idx]

    absolute_pck3d = absolute_pck3d_frame_mocap(
        j3d_pred, j3d_gt, num_steps=num_steps,
        dist_max_mm=dist_max_mm, valid_mask=valid_mask,
    )
    relative_pck3d = relative_pck3d_frame_mocap(
        j3d_pred, j3d_gt, num_steps=num_steps,
        dist_max_mm=dist_max_mm, valid_mask=valid_mask,
    )
    right_root_relative_pck3d = right_root_relative_pck3d_frame_mocap(
        j3d_pred, j3d_gt, num_steps=num_steps,
        dist_max_mm=dist_max_mm, valid_mask=valid_mask,
    )

    joint_loss = mepj_frame_mocap(j3d_pred, j3d_gt, valid_mask=valid_mask).item()

    # Root distance between left and right wrists (on raw mocap joints)
    root_distance = [
        torch.norm((j3d_gt[0, 15] - j3d_gt[1, 15]), p=2)
        .cpu()
        .numpy()
        .tolist()
    ]

    return {
        "root_distance": root_distance,
        "joint_loss": joint_loss,
        "absolute_pck3d": absolute_pck3d,
        "relative_pck3d": relative_pck3d,
        "right_root_relative_pck3d": right_root_relative_pck3d,
    }


def evaluate_hand_mocap(pred, j3d_gts, device, num_steps, use_valid_mask=True):
    """Evaluate hand predictions against Mocap GT (15 common joints).

    Args:
        pred: Model output dict with pred["left"]["j3d"], pred["right"]["j3d"], etc.
        j3d_gts: (B, 2, 16, 3) Mocap GT in metres
        device: torch device
        num_steps: Number of steps for PCK computation
        use_valid_mask: Whether to filter by hand validity

    Returns:
        List of frame dicts with scores.
    """
    frames = []
    batch_size = pred["left"]["j3d"].shape[0]

    left_valid = pred.get("left_valid", torch.ones(batch_size, device=device))
    right_valid = pred.get("right_valid", torch.ones(batch_size, device=device))

    for idx in range(batch_size):
        left_is_valid = left_valid[idx].item() > 0.5 if use_valid_mask else True
        right_is_valid = right_valid[idx].item() > 0.5 if use_valid_mask else True

        if use_valid_mask and not (left_is_valid or right_is_valid):
            continue

        hands = {
            "left": {
                "j3d": pred["left"]["j3d"][idx],
                "valid": left_is_valid,
            },
            "right": {
                "j3d": pred["right"]["j3d"][idx],
                "valid": right_is_valid,
            },
        }

        j3d_pred = torch.cat(
            [hands["left"]["j3d"].unsqueeze(0), hands["right"]["j3d"].unsqueeze(0)],
            dim=0,
        )
        j3d_gt = j3d_gts[idx].unsqueeze(0)

        valid_mask_tensor = None
        if use_valid_mask:
            valid_mask_tensor = torch.tensor(
                [float(left_is_valid), float(right_is_valid)], device=device
            )

        # Convert from metres to millimetres
        scores = evaluate_joints_real_mocap(
            j3d_pred=j3d_pred * 1000,
            j3d_gts=j3d_gt * 1000,
            num_steps=num_steps,
            valid_mask=valid_mask_tensor,
        )
        hands["scores"] = scores
        frames.append(hands)

    return frames
