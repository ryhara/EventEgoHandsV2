import torch
import numpy as np
from sklearn import metrics as skmetrics


def absolute_pck3d_frame(joints_pred, joints_gt, num_steps=100, dist_max_mm=100, valid_mask=None):
    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0)
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0)

    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1)

    # Apply valid mask if provided
    if valid_mask is not None:
        # valid_mask: [2] for [left, right]
        # Expand to match joints: each hand has 21 joints
        num_joints_per_hand = joints_pred.shape[0] // 2
        mask = torch.cat([
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[0],
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[1]
        ])
        # Only compute for valid joints
        valid_indices = mask > 0.5
        if valid_indices.sum() == 0:
            return np.zeros(num_steps + 1)
        dists = dists[valid_indices]

    pck = np.zeros(num_steps + 1)
    for s in range(num_steps + 1):
        dist_s = (dist_max_mm / num_steps) * s
        pck[s] = (dists < dist_s).float().mean()

    return pck


def relative_pck3d_frame(joints_pred, joints_gt, num_steps=100, dist_max_mm=100, mode="default", valid_mask=None):
    # # relative to root joint
    # HOT3D
    if mode == "NHOT3D":
        joints_pred = joints_pred - joints_pred[:, 5:6, :]
        joints_gt = joints_gt - joints_gt[:, 5:6, :]
    # Ev2Hands
    else:
        joints_pred = joints_pred - joints_pred[:, :1, :]
        joints_gt = joints_gt - joints_gt[:, :1, :]

    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0)
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0)

    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1)

    # Apply valid mask if provided
    if valid_mask is not None:
        # valid_mask: [2] for [left, right]
        # Expand to match joints: each hand has 21 joints
        num_joints_per_hand = joints_pred.shape[0] // 2
        mask = torch.cat([
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[0],
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[1]
        ])
        # Only compute for valid joints
        valid_indices = mask > 0.5
        if valid_indices.sum() == 0:
            return np.zeros(num_steps + 1)
        dists = dists[valid_indices]

    pck = np.zeros(num_steps + 1)
    for s in range(num_steps + 1):
        dist_s = (dist_max_mm / num_steps) * s
        pck[s] = (dists < dist_s).float().mean()

    return pck


def right_root_relative_pck3d_frame(
    joints_pred, joints_gt, num_steps=100, dist_max_mm=100, mode="default", valid_mask=None
):
    # HOT3D
    if mode == "NHOT3D":
        joints_pred = joints_pred - joints_pred[1:, 5:6, :]
        joints_gt = joints_gt - joints_gt[1:, 5:6, :]
    # Ev2Hands
    else:
        joints_pred = joints_pred - joints_pred[1:, :1, :]  # 1: - select right hand
        joints_gt = joints_gt - joints_gt[1:, :1, :]  # 1: - select right hand

    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0)
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0)

    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1)

    # Apply valid mask if provided
    if valid_mask is not None:
        # valid_mask: [2] for [left, right]
        # Expand to match joints: each hand has 21 joints
        num_joints_per_hand = joints_pred.shape[0] // 2
        mask = torch.cat([
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[0],
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[1]
        ])
        # Only compute for valid joints
        valid_indices = mask > 0.5
        if valid_indices.sum() == 0:
            return np.zeros(num_steps + 1)
        dists = dists[valid_indices]

    pck = np.zeros(num_steps + 1)
    for s in range(num_steps + 1):
        dist_s = (dist_max_mm / num_steps) * s
        pck[s] = (dists < dist_s).float().mean()

    return pck


def get_auc(pck3d):
    auc = skmetrics.auc(range(pck3d.shape[0]), pck3d) / pck3d.shape[0]
    auc = round(auc, 3)

    return auc


# L2 distance between joints
def mepj_frame(joints_pred, joints_gt, num_steps=100, dist_max_mm=100, mode="default", valid_mask=None):
    # # relative to root joint
    if mode == "NHOT3D":
        joints_pred = joints_pred - joints_pred[:, 5:6, :]
        joints_gt = joints_gt - joints_gt[:, 5:6, :]
    else:
        joints_pred = joints_pred - joints_pred[:, :1, :]
        joints_gt = joints_gt - joints_gt[:, :1, :]

    joints_pred = torch.cat([joints_pred[0], joints_pred[1]], 0)
    joints_gt = torch.cat([joints_gt[0], joints_gt[1]], 0)

    # compute distances
    dists = torch.norm(joints_pred - joints_gt, p=2, dim=1)

    # Apply valid mask if provided
    if valid_mask is not None:
        # valid_mask: [2] for [left, right]
        # Expand to match joints: each hand has 21 joints
        num_joints_per_hand = joints_pred.shape[0] // 2
        mask = torch.cat([
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[0],
            torch.ones(num_joints_per_hand, device=dists.device) * valid_mask[1]
        ])
        # Only compute for valid joints
        valid_indices = mask > 0.5
        if valid_indices.sum() == 0:
            return torch.tensor(0.0, device=dists.device)
        dists = dists[valid_indices]

    return dists.mean()


def evaluate_joints_real(j3d_pred, j3d_gts, num_steps, mode="default", valid_mask=None):
    """
    Evaluate 3D joint predictions against ground truth.

    Args:
        j3d_pred: Predicted joints [2, 21, 3] for [left, right]
        j3d_gts: Ground truth joints (can be list or single tensor)
        num_steps: Number of steps for PCK computation
        mode: Evaluation mode ("NHOT3D" or "default")
        valid_mask: Optional [2] tensor indicating which hands are valid [left, right]
                   If None, both hands are considered valid (backward compatible)

    Returns:
        Dictionary with evaluation metrics
    """
    aucs = list()
    dist_max_mm = 100
    # Pick the GT candidate with the best right-root-relative AUC
    for j3d_gt in j3d_gts:
        right_root_relative_pck3d = right_root_relative_pck3d_frame(
            j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=dist_max_mm, mode=mode, valid_mask=valid_mask
        )
        aucs.append(get_auc(right_root_relative_pck3d))

    idx = np.argmax(aucs)

    j3d_gt = j3d_gts[idx]
    absolute_pck3d = absolute_pck3d_frame(
        j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=dist_max_mm, valid_mask=valid_mask
    )
    relative_pck3d = relative_pck3d_frame(
        j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=dist_max_mm, mode=mode, valid_mask=valid_mask
    )
    right_root_relative_pck3d = right_root_relative_pck3d_frame(
        j3d_pred, j3d_gt, num_steps=num_steps, dist_max_mm=dist_max_mm, mode=mode, valid_mask=valid_mask
    )

    joint_loss = mepj_frame(j3d_pred, j3d_gt, mode=mode, valid_mask=valid_mask).item()

    root_distance = [
        torch.norm((j3d_gt[0] - j3d_gt[1]), p=2, dim=-1)
        .min(-1)[0]
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


def to_scalar(v):
    """Convert tensor or number to Python scalar."""
    return v.item() if isinstance(v, torch.Tensor) else v


def evaluate_hand(pred, j3d_gts, device, num_steps, use_valid_mask=True):
    """Calculate N-HOT3D style evaluation metrics for a batch.

    Args:
        pred (dict[str, dict[str, torch.Tensor]]): Predictions from the network.
        j3d_gts (torch.Tensor): Ground truth joints with shape [B, 2, 21, 3].
        device (torch.device): CUDA / CPU device.
        num_steps (int): Threshold max (in mm) used for AUC computation.
        use_valid_mask (bool): If True, only evaluate hands detected by YOLO.
            Models without detection output all-valid masks, so this is a no-op
            for them.

    Returns:
        list[dict]: Per frame result dictionaries.
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
        for side in ("left", "right"):
            if "vertices" in pred[side]:
                hands[side]["vertices"] = pred[side]["vertices"][idx]

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
        scores = evaluate_joints_real(
            j3d_pred=j3d_pred * 1000,
            j3d_gts=j3d_gt * 1000,
            num_steps=num_steps,
            mode="NHOT3D",
            valid_mask=valid_mask_tensor,
        )
        hands["scores"] = scores
        frames.append(hands)
    return frames


def create_metrics_accumulator(num_steps):
    """Create a metrics accumulator dict for one category."""
    return {
        "pa_mpjpe_list": [],
        "mpjpe_list": [],
        "mpvpe_list": [],
        "abs_pa_mpjpe_list": [],
        "abs_mpjpe_list": [],
        "abs_mpvpe_list": [],
        "absolute_pck3d": np.zeros(num_steps + 1),
        "relative_pck3d": np.zeros(num_steps + 1),
        "right_root_relative_pck3d": np.zeros(num_steps + 1),
        "joint_loss": 0.0,
        "frame_index": 0,
    }


def accumulate_mpjpe(accum, outputs, gt_joints, valid_indices, device, is_mocap=False):
    """Compute and accumulate MPJPE metrics for given valid indices."""
    from .hand_evaluation import calc_MPJPE3d, calc_MPVPE3d, calc_PA_MPJPE
    from .eval_mocap import calc_MPJPE3d_mocap, calc_PA_MPJPE_mocap

    if len(valid_indices) == 0:
        return

    filtered_outputs = {
        "left": {k: v[valid_indices] for k, v in outputs["left"].items()},
        "right": {k: v[valid_indices] for k, v in outputs["right"].items()},
    }
    filtered_gt_joints = {
        "left": {k: v[valid_indices] for k, v in gt_joints["left"].items()},
        "right": {k: v[valid_indices] for k, v in gt_joints["right"].items()},
    }

    if is_mocap:
        pa_mpjpe = (
            calc_PA_MPJPE_mocap(
                filtered_outputs, filtered_gt_joints, device, mode="relative",
            )
            * 1000
        )
        mpjpe = (
            calc_MPJPE3d_mocap(
                filtered_outputs, filtered_gt_joints, device, mode="relative",
            )
            * 1000
        )
        abs_pa_mpjpe = (
            calc_PA_MPJPE_mocap(
                filtered_outputs, filtered_gt_joints, device, mode="absolute",
            )
            * 1000
        )
        abs_mpjpe = (
            calc_MPJPE3d_mocap(
                filtered_outputs, filtered_gt_joints, device, mode="absolute",
            )
            * 1000
        )
        mpvpe = 0.0
        abs_mpvpe = 0.0
    else:
        pa_mpjpe = (
            calc_PA_MPJPE(
                filtered_outputs, filtered_gt_joints, device,
                mode="relative", pred_key="j3d", coordinate="camera",
            )
            * 1000
        )
        mpjpe = (
            calc_MPJPE3d(
                filtered_outputs, filtered_gt_joints, device,
                mode="relative", pred_key="j3d", coordinate="camera",
            )
            * 1000
        )
        mpvpe = (
            calc_MPVPE3d(
                filtered_outputs, filtered_gt_joints, device,
                mode="relative", coordinate="camera",
            )
            * 1000
        )
        abs_pa_mpjpe = (
            calc_PA_MPJPE(
                filtered_outputs, filtered_gt_joints, device,
                mode="absolute", pred_key="j3d", coordinate="camera",
            )
            * 1000
        )
        abs_mpjpe = (
            calc_MPJPE3d(
                filtered_outputs, filtered_gt_joints, device,
                mode="absolute", pred_key="j3d", coordinate="camera",
            )
            * 1000
        )
        abs_mpvpe = (
            calc_MPVPE3d(
                filtered_outputs, filtered_gt_joints, device,
                mode="absolute", coordinate="camera",
            )
            * 1000
        )

    accum["pa_mpjpe_list"].append(to_scalar(pa_mpjpe))
    accum["mpjpe_list"].append(to_scalar(mpjpe))
    accum["mpvpe_list"].append(to_scalar(mpvpe))
    accum["abs_pa_mpjpe_list"].append(to_scalar(abs_pa_mpjpe))
    accum["abs_mpjpe_list"].append(to_scalar(abs_mpjpe))
    accum["abs_mpvpe_list"].append(to_scalar(abs_mpvpe))


def accumulate_pck(
    accum, outputs, j3d_gts, cat_indices, left_valid, right_valid,
    device, num_steps, is_mocap=False,
):
    """Compute and accumulate PCK/AUC metrics for given category indices."""
    from .eval_mocap import evaluate_hand_mocap

    if len(cat_indices) == 0:
        return

    cat_outputs_pck = {
        "left": {k: v[cat_indices] for k, v in outputs["left"].items()},
        "right": {k: v[cat_indices] for k, v in outputs["right"].items()},
        "left_valid": left_valid[cat_indices],
        "right_valid": right_valid[cat_indices],
    }
    cat_j3d_gts = j3d_gts[cat_indices]

    if is_mocap:
        frames = evaluate_hand_mocap(
            cat_outputs_pck, cat_j3d_gts, device, num_steps, use_valid_mask=True
        )
    else:
        frames = evaluate_hand(
            cat_outputs_pck, cat_j3d_gts, device, num_steps, use_valid_mask=True
        )

    for frame in frames:
        scores = frame["scores"]
        accum["absolute_pck3d"] += scores["absolute_pck3d"]
        accum["relative_pck3d"] += scores["relative_pck3d"]
        accum["right_root_relative_pck3d"] += scores["right_root_relative_pck3d"]
        accum["joint_loss"] += scores["joint_loss"]
        accum["frame_index"] += 1


def compute_and_print_metrics(accum, label):
    """Compute final metrics from accumulators, print them, and return as dict."""
    if len(accum["pa_mpjpe_list"]) > 0:
        mean_pa_mpjpe = sum(accum["pa_mpjpe_list"]) / len(accum["pa_mpjpe_list"])
        mean_mpjpe = sum(accum["mpjpe_list"]) / len(accum["mpjpe_list"])
        mean_mpvpe = sum(accum["mpvpe_list"]) / len(accum["mpvpe_list"])
        mean_abs_pa_mpjpe = sum(accum["abs_pa_mpjpe_list"]) / len(
            accum["abs_pa_mpjpe_list"]
        )
        mean_abs_mpjpe = sum(accum["abs_mpjpe_list"]) / len(accum["abs_mpjpe_list"])
        mean_abs_mpvpe = sum(accum["abs_mpvpe_list"]) / len(accum["abs_mpvpe_list"])
    else:
        mean_pa_mpjpe = mean_mpjpe = mean_mpvpe = 0.0
        mean_abs_pa_mpjpe = mean_abs_mpjpe = mean_abs_mpvpe = 0.0

    fi = max(accum["frame_index"], 1)
    absolute_pck3d = accum["absolute_pck3d"] / fi
    relative_pck3d = accum["relative_pck3d"] / fi
    right_root_relative_pck3d = accum["right_root_relative_pck3d"] / fi
    joint_loss_mean = accum["joint_loss"] / fi

    absolute_auc = get_auc(absolute_pck3d)
    relative_auc = get_auc(relative_pck3d)
    right_root_relative_auc = get_auc(right_root_relative_pck3d)

    print(f"\n=== {label} ===")
    print(
        f"[Relative] Mean PA-MPJPE: {mean_pa_mpjpe:.4f}mm, "
        f"Mean MPJPE: {mean_mpjpe:.4f}mm, "
        f"Mean MPVPE: {mean_mpvpe:.4f}mm"
    )
    print(
        f"[Absolute] Mean PA-MPJPE: {mean_abs_pa_mpjpe:.4f}mm, "
        f"Mean MPJPE: {mean_abs_mpjpe:.4f}mm, "
        f"Mean MPVPE: {mean_abs_mpvpe:.4f}mm"
    )
    print(
        f"Absolute AUC: {to_scalar(absolute_auc):.4f}, "
        f"Relative AUC: {to_scalar(relative_auc):.4f}, "
        f"Right Root Relative AUC: {to_scalar(right_root_relative_auc):.4f}, "
        f"Joint Loss: {joint_loss_mean:.4f}"
    )

    return {
        "Mean PA-MPJPE (Relative)": mean_pa_mpjpe,
        "Mean MPJPE (Relative)": mean_mpjpe,
        "Mean MPVPE (Relative)": mean_mpvpe,
        "Mean PA-MPJPE (Absolute)": mean_abs_pa_mpjpe,
        "Mean MPJPE (Absolute)": mean_abs_mpjpe,
        "Mean MPVPE (Absolute)": mean_abs_mpvpe,
        "Absolute AUC": to_scalar(absolute_auc),
        "Relative AUC": to_scalar(relative_auc),
        "Right Root Relative AUC": to_scalar(right_root_relative_auc),
        "Joint Loss": joint_loss_mean,
    }
