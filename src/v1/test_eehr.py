import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from datasets.eehr_dataset import EEHRDataset, load_light_dark_mapping
from models import load_unet
from factory import get_hand_model
from utils import (
    HandEstimationConfig,
    dilation_masks,
    calc_PA_MPJPE,
    calc_MPJPE3d,
    calc_MPVPE3d,
)
from utils.joint_evaluation import (
    compute_and_print_metrics,
    create_metrics_accumulator,
    evaluate_hand,
)

from utils.eval_mocap import (
    calc_MPJPE3d_mocap,
    calc_PA_MPJPE_mocap,
    evaluate_hand_mocap,
)
from v1.train_eehr import (
    transform_event_coords_to_input,
    mask_event_cloud_batch_local,
    pc_normalize_batch_local,
)

log = logging.getLogger(os.path.basename(__file__))

os.environ["ERPC"] = "1"


# ---------------------------------------------------------------------------
# Collate functions (from train_eehr.py)
# ---------------------------------------------------------------------------

def recursive_collate_fn(key, values):
    if key == "events":
        # (5, N) -> (N, 5), then pad to same length
        events = [ev.clone().detach().permute(1, 0) for ev in values]
        padded_events = torch.nn.utils.rnn.pad_sequence(
            events, batch_first=True, padding_value=0
        )  # (batch_size, N, 5)
        return padded_events
    elif isinstance(values[0], dict):
        return {
            k: recursive_collate_fn(k, [v[k] for v in values])
            for k in values[0].keys()
        }
    elif isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    elif isinstance(values[0], (str, bool, int, float, type(None))):
        return values
    else:
        return values


def collate_fn(batch):
    batched_data = {
        key: recursive_collate_fn(key, [b[key] for b in batch])
        for key in batch[0].keys()
    }
    return batched_data


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_test_dataset(cfg):
    return EEHRDataset(
        dataset_root=cfg.real_dataset_root,
        split="test",
        fps=getattr(cfg, "real_dataset_fps", 30),
        output_size=(cfg.output_height, cfg.output_width),
        use_center_crop=getattr(cfg, "use_center_crop", False),
        ignore_files_count=getattr(cfg, "ignore_files_count", 0),
        dataset_divide=1,
        load_h5=True,
        event_number_threshold=cfg.event_number_threshold,
        annotation_source=getattr(cfg, "annotation_source", "mano"),
        annotation_fps=getattr(cfg, "annotation_fps", 30),
        lnes_from_h5=getattr(cfg, "lnes_from_h5", False),
        lnes_window_sec=getattr(cfg, "lnes_window_sec", 1.0 / 30.0),
    )


# ---------------------------------------------------------------------------
# Light / Dark mapping and per-category metrics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def test_eehr(net, segment_net, cfg, device):
    """Evaluate full pipeline (UNet segmentation + EventEgoHandsV1) on real data."""
    net.eval()
    segment_net.eval()

    is_mocap = getattr(cfg, "annotation_source", "mano") == "mocap"

    test_dataset = create_test_dataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    light_dark_mapping = load_light_dark_mapping(cfg.real_dataset_root)

    num_steps = 100  # AUC thresholds: 0mm to 100mm
    categories = ["all", "light", "dark"]
    accum = {cat: create_metrics_accumulator(num_steps) for cat in categories}

    with torch.no_grad():
        for index, batch in enumerate(tqdm(
            test_loader, total=len(test_loader), desc="Evaluation", unit="batch"
        )):
            batch = EEHRDataset.to_device(batch, device)
            events = batch["events"]
            event_frames = batch["lnes"]

            input_size = event_frames.shape[-1]
            events = transform_event_coords_to_input(events, input_size)

            # Segmentation pipeline
            segment_outputs = segment_net(event_frames)
            segment_outputs = torch.sigmoid(segment_outputs)
            segment_outputs = segment_outputs > cfg.sigmoid_threshold
            batch_size = segment_outputs.shape[0]
            segment_outputs = dilation_masks(
                segment_outputs,
                batch_size=batch_size,
                kernel_size=cfg.kernel_size,
                itterations=cfg.itterations,
            )
            segment_outputs = (
                torch.tensor(segment_outputs, dtype=torch.float32)
                .to(device)
                .unsqueeze(1)
            )

            # Mask and normalize event cloud
            masked_events = mask_event_cloud_batch_local(
                segment_outputs, events, 2048
            )
            masked_events[:, :, :3] = pc_normalize_batch_local(
                masked_events[:, :, :3], input_size=input_size
            )
            masked_events = masked_events.permute(0, 2, 1)  # (B, 5, N)
            masked_events = masked_events.to(device)

            # Hand estimation inference
            outputs = net(masked_events)

            # Model output is already in HOT3D order (20 joints)
            # because MANOHandModel applies mano_joint_mapping by default.
            eval_outputs = {}
            for hand_type in ["left", "right"]:
                eval_outputs[hand_type] = dict(outputs[hand_type])

            # Ground truth joints
            gt_joints = batch["gt_joints"]

            # HOT3D style AUC metrics
            left_gt_joints = gt_joints["left"]["joints_3d"]
            right_gt_joints = gt_joints["right"]["joints_3d"]
            j3d_gts = torch.cat(
                [left_gt_joints.unsqueeze(1), right_gt_joints.unsqueeze(1)], dim=1
            )  # [B, 2, J, 3]

            # Determine light/dark category per item
            sequence_names = batch["sequence_name"]
            item_categories = [
                light_dark_mapping.get(name, "unknown") for name in sequence_names
            ]

            # --- Accumulate metrics per category ---
            for cat in categories:
                if cat == "all":
                    cat_indices = list(range(batch_size))
                else:
                    cat_indices = [
                        i for i, c in enumerate(item_categories) if c == cat
                    ]
                if len(cat_indices) == 0:
                    continue

                # Filter outputs and gt for this category
                filtered_outputs = {
                    hand_type: {k: v[cat_indices] for k, v in eval_outputs[hand_type].items()}
                    for hand_type in ["left", "right"]
                }
                filtered_gt_joints = {
                    hand_type: {k: v[cat_indices] for k, v in gt_joints[hand_type].items()}
                    for hand_type in ["left", "right"]
                }

                # MPJPE metrics
                if is_mocap:
                    pa_mpjpe = (
                        calc_PA_MPJPE_mocap(
                            filtered_outputs, filtered_gt_joints, device, mode="relative",
                        ) * 1000
                    )
                    mpjpe = (
                        calc_MPJPE3d_mocap(
                            filtered_outputs, filtered_gt_joints, device, mode="relative",
                        ) * 1000
                    )
                    abs_pa_mpjpe = (
                        calc_PA_MPJPE_mocap(
                            filtered_outputs, filtered_gt_joints, device, mode="absolute",
                        ) * 1000
                    )
                    abs_mpjpe = (
                        calc_MPJPE3d_mocap(
                            filtered_outputs, filtered_gt_joints, device, mode="absolute",
                        ) * 1000
                    )
                    mpvpe = 0.0
                    abs_mpvpe = 0.0
                else:
                    pa_mpjpe = (
                        calc_PA_MPJPE(
                            filtered_outputs, filtered_gt_joints, device,
                            mode="relative", pred_key="j3d", coordinate="camera",
                        ) * 1000
                    )
                    mpjpe = (
                        calc_MPJPE3d(
                            filtered_outputs, filtered_gt_joints, device,
                            mode="relative", pred_key="j3d", coordinate="camera",
                        ) * 1000
                    )
                    mpvpe = (
                        calc_MPVPE3d(
                            filtered_outputs, filtered_gt_joints, device,
                            mode="relative", coordinate="camera",
                        ) * 1000
                    )
                    abs_pa_mpjpe = (
                        calc_PA_MPJPE(
                            filtered_outputs, filtered_gt_joints, device,
                            mode="absolute", pred_key="j3d", coordinate="camera",
                        ) * 1000
                    )
                    abs_mpjpe = (
                        calc_MPJPE3d(
                            filtered_outputs, filtered_gt_joints, device,
                            mode="absolute", pred_key="j3d", coordinate="camera",
                        ) * 1000
                    )
                    abs_mpvpe = (
                        calc_MPVPE3d(
                            filtered_outputs, filtered_gt_joints, device,
                            mode="absolute", coordinate="camera",
                        ) * 1000
                    )

                _s = lambda v: v.item() if isinstance(v, torch.Tensor) else v
                accum[cat]["pa_mpjpe_list"].append(_s(pa_mpjpe))
                accum[cat]["mpjpe_list"].append(_s(mpjpe))
                accum[cat]["mpvpe_list"].append(_s(mpvpe))
                accum[cat]["abs_pa_mpjpe_list"].append(_s(abs_pa_mpjpe))
                accum[cat]["abs_mpjpe_list"].append(_s(abs_mpjpe))
                accum[cat]["abs_mpvpe_list"].append(_s(abs_mpvpe))

                # PCK / AUC metrics
                cat_j3d_gts = j3d_gts[cat_indices]

                if is_mocap:
                    frames = evaluate_hand_mocap(
                        filtered_outputs, cat_j3d_gts, device, num_steps, use_valid_mask=False
                    )
                else:
                    frames = evaluate_hand(filtered_outputs, cat_j3d_gts, device, num_steps)

                for frame in frames:
                    scores = frame["scores"]
                    accum[cat]["absolute_pck3d"] += scores["absolute_pck3d"]
                    accum[cat]["relative_pck3d"] += scores["relative_pck3d"]
                    accum[cat]["right_root_relative_pck3d"] += scores["right_root_relative_pck3d"]
                    accum[cat]["joint_loss"] += scores["joint_loss"]
                    accum[cat]["frame_index"] += 1

    # Compute and print metrics per category
    all_metrics = {}
    for cat in categories:
        metrics = compute_and_print_metrics(accum[cat], cat.upper())
        prefixed = {f"{cat}/{k}": v for k, v in metrics.items()}
        all_metrics.update(prefixed)

    wandb.log(all_metrics)

    return all_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: HandEstimationConfig) -> None:
    torch.cuda.empty_cache()

    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v1_test_eehr_{cfg.model}_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(
        f"cuda:{device_first}" if torch.cuda.is_available() else "cpu"
    )

    # Load segmentation model (frozen)
    print(f"Loading segmentation model from: {cfg.segmentation_checkpoint_path}")
    segment_net = load_unet(
        cfg.segmentation_checkpoint_path,
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        device=device,
    )
    for param in segment_net.parameters():
        param.requires_grad = False
    segment_net.eval()

    # Load hand estimation model
    net = get_hand_model(cfg)
    log.info(f"model parameters: {sum(p.numel() for p in net.parameters())}")
    net = net.to(device)

    checkpoint_path = cfg.test.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = state_dict[key]
    net.load_state_dict(new_state_dict)
    log.info(f"Load hand model from {checkpoint_path}")

    print(f"annotation_source: {getattr(cfg, 'annotation_source', 'mano')}")
    print(f"annotation_fps: {getattr(cfg, 'annotation_fps', 30)}")

    test_eehr(net, segment_net, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_train_eehr.yaml"),
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
