#!/usr/bin/env python3
"""Compare EventEgoHandsV1 predictions with the ground truth on N-HOT3D, in Rerun.

The inference path is identical to `src/v1/test_nhot3d.py` (U-Net segmentation ->
threshold -> dilation -> event-cloud filtering -> EventEgoHandsV1), and the
predicted and ground-truth meshes are logged into one 3D view so they can be
overlaid or toggled per hand.

Usage:
    python src/v1/vis_nhot3d.py --config src/v1/config/config_test_nhot3d.yaml \
        --sequence-ids P0002_016222d1
    # Web viewer (open http://localhost:9090 in a browser):
    python src/v1/vis_nhot3d.py --config ... --web-viewer
    # Save to .rrd (serve it later with src/viewer/serve_rrd.sh):
    python src/v1/vis_nhot3d.py --config ... --recording out.rrd
"""

import argparse
import os
import random
import sys

import numpy as np
import rerun as rr
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from datasets.nhot3d_dataset import NHOT3DDataset
from factory import get_hand_model
from models import load_unet
from utils import (
    HandEstimationConfig,
    dilation_masks,
    mask_event_cloud_batch,
    pc_normalize_batch,
)
from utils.rerun_vis import (
    GT_COLOR,
    HANDS,
    PRED_COLOR,
    PRED_COLOR_2D,
    add_rerun_args,
    build_blueprint,
    clear_hand,
    draw_joints_2d,
    event_cloud_to_rgb,
    event_frame_to_rgb,
    init_rerun,
    keep_alive_if_web,
    lnes_to_rgb,
    log_hand,
    mask_to_rgb,
    project_3d_to_2d,
    sequence_subset,
)

os.environ["ERPC"] = "1"

# MANORegressor feeds `betas[0]` to the MANO layer, so every sample in a batch
# would share the first sample's shape. One frame per batch keeps each
# visualised mesh consistent with its own predicted shape.
BATCH_SIZE = 1


def recursive_collate_fn(key, values):
    if key == "events":
        events = [ev.clone().detach().permute(1, 0) for ev in values]
        return torch.nn.utils.rnn.pad_sequence(events, batch_first=True, padding_value=0)
    elif isinstance(values[0], dict):
        return {k: recursive_collate_fn(k, [v[k] for v in values]) for k in values[0]}
    elif isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    else:
        return values


def collate_fn(batch):
    return {key: recursive_collate_fn(key, [b[key] for b in batch]) for key in batch[0]}


def is_valid(flag) -> bool:
    """Read a per-hand `valid` entry, which collate may leave as a list or tensor."""
    if isinstance(flag, (list, tuple)):
        flag = flag[0]
    if torch.is_tensor(flag):
        flag = flag.reshape(-1)[0].item()
    return bool(flag)


def create_dataset(cfg: HandEstimationConfig) -> NHOT3DDataset:
    return NHOT3DDataset(
        event_input_dir=cfg.test.event_input_dir,
        gt_file_dir=cfg.test.gt_file_dir,
        object_library_path=cfg.test.object_library_path,
        mano_hand_model_path=cfg.mano_hand_model_path,
        is_get_image=cfg.test.is_get_image,
        augmentation=False,
        is_normalize=False,
        is_rotate=False,
        mode=cfg.dataset_mode,
        output_size=(cfg.output_height, cfg.output_width),
        event_number_threshold=cfg.event_number_threshold,
        time_bin=cfg.time_bin,
        is_sequence=cfg.is_sequence,
        dataset_divide=cfg.dataset_divide,
    )


def image_views(cfg, args) -> list:
    views = [
        ("Event Frame", "/2D/event_frame"),
        ("LNES", "/2D/lnes"),
        ("Segmentation", "/2D/segmentation"),
        ("Masked Events", "/2D/masked_events"),
    ]
    if cfg.test.is_get_image:
        views.append(("RGB", "/2D/rgb"))
    if args.project_2d:
        views.append(("Pred Joints 2D", "/2D/pred_joints"))
    return views


def visualize(net, segment_net, cfg: HandEstimationConfig, args, device) -> None:
    base_dataset = create_dataset(cfg)
    dataset = sequence_subset(
        base_dataset, args.sequence_ids, lambda i: base_dataset.event_input_files[i]
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    focal_length = float(getattr(cfg, "focal_length", 5000.0))
    total = len(loader) if args.num_samples <= 0 else min(args.num_samples, len(loader))

    with torch.no_grad():
        for frame, batch in enumerate(tqdm(loader, total=total, desc="Visualizing", unit="frame")):
            if 0 < args.num_samples <= frame:
                break
            rr.set_time_sequence("frame", frame)

            batch = NHOT3DDataset.to_device(batch, device)
            events = batch["events"]
            event_frames = batch["LNES"]

            # --- Ground truth ---
            gt_hand_poses = batch["gt_hand_poses"]
            gt_joints = batch["gt_joints"]
            for hand in HANDS:
                if not is_valid(gt_hand_poses[hand]["valid"]):
                    clear_hand("3D/gt", hand)
                    continue
                log_hand(
                    "3D/gt",
                    hand,
                    GT_COLOR[hand],
                    vertices=gt_joints[hand]["vertices_3d"][0].cpu().numpy(),
                    triangles=gt_hand_poses[hand]["triangles"][0].cpu().numpy(),
                    joints=gt_joints[hand]["joints_3d"][0].cpu().numpy(),
                )

            # --- Segmentation (same as test_nhot3d.py) ---
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
            segment_outputs = torch.tensor(segment_outputs, dtype=torch.float32).to(device)
            segment_outputs = segment_outputs.unsqueeze(1)

            masked_events = mask_event_cloud_batch(segment_outputs, events, 2048)
            masked_event_image = event_cloud_to_rgb(
                masked_events[0], cfg.output_height, cfg.output_width
            )

            masked_events[:, :, :3] = pc_normalize_batch(masked_events[:, :, :3])
            masked_events = masked_events.permute(0, 2, 1).to(device)

            # --- Hand reconstruction ---
            outputs = net(masked_events)

            for hand in HANDS:
                log_hand(
                    "3D/pred",
                    hand,
                    PRED_COLOR[hand],
                    vertices=outputs[hand]["vertices"][0].cpu().numpy(),
                    triangles=outputs[hand]["faces"].cpu().numpy(),
                    joints=outputs[hand]["j3d"][0].cpu().numpy(),
                )

            # --- 2D views ---
            event_frame_rgb = event_frame_to_rgb(batch["event_frame"][0])
            rr.log("2D/event_frame", rr.Image(event_frame_rgb))
            rr.log("2D/lnes", rr.Image(lnes_to_rgb(event_frames[0])))
            rr.log("2D/segmentation", rr.Image(mask_to_rgb(segment_outputs[0, 0])))
            rr.log("2D/masked_events", rr.Image(masked_event_image))
            rr.log("logs", rr.TextLog(str(batch["event_file_path"][0])))

            if cfg.test.is_get_image and "image" in gt_hand_poses:
                rr.log("2D/rgb", rr.Image(np.asarray(gt_hand_poses["image"][0])))

            if args.project_2d:
                overlay = event_frame_rgb.copy()
                height, width = overlay.shape[:2]
                principal_point = np.array([width / 2.0, height / 2.0], dtype=np.float32)
                for hand in HANDS:
                    joints_2d = project_3d_to_2d(
                        outputs[hand]["j3d"][0].cpu().numpy(),
                        np.zeros(3, dtype=np.float32),
                        focal_length,
                    )
                    overlay = draw_joints_2d(
                        overlay, joints_2d + principal_point, PRED_COLOR_2D[hand]
                    )
                rr.log("2D/pred_joints", rr.Image(overlay))


def main(cfg: HandEstimationConfig, args) -> None:
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device_first = cfg.devices[0]
    device = torch.device(f"cuda:{device_first}" if torch.cuda.is_available() else "cpu")

    print(f"Loading segmentation model from: {cfg.segmentation_checkpoint_path}")
    segment_net = load_unet(
        cfg.segmentation_checkpoint_path,
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        device=device,
    )
    segment_net.eval()

    checkpoint_path = args.checkpoint or cfg.test.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    net = get_hand_model(cfg).to(device)
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    net.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    net.eval()
    print(f"Loading hand model from: {checkpoint_path}")

    blueprint = build_blueprint(image_views(cfg, args))
    init_rerun("v1_vis_nhot3d", args, blueprint)

    visualize(net, segment_net, cfg, args, device)
    print("Done")
    keep_alive_if_web(args)


def arg_parse():
    parser = argparse.ArgumentParser(description="EventEgoHandsV1 prediction viewer (N-HOT3D)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "config_test_nhot3d.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Override test.checkpoint_path"
    )
    parser.add_argument(
        "--sequence-ids", type=str, nargs="+", default=None,
        help="Only visualize these sequences (e.g. P0002_016222d1)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=0, help="Number of frames to visualize (0 = all)"
    )
    parser.add_argument(
        "--project-2d",
        action="store_true",
        help="Overlay predicted joints on the event frame with a pinhole "
             "approximation (ignores the Aria distortion model)",
    )
    add_rerun_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    cfg = OmegaConf.load(args.config)
    main(cfg, args)
