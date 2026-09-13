#!/usr/bin/env python3
"""Compare EventEgoHandsV1 predictions with the ground truth on EEH-R, in Rerun.

The inference path is identical to `src/v1/test_eehr.py` (event coordinate
transform -> U-Net segmentation -> threshold -> dilation -> event-cloud filtering
-> EventEgoHandsV1), and the predicted and ground-truth hands are logged into one
3D view.

EEH-R annotations carry no mesh topology, so the ground-truth mesh reuses the
MANO faces the model returns. With `annotation_source: mocap` the ground truth is
16 joints and has no vertices, so only the skeleton is drawn.

Usage:
    python src/v1/vis_eehr.py --config src/v1/config/config_train_eehr.yaml \
        --sequence-ids P04_01
    # Save to .rrd (serve it later with src/viewer/serve_rrd.sh):
    python src/v1/vis_eehr.py --config ... --recording out.rrd
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

from datasets.eehr_dataset import EEHRDataset
from factory import get_hand_model
from models import load_unet
from utils import HandEstimationConfig, dilation_masks
from utils.rerun_vis import (
    GT_COLOR,
    HANDS,
    PRED_COLOR,
    add_rerun_args,
    build_blueprint,
    clear_hand,
    event_cloud_to_rgb,
    event_frame_to_rgb,
    init_rerun,
    keep_alive_if_web,
    lnes_to_rgb,
    log_hand,
    mask_to_rgb,
    sequence_subset,
)
from v1.test_eehr import collate_fn, create_test_dataset
from v1.train_eehr import (
    mask_event_cloud_batch_local,
    pc_normalize_batch_local,
    transform_event_coords_to_input,
)

os.environ["ERPC"] = "1"

# See src/v1/vis_nhot3d.py: MANORegressor shares `betas[0]` across a batch.
BATCH_SIZE = 1


def is_valid(flag) -> bool:
    """Read a per-hand `valid` entry, which collate may leave as a list or tensor."""
    if isinstance(flag, (list, tuple)):
        flag = flag[0]
    if torch.is_tensor(flag):
        flag = flag.reshape(-1)[0].item()
    return bool(flag)


def image_views(has_grayscale: bool) -> list:
    views = [
        ("Event Frame", "/2D/event_frame"),
        ("LNES", "/2D/lnes"),
        ("Segmentation", "/2D/segmentation"),
        ("Masked Events", "/2D/masked_events"),
    ]
    if has_grayscale:
        views.append(("Grayscale", "/2D/grayscale"))
    return views


def visualize(net, segment_net, cfg: HandEstimationConfig, args, device) -> None:
    base_dataset = create_test_dataset(cfg)
    # The evaluation dataset does not load the reference images; the viewer does.
    base_dataset.load_grayscale_image = args.grayscale
    dataset = sequence_subset(
        base_dataset, args.sequence_ids, lambda i: base_dataset.frame_list[i][0]
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    total = len(loader) if args.num_samples <= 0 else min(args.num_samples, len(loader))

    with torch.no_grad():
        for frame, batch in enumerate(tqdm(loader, total=total, desc="Visualizing", unit="frame")):
            if 0 < args.num_samples <= frame:
                break
            rr.set_time_sequence("frame", frame)

            batch = EEHRDataset.to_device(batch, device)
            events = batch["events"]
            event_frames = batch["lnes"]

            input_size = event_frames.shape[-1]
            events = transform_event_coords_to_input(events, input_size)

            # --- Segmentation (same as test_eehr.py) ---
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
                torch.tensor(segment_outputs, dtype=torch.float32).to(device).unsqueeze(1)
            )

            masked_events = mask_event_cloud_batch_local(segment_outputs, events, 2048)
            masked_event_image = event_cloud_to_rgb(
                masked_events[0], input_size, input_size
            )

            masked_events[:, :, :3] = pc_normalize_batch_local(
                masked_events[:, :, :3], input_size=input_size
            )
            masked_events = masked_events.permute(0, 2, 1).to(device)

            # --- Hand reconstruction ---
            outputs = net(masked_events)

            # --- Ground truth (mesh topology borrowed from the MANO output) ---
            gt_hand_poses = batch["gt_hand_poses"]
            gt_joints = batch["gt_joints"]
            for hand in HANDS:
                if not is_valid(gt_hand_poses[hand]["valid"]):
                    clear_hand("3D/gt", hand)
                    continue

                gt_vertices = gt_joints[hand]["vertices_3d"][0].cpu().numpy()
                gt_faces = outputs[hand]["faces"].cpu().numpy()
                # mocap annotations carry joints only, so the zero-filled
                # placeholder vertices are not drawn.
                if not np.any(gt_vertices) or len(gt_vertices) != gt_faces.max() + 1:
                    gt_vertices, gt_faces = None, None

                log_hand(
                    "3D/gt",
                    hand,
                    GT_COLOR[hand],
                    vertices=gt_vertices,
                    triangles=gt_faces,
                    joints=gt_joints[hand]["joints_3d"][0].cpu().numpy(),
                )

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
            rr.log("2D/event_frame", rr.Image(event_frame_to_rgb(batch["event_frame"][0])))
            rr.log("2D/lnes", rr.Image(lnes_to_rgb(event_frames[0])))
            rr.log("2D/segmentation", rr.Image(mask_to_rgb(segment_outputs[0, 0])))
            rr.log("2D/masked_events", rr.Image(masked_event_image))
            rr.log(
                "logs",
                rr.TextLog(f"{batch['sequence_name'][0]}_frame_{batch['frame_number'][0]}"),
            )

            if "grayscale_image" in batch:
                gray = np.asarray(batch["grayscale_image"][0])
                rr.log("2D/grayscale", rr.Image(gray))


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
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    net.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    net.eval()
    print(f"Loading hand model from: {checkpoint_path}")

    blueprint = build_blueprint(image_views(args.grayscale))
    init_rerun("v1_vis_eehr", args, blueprint)

    visualize(net, segment_net, cfg, args, device)
    print("Done")
    keep_alive_if_web(args)


def arg_parse():
    parser = argparse.ArgumentParser(description="EventEgoHandsV1 prediction viewer (EEH-R)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "config_train_eehr.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Override test.checkpoint_path"
    )
    parser.add_argument(
        "--sequence-ids", type=str, nargs="+", default=None,
        help="Only visualize these sequences (e.g. P04_01)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=0, help="Number of frames to visualize (0 = all)"
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Also load and show the grayscale reference images",
    )
    add_rerun_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    cfg = OmegaConf.load(args.config)
    main(cfg, args)
