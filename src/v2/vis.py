#!/usr/bin/env python3
"""Compare EventEgoHandsV2 predictions with the ground truth in Rerun.

The inference path is identical to `src/v2/test.py` (YOLO segmentation ->
EventEgoHandsV2 with mask attention), and the predicted and ground-truth hands
are logged into one 3D view. The dataset is selected by `dataset_type` in the
config, exactly like train.py / test.py.

EEH-R annotations carry no mesh topology, so the ground-truth mesh reuses the
MANO faces the model returns. With `annotation_source: mocap` the ground truth is
16 joints and has no vertices, so only the skeleton is drawn.

Usage:
    python src/v2/vis.py --config src/v2/config/config_nhot3d.yaml \
        --sequence-ids P0002_016222d1
    python src/v2/vis.py --config src/v2/config/config_eehr.yaml --sequence-ids P04_01
    # Save to .rrd (serve it later with src/viewer/serve_rrd.sh):
    python src/v2/vis.py --config ... --recording out.rrd
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
from ultralytics import YOLO

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from factory import get_hand_model_v2
from utils import HandEstimationConfig
from utils.rerun_vis import (
    GT_COLOR,
    HANDS,
    PRED_COLOR,
    PRED_COLOR_2D,
    add_rerun_args,
    build_blueprint,
    clear_hand,
    event_frame_to_rgb,
    init_rerun,
    keep_alive_if_web,
    lnes_to_rgb,
    log_hand,
    overlay_mask,
    sequence_subset,
)
from v2.detection_utils import get_yolo_results
from v2.test import create_test_dataset, dataset_tag

os.environ["ERPC"] = "1"

# The MANO decoder is per-sample here, but one frame per batch keeps the YOLO
# detections and the logged frame index aligned one to one.
BATCH_SIZE = 1

# YOLO class ids, matching EventEgoHandsV2._apply_masks_to_images.
CLASS_TO_HAND = {0: "left", 1: "right"}


def is_valid(flag) -> bool:
    """Read a per-hand `valid` entry, which the datasets may store as a list or tensor."""
    if isinstance(flag, (list, tuple)):
        flag = flag[0]
    if torch.is_tensor(flag):
        flag = flag.reshape(-1)[0].item()
    return bool(flag)


def sequence_key_fn(dataset, is_real: bool):
    """Return the per-index string that `--sequence-ids` is matched against."""
    if is_real:
        return lambda i: dataset.frame_list[i][0]
    return lambda i: dataset.image_files[i]


def log_yolo_masks(results, event_frame_rgb: np.ndarray) -> None:
    """Overlay the detected hand masks on the event frame."""
    overlay = event_frame_rgb.copy()
    masks = getattr(results, "masks", None)
    boxes = getattr(results, "boxes", None)

    if masks is not None and boxes is not None and len(boxes) > 0:
        classes = boxes.cls
        for det_idx in range(len(boxes)):
            hand = CLASS_TO_HAND.get(int(classes[det_idx].item()))
            color = PRED_COLOR_2D.get(hand, (255, 255, 255))
            overlay = overlay_mask(overlay, masks.data[det_idx], color)

    rr.log("2D/yolo_mask", rr.Image(overlay))


def visualize(yolo_model, hand_net, cfg: HandEstimationConfig, args, device) -> None:
    is_real = dataset_tag(cfg) == "eehr"

    base_dataset, DatasetClass = create_test_dataset(cfg)
    dataset = sequence_subset(
        base_dataset, args.sequence_ids, sequence_key_fn(base_dataset, is_real)
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=False,
    )

    total = len(loader) if args.num_samples <= 0 else min(args.num_samples, len(loader))

    with torch.no_grad():
        for frame, batch in enumerate(tqdm(loader, total=total, desc="Visualizing", unit="frame")):
            if 0 < args.num_samples <= frame:
                break
            rr.set_time_sequence("frame", frame)

            batch = DatasetClass.to_device(batch, device)
            yolo_results = get_yolo_results(batch, yolo_model, cfg, device, BATCH_SIZE)

            lnes = batch["lnes"].to(device)
            outputs = hand_net(lnes, yolo_results)

            # --- Prediction (hands YOLO did not detect are cleared) ---
            for hand in HANDS:
                if not is_valid(outputs.get(f"{hand}_valid", torch.ones(1))):
                    clear_hand("3D/pred", hand)
                    continue
                log_hand(
                    "3D/pred",
                    hand,
                    PRED_COLOR[hand],
                    vertices=outputs[hand]["vertices"][0].cpu().numpy(),
                    triangles=outputs[hand]["faces"].cpu().numpy(),
                    joints=outputs[hand]["j3d"][0].cpu().numpy(),
                )

            # --- Ground truth ---
            gt_hand_poses = batch["gt_hand_poses"]
            gt_joints = batch["gt_joints"]
            for hand in HANDS:
                if not is_valid(gt_hand_poses[hand]["valid"]):
                    clear_hand("3D/gt", hand)
                    continue

                gt_vertices = gt_joints[hand]["vertices_3d"][0].cpu().numpy()
                if "triangles" in gt_hand_poses[hand]:
                    gt_faces = gt_hand_poses[hand]["triangles"][0].cpu().numpy()
                else:
                    # EEH-R has no mesh topology; borrow the MANO faces.
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

            # --- 2D views ---
            event_frame_rgb = event_frame_to_rgb(batch["event_frame"][0])
            rr.log("2D/event_frame", rr.Image(event_frame_rgb))
            rr.log("2D/lnes", rr.Image(lnes_to_rgb(lnes[0])))
            log_yolo_masks(yolo_results[0], event_frame_rgb)
            rr.log(
                "logs",
                rr.TextLog(f"{batch['sequence_name'][0]}_frame_{batch['frame_number'][0]}"),
            )

            if not is_real and "image" in gt_hand_poses:
                rr.log("2D/rgb", rr.Image(np.asarray(gt_hand_poses["image"][0])))


def main(cfg: HandEstimationConfig, args) -> None:
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device_first = cfg.devices[0]
    device = torch.device(f"cuda:{device_first}" if torch.cuda.is_available() else "cpu")

    yolo_model = None
    if not getattr(cfg, "use_gt_mask", False):
        print(f"Loading YOLO model from: {cfg.yolo_checkpoint_path}")
        yolo_model = YOLO(cfg.yolo_checkpoint_path)
        yolo_model.to(device)
    else:
        print("use_gt_mask=True: skipping YOLO model loading")

    hand_net = get_hand_model_v2(cfg, device)

    checkpoint_path = args.checkpoint or cfg.test.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing_keys, unexpected_keys = hand_net.load_state_dict(
        {k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False
    )
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    hand_net.eval()
    print(f"Loading hand model from: {checkpoint_path}")

    views = [
        ("Event Frame", "/2D/event_frame"),
        ("LNES", "/2D/lnes"),
        ("YOLO Mask", "/2D/yolo_mask"),
    ]
    if dataset_tag(cfg) != "eehr" and cfg.test.is_get_image:
        views.append(("RGB", "/2D/rgb"))

    init_rerun(f"v2_vis_{dataset_tag(cfg)}", args, build_blueprint(views))

    visualize(yolo_model, hand_net, cfg, args, device)
    print("Done")
    keep_alive_if_web(args)


def arg_parse():
    parser = argparse.ArgumentParser(description="EventEgoHandsV2 prediction viewer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Override test.checkpoint_path"
    )
    parser.add_argument(
        "--sequence-ids", type=str, nargs="+", default=None,
        help="Only visualize these sequences (e.g. P0002_016222d1 / P04_01)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=0, help="Number of frames to visualize (0 = all)"
    )
    add_rerun_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    cfg = OmegaConf.load(args.config)
    main(cfg, args)
