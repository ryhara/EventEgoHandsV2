"""Evaluate EventEgoHandsV2 on N-HOT3D (synthetic) or EEH-R (real).

The dataset is selected by `dataset_type` in the config file:
    python test.py --config config/config_nhot3d.yaml
    python test.py --config config/config_eehr.yaml
"""

import argparse
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
from ultralytics import YOLO

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from datasets.eehr_dataset import EEHRDataset, load_light_dark_mapping
from datasets.nhot3d_yolo_dataset import NHOT3DYOLODataset
from factory import get_hand_model_v2
from utils import HandEstimationConfig
from utils.joint_evaluation import (
    accumulate_mpjpe,
    accumulate_pck,
    compute_and_print_metrics,
    create_metrics_accumulator,
)
from v2.detection_utils import get_yolo_results

os.environ["ERPC"] = "1"


def dataset_tag(cfg) -> str:
    return "eehr" if getattr(cfg, "dataset_type", "synth") == "real" else "nhot3d"


def create_test_dataset(cfg):
    """Create test dataset based on dataset_type in config.

    Returns:
        Dataset instance and its class (for to_device method)
    """
    if getattr(cfg, "dataset_type", "synth") == "real":
        dataset = EEHRDataset(
            dataset_root=cfg.real_dataset_root,
            split="test",
            fps=getattr(cfg, "real_dataset_fps", 30),
            output_size=(cfg.output_height, cfg.output_width),
            use_center_crop=getattr(cfg, "use_center_crop", False),
            ignore_files_count=cfg.ignore_files_count,
            dataset_divide=1,  # Use full dataset for evaluation
            annotation_source=getattr(cfg, "annotation_source", "mano"),
            annotation_fps=getattr(cfg, "annotation_fps", 30),
            lnes_from_h5=getattr(cfg, "lnes_from_h5", False),
            lnes_window_sec=getattr(cfg, "lnes_window_sec", 1.0 / 30.0),
            use_gt_mask=getattr(cfg, "use_gt_mask", False),
            gt_mask_root=getattr(cfg, "gt_mask_root", None),
            allow_subframe_gt_mask=getattr(cfg, "allow_subframe_gt_mask", False),
        )
        return dataset, EEHRDataset
    else:
        dataset = NHOT3DYOLODataset(
            yolo_image_dir=cfg.test.yolo_image_dir + "/test",
            gt_file_dir=cfg.test.gt_file_dir + "/test",
            object_library_path=cfg.test.object_library_path,
            mano_hand_model_path=cfg.mano_hand_model_path,
            is_get_image=cfg.test.is_get_image,
            rgb_dir=cfg.test.rgb_dir,
            mode=cfg.dataset_mode,
            output_size=(cfg.output_height, cfg.output_width),
            use_center_crop=getattr(cfg, "use_center_crop", True),
            augmentation=False,  # Disable augmentation for evaluation
            ignore_files_count=cfg.ignore_files_count,
            dataset_divide=1,  # Use full dataset for evaluation
            lnes_blosc2_dir=getattr(cfg.test, "lnes_blosc2_dir", "") + "/test",
        )
        return dataset, NHOT3DYOLODataset


def test(yolo_model, hand_net, cfg: HandEstimationConfig, device):
    """Evaluate YOLO-based segmentation and hand estimation with mask attention V2."""
    is_real = dataset_tag(cfg) == "eehr"
    is_mocap = is_real and getattr(cfg, "annotation_source", "mano") == "mocap"

    if yolo_model is not None:
        yolo_model.eval()
    hand_net.eval()

    test_dataset, DatasetClass = create_test_dataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=False,
    )

    # Real dataset is additionally evaluated per light/dark scene category
    if is_real:
        light_dark_mapping = load_light_dark_mapping(cfg.real_dataset_root)
        categories = ["all", "light", "dark"]
    else:
        light_dark_mapping = {}
        categories = ["all"]

    num_steps = 100  # AUC thresholds: 0mm to 100mm
    accum = {cat: create_metrics_accumulator(num_steps) for cat in categories}

    with torch.no_grad():
        for batch in tqdm(
            test_loader, total=len(test_loader), desc="Evaluation", unit="batch"
        ):
            batch = DatasetClass.to_device(batch, device)

            yolo_results = get_yolo_results(
                batch, yolo_model, cfg, device, cfg.test.batch_size
            )

            lnes = batch["lnes"].to(device)
            outputs = hand_net(lnes, yolo_results)

            gt_joints = batch["gt_joints"]
            batch_size = lnes.shape[0]

            if is_real:
                sequence_names = batch["sequence_name"]
                item_categories = [
                    light_dark_mapping.get(name, "unknown") for name in sequence_names
                ]
            else:
                item_categories = ["all"] * batch_size

            left_valid = outputs.get(
                "left_valid", torch.ones(batch_size, device=device)
            )
            right_valid = outputs.get(
                "right_valid", torch.ones(batch_size, device=device)
            )

            valid_indices = []
            for i in range(batch_size):
                if left_valid[i] > 0.5 or right_valid[i] > 0.5:
                    valid_indices.append(i)
            valid_indices_set = set(valid_indices)

            # PCK j3d_gts
            left_gt_joints = gt_joints["left"]["joints_3d"]
            right_gt_joints = gt_joints["right"]["joints_3d"]
            j3d_gts = torch.cat(
                [left_gt_joints.unsqueeze(1), right_gt_joints.unsqueeze(1)], dim=1
            )

            # Accumulate metrics per category
            for cat in categories:
                if cat == "all":
                    cat_indices = list(range(batch_size))
                else:
                    cat_indices = [
                        i for i, c in enumerate(item_categories) if c == cat
                    ]
                if len(cat_indices) == 0:
                    continue

                # MPJPE
                cat_valid_indices = [i for i in cat_indices if i in valid_indices_set]
                accumulate_mpjpe(
                    accum[cat], outputs, gt_joints, cat_valid_indices, device, is_mocap
                )

                # PCK / AUC
                accumulate_pck(
                    accum[cat],
                    outputs,
                    j3d_gts,
                    cat_indices,
                    left_valid,
                    right_valid,
                    device,
                    num_steps,
                    is_mocap,
                )

    # Compute and print metrics per category
    all_metrics = {}
    for cat in categories:
        metrics = compute_and_print_metrics(accum[cat], cat.upper())
        prefixed = {f"{cat}/{k}": v for k, v in metrics.items()}
        all_metrics.update(prefixed)

    wandb.log(all_metrics)


def main(cfg: HandEstimationConfig):
    # Seed everything for reproducibility
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v2_test_{dataset_tag(cfg)}_{cfg.model}_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    wandb_log.config.update(OmegaConf.to_container(cfg))

    device_first = cfg.devices[0]
    device = torch.device(
        f"cuda:{device_first}" if torch.cuda.is_available() else "cpu"
    )

    # ------------------ YOLO segmentation model ------------------ #
    use_gt_mask = getattr(cfg, "use_gt_mask", False)
    yolo_model = None
    if not use_gt_mask:
        print(f"Loading YOLO model from: {cfg.yolo_checkpoint_path}")
        yolo_model = YOLO(cfg.yolo_checkpoint_path)
        yolo_model.to(device)
        print("YOLO model loaded successfully")
    else:
        print("use_gt_mask=True: skipping YOLO model loading")

    # ------------------ Hand estimation network (Mask Attention V2) ------------------ #
    hand_net = get_hand_model_v2(cfg, device)

    checkpoint_path = cfg.test.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = hand_net.load_state_dict(
        new_state_dict, strict=False
    )
    if missing_keys:
        print(f"Missing keys (new attention modules): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    print(f"Loading hand model from: {checkpoint_path}")

    test(yolo_model, hand_net, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
