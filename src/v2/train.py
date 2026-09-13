"""Train EventEgoHandsV2 on N-HOT3D (synthetic) or EEH-R (real).

The dataset is selected by `dataset_type` in the config file:
    python train.py --config config/config_nhot3d.yaml
    python train.py --config config/config_eehr.yaml
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from datasets.eehr_dataset import EEHRDataset
from datasets.nhot3d_yolo_dataset import NHOT3DYOLODataset
from factory import get_hand_model_v2
from models.CustomLoss import CustomLoss
from utils import HandEstimationConfig
from v2.detection_utils import get_yolo_results

log = logging.getLogger(os.path.basename(__file__))


def dataset_tag(cfg) -> str:
    return "eehr" if getattr(cfg, "dataset_type", "synth") == "real" else "nhot3d"


def create_dataset(cfg, split: str):
    """Create dataset based on dataset_type in config.

    Args:
        cfg: Configuration object
        split: "train" or "valid"

    Returns:
        Dataset instance (NHOT3DYOLODataset or EEHRDataset)
    """
    if getattr(cfg, "dataset_type", "synth") == "real":
        return EEHRDataset(
            dataset_root=cfg.real_dataset_root,
            split=split,
            fps=getattr(cfg, "real_dataset_fps", 30),
            output_size=(cfg.output_height, cfg.output_width),
            use_center_crop=getattr(cfg, "use_center_crop", False),
            ignore_files_count=cfg.ignore_files_count,
            dataset_divide=cfg.train.dataset_divide,
            annotation_source=getattr(cfg, "annotation_source", "mano"),
            annotation_fps=getattr(cfg, "annotation_fps", 30),
            lnes_from_h5=getattr(cfg, "lnes_from_h5", False),
            lnes_window_sec=getattr(cfg, "lnes_window_sec", 1.0 / 30.0),
            use_gt_mask=getattr(cfg, "use_gt_mask", False),
            gt_mask_root=getattr(cfg, "gt_mask_root", None),
            allow_subframe_gt_mask=getattr(cfg, "allow_subframe_gt_mask", False),
        )
    else:
        return NHOT3DYOLODataset(
            yolo_image_dir=cfg.train.yolo_image_dir + "/" + split,
            gt_file_dir=cfg.train.gt_file_dir + "/" + split,
            object_library_path=cfg.train.object_library_path,
            mano_hand_model_path=cfg.mano_hand_model_path,
            is_get_image=cfg.train.is_get_image,
            rgb_dir=cfg.train.rgb_dir,
            mode=cfg.dataset_mode,
            output_size=(cfg.output_height, cfg.output_width),
            use_center_crop=getattr(cfg, "use_center_crop", True),
            augmentation=getattr(cfg.train, "augmentation", False),
            ignore_files_count=cfg.ignore_files_count if split == "valid" else 0,
            dataset_divide=cfg.train.dataset_divide,
            lnes_blosc2_dir=getattr(cfg.train, "lnes_blosc2_dir", "") + "/" + split,
        )


def train(yolo_model, hand_net, cfg: HandEstimationConfig, device):
    epochs = cfg.train.epochs
    tag = dataset_tag(cfg)

    print(f"Using dataset type: {getattr(cfg, 'dataset_type', 'synth')}")
    if tag == "eehr":
        print(f"annotation_source: {getattr(cfg, 'annotation_source', 'mano')}")
        print(f"annotation_fps: {getattr(cfg, 'annotation_fps', 30)}")
        print(f"use_gt_mask: {getattr(cfg, 'use_gt_mask', False)}")

    train_dataset = create_dataset(cfg, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=False,
    )

    validation_dataset = create_dataset(cfg, "valid")
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=False,
    )

    DatasetClass = EEHRDataset if tag == "eehr" else NHOT3DYOLODataset

    optimizer = optim.AdamW(
        hand_net.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = None
    if getattr(cfg.train, "use_scheduler", False):
        n_iter_per_epoch = len(train_loader)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epochs * n_iter_per_epoch,
            lr_min=getattr(cfg.train, "min_lr", 1e-7),
            warmup_t=getattr(cfg.train, "warmup_epochs", 1) * n_iter_per_epoch,
            warmup_lr_init=getattr(cfg.train, "warmup_lr", 1e-7),
            warmup_prefix=getattr(cfg.train, "warmup_prefix", True),
            cycle_limit=getattr(cfg.train, "cycle_limit", 1),
            t_in_epochs=False,
        )

    criterion = CustomLoss(
        device=device,
        cfg=cfg,
        hand_pose_dim=cfg.mano_cmps,
        mano_hand_model_path=cfg.mano_hand_model_path,
        crop_size=(cfg.output_height, cfg.output_width),
    )

    for epoch in range(epochs):
        train_loss = 0
        hand_net.train()
        num_updates = epoch * len(train_loader)

        for index, batch in enumerate(
            tqdm(train_loader, total=len(train_loader), desc=f"Train Epoch {epoch + 1}/{epochs}")
        ):
            batch = DatasetClass.to_device(batch, device)

            with torch.no_grad():
                yolo_results = get_yolo_results(
                    batch, yolo_model, cfg, device, cfg.train.batch_size
                )

            lnes = batch["lnes"].to(device)
            outputs = hand_net(lnes, yolo_results)

            batch["gt_hand_poses"]["left"]["valid"] = outputs["left_valid"]
            batch["gt_hand_poses"]["right"]["valid"] = outputs["right_valid"]

            losses = criterion(outputs, batch)
            loss = sum(losses.values())
            losses["loss"] = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                num_updates += 1
                scheduler.step_update(num_updates)

            train_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

            if index % cfg.train.log_interval == 0:
                log_dict = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in losses.items()
                }
                log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
                wandb.log(log_dict)

        train_loss /= len(train_loader)
        wandb.log({"train_loss_mean": train_loss})

        hand_net.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                total=len(validation_loader),
                desc=f"Validation Epoch {epoch + 1}/{epochs}",
                unit="batch",
            ):
                batch = DatasetClass.to_device(batch, device)
                yolo_results = get_yolo_results(
                    batch, yolo_model, cfg, device, cfg.train.batch_size
                )

                lnes = batch["lnes"].to(device)
                outputs = hand_net(lnes, yolo_results)

                batch["gt_hand_poses"]["left"]["valid"] = outputs["left_valid"]
                batch["gt_hand_poses"]["right"]["valid"] = outputs["right_valid"]

                losses = criterion(outputs, batch)
                loss = sum(losses.values())
                val_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

        val_loss /= len(validation_loader)
        wandb.log({"val_loss_mean": val_loss}, commit=False)

        if (epoch + 1) % cfg.train.save_interval == 0:
            date_str = datetime.now().strftime("%Y-%m-%d/")
            save_dir = os.path.join(cfg.train.save_path, date_str)
            os.makedirs(save_dir, exist_ok=True)
            time_str = datetime.now().strftime("%H-%M-%S")
            save_path = os.path.join(
                save_dir,
                f"v2_{tag}_{epoch + 1}_{time_str}_val_loss_{val_loss:.4f}.pth",
            )
            print(f"Save model to {save_path}")
            torch.save(hand_net.state_dict(), save_path)


def main(cfg: HandEstimationConfig) -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v2_train_{dataset_tag(cfg)}_{cfg.model}_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(f"cuda:{device_first}" if torch.cuda.is_available() else "cpu")

    use_gt_mask = getattr(cfg, "use_gt_mask", False)
    yolo_model = None
    if not use_gt_mask:
        print(f"Loading YOLO model from: {cfg.yolo_checkpoint_path}")
        yolo_model = YOLO(cfg.yolo_checkpoint_path)
        yolo_model.to(device)
        print("YOLO model loaded successfully")
    else:
        print("use_gt_mask=True: skipping YOLO model loading")

    hand_net = get_hand_model_v2(cfg, device)

    log.info("Attention settings (V2):")
    log.info(f"  use_cross_attention: {getattr(cfg, 'use_cross_attention', True)}")
    log.info(f"  use_self_attention: {getattr(cfg, 'use_self_attention', True)}")
    log.info(f"  attention_num_heads: {getattr(cfg, 'attention_num_heads', 4)}")
    log.info(f"  attention_dropout: {getattr(cfg, 'attention_dropout', 0.1)}")
    log.info(f"  use_positional_encoding: {getattr(cfg, 'use_positional_encoding', True)}")
    log.info(f"  num_attention_layers: {getattr(cfg, 'num_attention_layers', 1)}")
    log.info(f"  use_learnable_pos_encoding: {getattr(cfg, 'use_learnable_pos_encoding', True)}")
    log.info(f"  pos_encoding_base_size: {getattr(cfg, 'pos_encoding_base_size', 14)}")
    log.info(f"  add_pos_encoding_every_layer: {getattr(cfg, 'add_pos_encoding_every_layer', True)}")
    log.info(f"  use_shared_backbone: {getattr(cfg, 'use_shared_backbone', False)}")
    log.info(f"  attention_order: {getattr(cfg, 'attention_order', 'self_first')}")
    log.info(f"  mask_dilation_kernel_size: {getattr(cfg, 'mask_dilation_kernel_size', 5)}")
    log.info(f"  mask_dilation_iterations: {getattr(cfg, 'mask_dilation_iterations', 2)}")

    log.info(
        f"Hand estimation model parameters: {sum(p.numel() for p in hand_net.parameters())}"
    )

    if os.path.exists(cfg.train.checkpoint_path):
        checkpoint = torch.load(
            cfg.train.checkpoint_path, weights_only=False, map_location=device
        )
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
        print(f"Load model from {cfg.train.checkpoint_path}")

    train(yolo_model, hand_net, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
