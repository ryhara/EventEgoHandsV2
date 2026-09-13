import argparse
import logging
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from datasets.eehr_dataset import EEHRDataset
from models.UNet import UNet

log = logging.getLogger(os.path.basename(__file__))


def binary_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute binary Dice loss.

    Args:
        pred: Predicted probabilities after sigmoid, shape [B, 1, H, W].
        target: Ground truth binary mask, shape [B, 1, H, W].
        smooth: Smoothing factor to avoid division by zero.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


class MaskLoss(nn.Module):
    """BCE + Dice combined loss for binary segmentation."""

    def __init__(self, mode="", alpha=0.3):
        super().__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> dict:
        bce_loss = self.bce_fn(input, target)
        dice_loss = binary_dice_loss(torch.sigmoid(input), target)
        total = (1 - self.alpha) * bce_loss + self.alpha * dice_loss
        return {"bce_loss": bce_loss, "dice_loss": dice_loss, "loss": total}


def parse_gt_yolo_label(label_path: str, width: int, height: int):
    """Parse YOLO polygon label and return list of detections."""
    if not label_path:
        return []
    if not os.path.exists(label_path):
        return []

    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            polygon = []
            for i in range(0, len(coords), 2):
                x = int(max(0, min(width - 1, round(coords[i] * width))))
                y = int(max(0, min(height - 1, round(coords[i + 1] * height))))
                polygon.append((x, y))

            if len(polygon) < 3:
                continue

            polygon_np = np.array(polygon, dtype=np.int32)
            detections.append(
                {
                    "class_id": class_id,
                    "polygon": polygon_np,
                }
            )

    return detections


def yolo_label_to_binary_mask(label_path: str, width: int, height: int) -> np.ndarray:
    """Convert YOLO polygon label to a binary mask.

    All hand classes are merged into a single binary mask (hand=1, background=0).

    Args:
        label_path: Path to YOLO label file.
        width: Image width.
        height: Image height.

    Returns:
        Binary mask of shape [H, W] as float32.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    detections = parse_gt_yolo_label(label_path, width, height)
    for det in detections:
        cv2.fillPoly(mask, [det["polygon"]], 1.0)
    return mask


def generate_gt_masks(batch: dict, device: torch.device) -> torch.Tensor:
    """Generate GT binary masks for a batch.

    Args:
        batch: Batch dict containing 'lnes' and 'gt_yolo_label_path'.
        device: Target device.

    Returns:
        Tensor of shape [B, 1, H, W] as float32.
    """
    lnes = batch["lnes"]
    batch_size = lnes.shape[0]
    height, width = lnes.shape[2], lnes.shape[3]
    label_paths = batch.get("gt_yolo_label_path", [""] * batch_size)

    masks = []
    for i in range(batch_size):
        mask = yolo_label_to_binary_mask(label_paths[i], width, height)
        masks.append(mask)

    masks_tensor = torch.tensor(np.stack(masks), dtype=torch.float32, device=device)
    return masks_tensor.unsqueeze(1)  # [B, 1, H, W]


def compute_iou(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute binary IoU between predicted and ground truth masks.

    Args:
        pred: Predicted logits, shape [B, 1, H, W].
        gt: Ground truth binary mask, shape [B, 1, H, W].
        threshold: Threshold for binarizing predictions.

    Returns:
        Mean IoU over the batch.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * gt).sum(dim=(1, 2, 3))
    union = ((pred_binary + gt) > 0).float().sum(dim=(1, 2, 3))
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def create_dataset(cfg, split: str):
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


def train_seg_eehr(net: nn.Module, cfg, device: torch.device):
    epochs = cfg.train.epochs

    print("=== U-Net Binary Segmentation Training (Real) ===")
    print(f"n_channels: {cfg.n_channels}, n_classes: {cfg.n_classes}")
    print(f"annotation_source: {getattr(cfg, 'annotation_source', 'mano')}")
    print(f"use_gt_mask: {getattr(cfg, 'use_gt_mask', False)}")
    print(f"gt_mask_root: {getattr(cfg, 'gt_mask_root', None)}")
    print(f"alpha (Dice weight): {getattr(cfg.train, 'alpha', 0.3)}")

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

    optimizer = optim.AdamW(
        net.parameters(),
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

    criterion = MaskLoss(mode="", alpha=getattr(cfg.train, "alpha", 0.3))

    for epoch in range(epochs):
        # --- Train ---
        net.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0
        num_updates = epoch * len(train_loader)

        for index, batch in enumerate(
            tqdm(train_loader, total=len(train_loader), desc=f"Train Epoch {epoch + 1}/{epochs}")
        ):
            lnes = batch["lnes"].to(device)
            gt_masks = generate_gt_masks(batch, device)

            logits = net(lnes)  # [B, 1, H, W]
            losses = criterion(logits, gt_masks)
            loss = losses["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                num_updates += 1
                scheduler.step_update(num_updates)

            train_loss_sum += loss.item()

            with torch.no_grad():
                batch_iou = compute_iou(logits, gt_masks)
                train_iou_sum += batch_iou

            if index % cfg.train.log_interval == 0:
                log_dict = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in losses.items()
                }
                log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
                log_dict["train_iou"] = batch_iou
                wandb.log(log_dict)

        train_loss_mean = train_loss_sum / len(train_loader)
        train_iou_mean = train_iou_sum / len(train_loader)
        wandb.log({
            "train_loss_mean": train_loss_mean,
            "train_iou_mean": train_iou_mean,
        })
        print(f"  Train Loss: {train_loss_mean:.4f}, Train IoU: {train_iou_mean:.4f}")

        # --- Validation ---
        net.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                total=len(validation_loader),
                desc=f"Val Epoch {epoch + 1}/{epochs}",
                unit="batch",
            ):
                lnes = batch["lnes"].to(device)
                gt_masks = generate_gt_masks(batch, device)

                logits = net(lnes)
                losses = criterion(logits, gt_masks)
                val_loss_sum += losses["loss"].item()
                val_iou_sum += compute_iou(logits, gt_masks)

        val_loss_mean = val_loss_sum / len(validation_loader)
        val_iou_mean = val_iou_sum / len(validation_loader)
        wandb.log({
            "val_loss_mean": val_loss_mean,
            "val_iou_mean": val_iou_mean,
        }, commit=False)
        print(f"  Val Loss: {val_loss_mean:.4f}, Val IoU: {val_iou_mean:.4f}")

        # --- Checkpoint ---
        if (epoch + 1) % cfg.train.save_interval == 0:
            date_str = datetime.now().strftime("%Y-%m-%d/")
            save_dir = os.path.join(cfg.train.save_path, date_str)
            os.makedirs(save_dir, exist_ok=True)
            time_str = datetime.now().strftime("%H-%M-%S")
            save_path = os.path.join(
                save_dir,
                f"v1_eehr_seg_{epoch + 1}_{time_str}_val_loss_{val_loss_mean:.4f}_iou_{val_iou_mean:.4f}.pth",
            )
            print(f"Save model to {save_path}")
            torch.save(net.state_dict(), save_path)


def main(cfg) -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v1_train_seg_eehr_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(f"cuda:{device_first}" if torch.cuda.is_available() else "cpu")

    # Dropout layers shift module indices in state_dict keys, so this must
    # match the value used when the checkpoint was trained.
    net = UNet(
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        dropout_prob=getattr(cfg, "segmentation_dropout_prob", 0.2),
    ).to(device)
    print(f"UNet created: n_channels={cfg.n_channels}, n_classes={cfg.n_classes}")

    if os.path.exists(cfg.train.checkpoint_path):
        state_dict = torch.load(
            cfg.train.checkpoint_path, weights_only=False, map_location=device
        )
        # Support both raw state_dict and wrapped checkpoint
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        print(f"Load model from {cfg.train.checkpoint_path}")

    train_seg_eehr(net, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_train_seg_eehr.yaml"),
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
