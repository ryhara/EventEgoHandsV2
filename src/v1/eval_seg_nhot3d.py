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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from datasets.nhot3d_dataset import NHOT3DDataset
from utils.segmentation_utils import dilation_masks
from utils.segmentation_utils import compute_pixel_metrics
from models.UNet import load_unet

log = logging.getLogger(os.path.basename(__file__))


class SegEvalDataset(Dataset):
    """Wrapper that uses NHOT3DDataset with mode='' to avoid
    mask_event_cloud_one, and manually loads GT masks for segmentation eval."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.base.mode = ""

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]

        # Load masks manually (same logic as NHOT3DDataset mask mode)
        mask_file_left = self.base.mask_files[idx * 2]
        mask_file_right = self.base.mask_files[idx * 2 + 1]
        mask_left = self.base.load_mask(mask_file_left)
        mask_right = self.base.load_mask(mask_file_right)
        mix_mask = self.base.mix_mask(mask_left, mask_right)
        mix_mask = mix_mask / 255.0
        mix_mask = np.clip(mix_mask, 0, 1)
        data["mask"] = torch.tensor(mix_mask, dtype=torch.float32).unsqueeze(0)

        return data


def collect_per_image_results(
    probs: torch.Tensor,
    pred_binary: torch.Tensor,
    gt: torch.Tensor,
    min_area: int = 1,
) -> list:
    """Collect per-image confidence, IoU, and gt/pred existence flags.

    UNet binary-segmentation variant: takes the (possibly dilated) binary
    prediction for IoU/existence and the raw sigmoid probabilities for the
    confidence score.

    Args:
        probs: Sigmoid probabilities, shape [B, 1, H, W].
        pred_binary: Final binary prediction mask (post-threshold, optionally
            dilated), shape [B, 1, H, W].
        gt: Ground truth binary mask, shape [B, 1, H, W].
        min_area: Minimum pixel count to consider a prediction as existing.

    Returns:
        List of dicts with 'confidence', 'iou', 'pred_exists', 'gt_exists'.
    """
    batch_size = probs.shape[0]
    results = []

    for i in range(batch_size):
        pred_mask = pred_binary[i]
        gt_mask = gt[i]

        pred_area = pred_mask.sum().item()
        gt_area = gt_mask.sum().item()
        pred_exists = pred_area >= min_area
        gt_exists = gt_area > 0

        if pred_exists:
            confidence = probs[i][pred_mask > 0].mean().item()
            inter = (pred_mask * gt_mask).sum().item()
            uni = ((pred_mask + gt_mask) > 0).float().sum().item()
            img_iou = inter / (uni + 1e-6)
        else:
            confidence = 0.0
            img_iou = 0.0

        results.append({
            "confidence": confidence,
            "iou": img_iou,
            "pred_exists": pred_exists,
            "gt_exists": gt_exists,
        })

    return results


def compute_ap(
    confidences: np.ndarray,
    ious: np.ndarray,
    gt_exists_for_preds: np.ndarray,
    num_gt: int,
    iou_threshold: float,
) -> float:
    """YOLO-style Average Precision at a given IoU threshold.

    TP = prediction with matching GT and IoU >= threshold.
    FP = prediction with no GT, or IoU below threshold.
    Recall denominator is the number of GT instances (not predictions).
    """
    if len(confidences) == 0 or num_gt == 0:
        return 0.0

    tp = (gt_exists_for_preds & (ious >= iou_threshold)).astype(np.float64)
    fp = 1.0 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / num_gt

    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def compute_map(per_image_results: list) -> dict:
    """YOLO-style mAP at IoU thresholds 0.50..0.95 (step 0.05).

    Recall denominator is the number of GT instances (images with gt_exists).
    Predictions are restricted to images with pred_exists=True.
    """
    thresholds = np.arange(0.50, 0.95 + 0.01, 0.05)
    ap_dict = {}

    if not per_image_results:
        return ap_dict

    num_gt = sum(r["gt_exists"] for r in per_image_results)
    pred_only = [r for r in per_image_results if r["pred_exists"]]

    if not pred_only or num_gt == 0:
        for thresh in thresholds:
            ap_dict[f"mAP{int(round(thresh, 2) * 100)}"] = 0.0
        ap_dict["mAP50-95"] = 0.0
        ap_dict["num_gt"] = int(num_gt)
        ap_dict["num_pred"] = len(pred_only)
        return ap_dict

    confidences = np.array([r["confidence"] for r in pred_only])
    ious = np.array([r["iou"] for r in pred_only])
    gt_flags = np.array([r["gt_exists"] for r in pred_only], dtype=bool)

    sorted_idx = np.argsort(-confidences)
    confidences = confidences[sorted_idx]
    ious = ious[sorted_idx]
    gt_flags = gt_flags[sorted_idx]

    ap_values = []
    for thresh in thresholds:
        thresh_val = round(thresh, 2)
        key = f"mAP{int(thresh_val * 100)}"
        ap = compute_ap(confidences, ious, gt_flags, num_gt, thresh_val)
        ap_dict[key] = ap
        ap_values.append(ap)

    ap_dict["mAP50-95"] = float(np.mean(ap_values))
    ap_dict["num_gt"] = int(num_gt)
    ap_dict["num_pred"] = len(pred_only)
    return ap_dict


def recursive_collate_fn(key, values):
    if key == "events":
        events = [ev.clone().detach().permute(1, 0) for ev in values]
        padded_events = torch.nn.utils.rnn.pad_sequence(
            events, batch_first=True, padding_value=0
        )
        return padded_events
    elif isinstance(values[0], dict):
        return {
            k: recursive_collate_fn(k, [v[k] for v in values]) for k in values[0].keys()
        }
    elif isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    else:
        return values


def collate_fn(batch):
    return {
        key: recursive_collate_fn(key, [b[key] for b in batch])
        for key in batch[0].keys()
    }


def create_test_dataset(cfg):
    """Create NHOT3DDataset wrapped for segmentation eval."""
    base_dataset = NHOT3DDataset(
        event_input_dir=cfg.test.event_input_dir,
        gt_file_dir=cfg.test.gt_file_dir,
        object_library_path=cfg.test.object_library_path,
        mano_hand_model_path=cfg.mano_hand_model_path,
        is_get_image=cfg.test.is_get_image,
        augmentation=False,
        is_normalize=False,
        output_size=(cfg.output_height, cfg.output_width),
        mode=cfg.dataset_mode,
        event_number_threshold=cfg.event_number_threshold,
        time_bin=cfg.time_bin,
        is_sequence=cfg.is_sequence,
        dataset_divide=cfg.dataset_divide if hasattr(cfg, "dataset_divide") else 1,
    )
    return SegEvalDataset(base_dataset)


def eval_seg_nhot3d(net, cfg, device):
    """Evaluate UNet binary segmentation model on HOT3D test set."""
    print("=== U-Net Binary Segmentation Evaluation (HOT3D) ===")
    print(f"n_channels: {cfg.n_channels}, n_classes: 1 (binary)")
    print(f"checkpoint: {cfg.test.checkpoint_path}")

    test_dataset = create_test_dataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    net.eval()

    # Accumulators for pixel metrics
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0

    # Per-image results for mAP
    all_per_image_results = []

    with torch.no_grad():
        for _, batch in enumerate(
            tqdm(test_loader, total=len(test_loader), desc="Evaluating")
        ):
            lnes = batch["LNES"].to(device)
            gt_masks = batch["mask"].to(device)  # [B, 1, H, W], float32

            logits = net(lnes)  # [B, 1, H, W]

            # Sigmoid + threshold + dilation
            probs = torch.sigmoid(logits)
            binary_preds = (probs > cfg.sigmoid_threshold).float()
            binary_preds = dilation_masks(
                binary_preds,
                kernel_size=cfg.kernel_size,
                iterations=cfg.itterations,
            )

            # Pixel-level metrics
            metrics = compute_pixel_metrics(binary_preds, gt_masks)
            total_iou += metrics["iou"]
            total_dice += metrics["dice"]
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_f1 += metrics["f1"]
            num_batches += 1

            # Per-image results for mAP (YOLO-style)
            per_image = collect_per_image_results(probs, binary_preds, gt_masks)
            all_per_image_results.extend(per_image)

            # Log per-batch metrics
            wandb.log(
                {
                    "eval/iou": metrics["iou"],
                    "eval/dice": metrics["dice"],
                    "eval/precision": metrics["precision"],
                    "eval/recall": metrics["recall"],
                    "eval/f1": metrics["f1"],
                }
            )

    # Compute averages
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches

    print("\n=== Pixel-level Metrics ===")
    print(f"  IoU:       {avg_iou:.4f}")
    print(f"  Dice:      {avg_dice:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1:        {avg_f1:.4f}")

    # Compute mAP
    map_results = compute_map(all_per_image_results)

    diagnostic_keys = {"num_gt", "num_pred"}
    num_gt = map_results.get("num_gt", 0)
    num_pred = map_results.get("num_pred", 0)
    print("\n=== mAP Metrics (YOLO-style) ===")
    print(f"  num_gt:   {num_gt}")
    print(f"  num_pred: {num_pred}")
    print(f"  mAP50:    {map_results.get('mAP50', 0.0):.4f}")
    print(f"  mAP50-95: {map_results.get('mAP50-95', 0.0):.4f}")
    for key in sorted(map_results.keys()):
        if key in ("mAP50", "mAP50-95") or key in diagnostic_keys:
            continue
        print(f"  {key}:    {map_results[key]:.4f}")

    # Log summary to wandb
    summary = {
        "eval/avg_iou": avg_iou,
        "eval/avg_dice": avg_dice,
        "eval/avg_precision": avg_precision,
        "eval/avg_recall": avg_recall,
        "eval/avg_f1": avg_f1,
    }
    for key, val in map_results.items():
        summary[f"eval/{key}"] = val

    wandb.log(summary)
    for key, val in summary.items():
        wandb.run.summary[key] = val

    print("\nEvaluation complete.")


def main(cfg) -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v1_eval_seg_nhot3d_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(
        f"cuda:{device_first}" if torch.cuda.is_available() else "cpu"
    )

    if not cfg.test.checkpoint_path or not os.path.exists(cfg.test.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {cfg.test.checkpoint_path}. "
            "Set cfg.test.checkpoint_path to a valid model path."
        )

    net = load_unet(
        cfg.test.checkpoint_path,
        n_channels=cfg.n_channels,
        n_classes=1,
        device=device,
    )
    print(f"Loaded model from {cfg.test.checkpoint_path}")

    eval_seg_nhot3d(net, cfg, device)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_test_seg_nhot3d.yaml"),
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
