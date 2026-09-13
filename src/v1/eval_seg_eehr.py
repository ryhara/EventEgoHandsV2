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

from v1.train_seg_eehr import (
    create_dataset,
    generate_gt_masks,
)
from models.UNet import load_unet
from utils.segmentation_utils import compute_pixel_metrics
from datasets.eehr_dataset import load_light_dark_mapping

log = logging.getLogger(os.path.basename(__file__))


def collect_per_image_results(
    pred: torch.Tensor, gt: torch.Tensor, min_area: int = 1
) -> list:
    """Collect per-image confidence, IoU, and gt/pred existence flags.

    Args:
        pred: Predicted logits, shape [B, 1, H, W].
        gt: Ground truth binary mask, shape [B, 1, H, W].
        min_area: Minimum pixel count to consider a prediction as existing.

    Returns:
        List of dicts with 'confidence', 'iou', 'pred_exists', 'gt_exists'.
    """
    probs = torch.sigmoid(pred)
    pred_binary = (probs > 0.5).float()
    batch_size = pred.shape[0]
    results = []

    for i in range(batch_size):
        pred_mask = pred_binary[i]  # [1, H, W]
        gt_mask = gt[i]  # [1, H, W]

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
    """Compute YOLO-style Average Precision at a given IoU threshold.

    TP = prediction with matching GT and IoU >= threshold.
    FP = prediction with no GT, or IoU below threshold.
    Recall denominator is the number of GT instances (not predictions).

    Args:
        confidences: Confidence scores for predictions, sorted descending.
        ious: IoU values for each prediction.
        gt_exists_for_preds: Boolean array indicating GT existence per prediction.
        num_gt: Total number of GT instances.
        iou_threshold: IoU threshold for TP/FP decision.

    Returns:
        AP value.
    """
    if len(confidences) == 0 or num_gt == 0:
        return 0.0

    tp = (gt_exists_for_preds & (ious >= iou_threshold)).astype(np.float64)
    fp = 1.0 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / num_gt

    # Prepend sentinel values
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])

    # Make precision monotonically decreasing (right to left)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AUC using trapezoidal integration
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def compute_map(per_image_results: list) -> dict:
    """Compute YOLO-style mAP at various IoU thresholds.

    Recall denominator is the number of GT instances (images with gt_exists).
    Predictions are restricted to images with pred_exists=True.

    Args:
        per_image_results: List of dicts with 'confidence', 'iou',
            'pred_exists', 'gt_exists'.

    Returns:
        Dict with mAP50, mAP55, ..., mAP90, mAP50-95, num_gt, num_pred.
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

    # Sort by confidence descending
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


def _run_inference_loop(
    net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    light_dark_mapping: dict,
    categories: list,
    target_cats: set,
    desc: str,
    log_pixel_metrics: bool,
) -> dict:
    """Iterate a DataLoader, accumulate per-image results filtered by category.

    Args:
        target_cats: Only frames whose light/dark category is in this set are
            collected. Use {"all", "light", "dark"} to keep everything.
        log_pixel_metrics: If True, accumulate pixel-level metrics across the
            whole loop and log per-batch values to wandb.

    Returns:
        Dict with 'per_image_by_cat' and (if log_pixel_metrics) 'pixel_metrics'.
    """
    per_image_results_by_cat = {cat: [] for cat in categories}
    total_iou = total_dice = total_precision = total_recall = total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=desc):
            lnes = batch["lnes"].to(device)
            gt_masks = generate_gt_masks(batch, device)
            logits = net(lnes)

            if log_pixel_metrics:
                metrics = compute_pixel_metrics(logits, gt_masks)
                total_iou += metrics["iou"]
                total_dice += metrics["dice"]
                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_f1 += metrics["f1"]
                num_batches += 1
                wandb.log({
                    "eval/iou": metrics["iou"],
                    "eval/dice": metrics["dice"],
                    "eval/precision": metrics["precision"],
                    "eval/recall": metrics["recall"],
                    "eval/f1": metrics["f1"],
                })

            per_image = collect_per_image_results(logits, gt_masks)
            sequence_names = batch["sequence_name"]
            item_categories = [
                light_dark_mapping.get(name, "unknown") for name in sequence_names
            ]
            for item, cat in zip(per_image, item_categories):
                if "all" in target_cats:
                    per_image_results_by_cat["all"].append(item)
                if cat in target_cats and cat in per_image_results_by_cat:
                    per_image_results_by_cat[cat].append(item)

    result = {"per_image_by_cat": per_image_results_by_cat}
    if log_pixel_metrics and num_batches > 0:
        result["pixel_metrics"] = {
            "iou": total_iou / num_batches,
            "dice": total_dice / num_batches,
            "precision": total_precision / num_batches,
            "recall": total_recall / num_batches,
            "f1": total_f1 / num_batches,
        }
    return result


def eval_seg_eehr(net: torch.nn.Module, cfg, device: torch.device):
    """Evaluate UNet segmentation model on the test split.

    Both light and dark frames are collected from the test split.
    """
    print("=== U-Net Binary Segmentation Evaluation (Real) ===")
    print(f"n_channels: {cfg.n_channels}, n_classes: {cfg.n_classes}")
    print(f"checkpoint: {cfg.train.checkpoint_path}")

    test_dataset = create_dataset(cfg, "test")
    test_dataset.dataset_divide = 1
    test_loader = DataLoader(
        test_dataset,
        batch_size=getattr(cfg, "test", {}).get("batch_size", cfg.train.batch_size),
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=False,
    )

    light_dark_mapping = load_light_dark_mapping(cfg.real_dataset_root)
    categories = ["all", "light", "dark"]

    net.eval()

    test_result = _run_inference_loop(
        net,
        test_loader,
        device,
        light_dark_mapping,
        categories,
        target_cats={"all", "light", "dark"},
        desc="Evaluating (test)",
        log_pixel_metrics=True,
    )

    per_image_results_by_cat = test_result["per_image_by_cat"]
    source_split = {cat: "test" for cat in categories}

    pixel = test_result.get("pixel_metrics", {})
    print("\n=== Pixel-level Metrics (test split) ===")
    print(f"  IoU:       {pixel.get('iou', 0.0):.4f}")
    print(f"  Dice:      {pixel.get('dice', 0.0):.4f}")
    print(f"  Precision: {pixel.get('precision', 0.0):.4f}")
    print(f"  Recall:    {pixel.get('recall', 0.0):.4f}")
    print(f"  F1:        {pixel.get('f1', 0.0):.4f}")

    map_results_by_cat = {
        cat: compute_map(per_image_results_by_cat[cat]) for cat in categories
    }

    print("\n=== mAP Metrics (YOLO-style) per light/dark category ===")
    diagnostic_keys = {"num_gt", "num_pred"}
    for cat in categories:
        n_items = len(per_image_results_by_cat[cat])
        cat_results = map_results_by_cat[cat]
        num_gt = cat_results.get("num_gt", 0)
        num_pred = cat_results.get("num_pred", 0)
        print(
            f"\n[{cat}] split={source_split[cat]} "
            f"(n={n_items}, gt={num_gt}, pred={num_pred})"
        )
        if n_items == 0:
            print("  (no samples)")
            continue
        print(f"  mAP50:    {cat_results.get('mAP50', 0.0):.4f}")
        print(f"  mAP50-95: {cat_results.get('mAP50-95', 0.0):.4f}")
        for key in sorted(cat_results.keys()):
            if key in ("mAP50", "mAP50-95") or key in diagnostic_keys:
                continue
            print(f"  {key}:    {cat_results[key]:.4f}")

    summary = {
        "eval/avg_iou": pixel.get("iou", 0.0),
        "eval/avg_dice": pixel.get("dice", 0.0),
        "eval/avg_precision": pixel.get("precision", 0.0),
        "eval/avg_recall": pixel.get("recall", 0.0),
        "eval/avg_f1": pixel.get("f1", 0.0),
    }
    for cat in categories:
        summary[f"eval/{cat}/num_samples"] = len(per_image_results_by_cat[cat])
        summary[f"eval/{cat}/source_split"] = source_split[cat]
        for key, val in map_results_by_cat[cat].items():
            summary[f"eval/{cat}/{key}"] = val

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
    name = f"v1_eval_seg_eehr_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(f"cuda:{device_first}" if torch.cuda.is_available() else "cpu")

    # Prefer test.checkpoint_path; train.checkpoint_path is only a resume slot
    checkpoint_path = getattr(cfg, "test", {}).get("checkpoint_path", "") or cfg.train.checkpoint_path
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Set cfg.test.checkpoint_path to a valid model path."
        )

    net = load_unet(
        checkpoint_path,
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        device=device,
    )
    print(f"Loaded model from {checkpoint_path}")

    eval_seg_eehr(net, cfg, device)
    wandb.finish()


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
