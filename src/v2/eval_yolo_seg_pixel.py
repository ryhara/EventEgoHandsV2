"""Evaluate the YOLO hand segmentation model with pixel-level metrics.

Reports IoU / Dice / Precision / Recall / F1 over the predicted masks, which is
directly comparable with the v1 U-Net evaluation (`src/v1/eval_seg_*.py`).
The dataset is picked with `--dataset`, the remaining settings are
module-level constants:
    python src/v2/eval_yolo_seg_pixel.py --dataset real     # EEH-R
    python src/v2/eval_yolo_seg_pixel.py --dataset synth    # N-HOT3D

See `eval_yolo_seg.py` for COCO-style mAP metrics.
"""

import argparse
import glob as glob_module
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm
from ultralytics import YOLO

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SRC_DIR)

from utils.segmentation_utils import compute_pixel_metrics

# ---------------------------------------------------------------------------
# Datasets (pick one with --dataset)
# ---------------------------------------------------------------------------
DATASETS = {
    # EEH-R (real)
    "real": {
        "name": "real",
        "weights": "/path/to/real.pt",
        "data_path": "/path/to/EEH-R/YOLO/train.yaml",
    },
    # N-HOT3D (synthetic)
    "synth": {
        "name": "synth",
        "weights": "/path/to/synth.pt",
        "data_path": "/path/to/NHOT3D_yolo_seg_lnes_frame_v2/train.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="real",
        help="which dataset preset to evaluate on (default: real)",
    )
    p.add_argument("--weights", help="override the preset's model weights")
    p.add_argument("--data", help="override the preset's YOLO data yaml")
    p.add_argument("--name", help="override the preset's run name (output directory)")
    p.add_argument(
        "--device", type=int, help=f"override the CUDA device (default: {DEVICE})"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2] / "data" / "save" / "yolo"
DEVICE = 0
IMAGE_SIZE = 640
BATCH_SIZE = 64
SEED = 0
CONF = 0.5


def set_seed(seed: int, device: int = 0) -> None:
    """Set random seeds for reproducibility on a specific CUDA device only."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)


def safe_float(value):
    """Convert various numeric types to float safely."""
    if value is None:
        return None
    if hasattr(value, "__len__"):
        length = len(value)
        if length == 0:
            return None
        return float(value[0])
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (ValueError, RuntimeError):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def collect_per_image_results(pred: torch.Tensor, gt: torch.Tensor) -> list:
    """Collect per-image confidence and IoU for mAP computation.

    Args:
        pred: Predicted binary mask, shape [B, 1, H, W].
        gt: Ground truth binary mask, shape [B, 1, H, W].

    Returns:
        List of dicts with 'confidence' and 'iou' per image.
    """
    pred_binary = pred.float()
    batch_size = pred.shape[0]
    results = []

    for i in range(batch_size):
        pred_mask = pred_binary[i]
        gt_mask = gt[i]

        # Confidence: fraction of positive pixels (proxy for YOLO binary mask)
        positive_mask = pred_mask > 0
        confidence = positive_mask.float().mean().item() if positive_mask.any() else 0.0

        inter = (pred_mask * gt_mask).sum().item()
        uni = ((pred_mask + gt_mask) > 0).float().sum().item()
        img_iou = inter / (uni + 1e-6)

        results.append({"confidence": confidence, "iou": img_iou})

    return results


def compute_ap(
    confidences: np.ndarray, ious: np.ndarray, iou_threshold: float
) -> float:
    """Compute Average Precision at a given IoU threshold."""
    n = len(confidences)
    if n == 0:
        return 0.0

    tp = (ious >= iou_threshold).astype(np.float64)
    fp = 1.0 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (n + 1e-6)

    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def compute_map(per_image_results: list) -> dict:
    """Compute mAP at various IoU thresholds (COCO-style)."""
    if not per_image_results:
        return {}

    confidences = np.array([r["confidence"] for r in per_image_results])
    ious = np.array([r["iou"] for r in per_image_results])

    sorted_idx = np.argsort(-confidences)
    confidences = confidences[sorted_idx]
    ious = ious[sorted_idx]

    thresholds = np.arange(0.50, 0.95 + 0.01, 0.05)
    ap_dict = {}
    ap_values = []

    for thresh in thresholds:
        thresh_val = round(thresh, 2)
        key = f"mAP{int(thresh_val * 100)}"
        ap = compute_ap(confidences, ious, thresh_val)
        ap_dict[key] = ap
        ap_values.append(ap)

    ap_dict["mAP50-95"] = float(np.mean(ap_values))
    return ap_dict


# ==========================================
# GT / Prediction mask helpers
# ==========================================


def parse_yolo_seg_label(label_path: str, img_h: int, img_w: int) -> np.ndarray:
    """Parse YOLO segmentation label file and return a binary mask [H, W].

    Each line: class_id x1 y1 x2 y2 ... (normalized polygon coords).
    All classes are merged into a single binary mask.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not Path(label_path).exists():
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # Skip class_id (parts[0]), rest are x y pairs
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        polygon = np.array(
            [[int(x * img_w), int(y * img_h)] for x, y in zip(xs, ys)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 1)

    return mask


def yolo_result_to_binary_mask(result, img_h: int, img_w: int) -> np.ndarray:
    """Convert YOLO prediction result to a binary mask [H, W].

    Merges all instance masks into a single binary mask.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if result.masks is None:
        return mask

    # result.masks.data: [N, mask_h, mask_w] on GPU
    masks_data = result.masks.data.cpu().numpy()  # [N, mask_h, mask_w]
    for m in masks_data:
        # Resize mask to original image size if needed
        if m.shape[0] != img_h or m.shape[1] != img_w:
            m_resized = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            m_resized = m
        mask = np.maximum(mask, (m_resized > 0.5).astype(np.uint8))

    return mask


# ==========================================
# Main
# ==========================================


def main():
    print("=== YOLO Segmentation Test Evaluation ===")

    args = parse_args()
    preset = DATASETS[args.dataset]
    name = args.name or preset["name"]
    data_path = args.data or preset["data_path"]
    if args.device is not None:
        global DEVICE
        DEVICE = args.device

    set_seed(SEED, device=DEVICE)

    # Load model
    model_path = Path(args.weights or preset["weights"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    print(f"Dataset: {args.dataset}")
    print(f"Data yaml: {data_path}")
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    wandb.init(
        project="EventEgoHands-yolo",
        name=f"test_yolo_seg_{args.dataset}",
    )
    print(f"Seed: {SEED}, Conf: {CONF}, Device: {DEVICE}, Image Size: {IMAGE_SIZE}, Batch Size: {BATCH_SIZE}")

    # ==========================================
    # Part 1: YOLO built-in evaluation (test set)
    # ==========================================
    print("\n=== Part 1: YOLO Built-in Test Set Evaluation ===")

    # The dataset path baked into the checkpoint points at wherever the model was
    # trained, which need not exist here, so always pass the yaml explicitly.
    metrics_test = model.val(
        data=data_path,
        project=str(PROJECT_DIR),
        name=f"{name}_test",
        task="segment",
        split="test",
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        conf=CONF,
        seed=SEED,
        exist_ok=True,
    )

    if metrics_test is not None:
        test_metrics_dict = {
            "precision": safe_float(metrics_test.seg.p)
            if hasattr(metrics_test.seg, "p")
            else None,
            "recall": safe_float(metrics_test.seg.r)
            if hasattr(metrics_test.seg, "r")
            else None,
            "mAP50": safe_float(metrics_test.seg.map50)
            if hasattr(metrics_test.seg, "map50")
            else None,
            "mAP50-95": safe_float(metrics_test.seg.map)
            if hasattr(metrics_test.seg, "map")
            else None,
            "mAP75": safe_float(metrics_test.seg.map75)
            if hasattr(metrics_test.seg, "map75")
            else None,
        }

        print(f"Test - Precision: {test_metrics_dict['precision']:.4f}")
        print(f"Test - Recall: {test_metrics_dict['recall']:.4f}")
        print(f"Test - mAP50: {test_metrics_dict['mAP50']:.4f}")
        print(f"Test - mAP75: {test_metrics_dict['mAP75']:.4f}")
        print(f"Test - mAP50-95: {test_metrics_dict['mAP50-95']:.4f}")

        print("\n=== YOLO Built-in Metrics (saved to test_metrics.json) ===")
        print(json.dumps(test_metrics_dict, indent=2))

        test_metrics_path = PROJECT_DIR / f"{name}_test" / "test_metrics.json"
        test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics_dict, f, indent=2)
        print(f"Test metrics saved to: {test_metrics_path}")

        wandb.log(
            {f"yolo/{k}": v for k, v in test_metrics_dict.items() if v is not None}
        )
    else:
        print("Warning: Test metrics evaluation returned None")

    # ==========================================
    # Part 2: Pixel-level evaluation (same as eval_seg_eehr.py)
    # ==========================================
    print("\n=== Part 2: Pixel-level Evaluation (IoU / Dice / mAP) ===")

    # Load dataset config to find test images and labels
    with open(data_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg["path"])
    test_img_dir = dataset_root / "images" / "test"
    test_label_dir = dataset_root / "labels" / "test"

    test_images = sorted(glob_module.glob(str(test_img_dir / "*.png")))
    if not test_images:
        test_images = sorted(glob_module.glob(str(test_img_dir / "*.jpg")))
    print(f"Found {len(test_images)} test images")

    # Accumulators
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0
    all_per_image_results = []

    # Process in batches
    for i in tqdm(range(0, len(test_images), BATCH_SIZE), desc="Pixel-level eval"):
        batch_paths = test_images[i : i + BATCH_SIZE]

        # Run YOLO prediction
        results = model.predict(
            source=batch_paths,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=CONF,
            verbose=False,
        )

        pred_masks_list = []
        gt_masks_list = []

        for img_path, result in zip(batch_paths, results):
            img_h, img_w = result.orig_shape

            # Predicted binary mask
            pred_mask = yolo_result_to_binary_mask(result, img_h, img_w)

            # GT binary mask from YOLO label
            label_path = test_label_dir / (Path(img_path).stem + ".txt")
            gt_mask = parse_yolo_seg_label(str(label_path), img_h, img_w)

            pred_masks_list.append(pred_mask)
            gt_masks_list.append(gt_mask)

        # Stack to tensors [B, 1, H, W]
        pred_tensor = torch.from_numpy(np.stack(pred_masks_list)[:, None, :, :]).float()
        gt_tensor = torch.from_numpy(np.stack(gt_masks_list)[:, None, :, :]).float()

        # Pixel-level metrics
        metrics = compute_pixel_metrics(pred_tensor, gt_tensor)
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1"]
        num_batches += 1

        # Per-image results for mAP
        per_image = collect_per_image_results(pred_tensor, gt_tensor)
        all_per_image_results.extend(per_image)

        wandb.log(
            {
                "pixel_eval/iou": metrics["iou"],
                "pixel_eval/dice": metrics["dice"],
                "pixel_eval/precision": metrics["precision"],
                "pixel_eval/recall": metrics["recall"],
                "pixel_eval/f1": metrics["f1"],
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

    print("\n=== mAP Metrics (COCO-style) ===")
    for key, val in sorted(map_results.items()):
        print(f"  {key}: {val:.4f}")

    # Log summary to wandb
    summary = {
        "pixel_eval/avg_iou": avg_iou,
        "pixel_eval/avg_dice": avg_dice,
        "pixel_eval/avg_precision": avg_precision,
        "pixel_eval/avg_recall": avg_recall,
        "pixel_eval/avg_f1": avg_f1,
    }
    for key, val in map_results.items():
        summary[f"pixel_eval/{key}"] = val

    wandb.log(summary)
    for key, val in summary.items():
        wandb.run.summary[key] = val

    # Save all metrics to JSON
    all_metrics = {
        "pixel_metrics": {
            "iou": avg_iou,
            "dice": avg_dice,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        },
        "map_metrics": map_results,
    }
    if metrics_test is not None:
        all_metrics["yolo_builtin"] = test_metrics_dict

    print("\n=== All Saved Metrics (saved to pixel_metrics.json) ===")
    print(json.dumps(all_metrics, indent=2))

    pixel_metrics_path = PROJECT_DIR / f"{name}_test" / "pixel_metrics.json"
    pixel_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pixel_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to: {pixel_metrics_path}")

    wandb.finish()
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
