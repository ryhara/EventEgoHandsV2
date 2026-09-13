import cv2 as cv
import numpy as np
import torch


def dilation_masks(masks, kernel_size=5, iterations=1):
    """
    Apply dilation to masks

    Args:
        masks (torch.Tensor): Input masks [batch_size, 1, H, W] or [batch_size, H, W]
        kernel_size (int): Size of dilation kernel
        iterations (int): Number of dilation iterations

    Returns:
        torch.Tensor: Dilated masks with same shape as input
    """
    masks_np = masks.detach().cpu().numpy().astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_masks_np = np.array(
        [cv.dilate(mask, kernel, iterations=iterations) for mask in masks_np]
    )

    dilated_masks = torch.tensor(
        dilated_masks_np, dtype=masks.dtype, device=masks.device
    )

    return dilated_masks


def compute_pixel_metrics(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute pixel-level segmentation metrics.

    Args:
        pred: Predicted logits or binary mask, shape [B, 1, H, W].
            Binary {0, 1} inputs pass through sigmoid thresholding unchanged.
        gt: Ground truth binary mask, shape [B, 1, H, W].
        threshold: Threshold for binarizing predictions.

    Returns:
        Dict with iou, dice, precision, recall, f1 averaged over the batch.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred_binary * gt).sum(dim=(1, 2, 3))
    union = ((pred_binary + gt) > 0).float().sum(dim=(1, 2, 3))
    pred_sum = pred_binary.sum(dim=(1, 2, 3))
    gt_sum = gt.sum(dim=(1, 2, 3))

    iou = intersection / (union + 1e-6)
    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-6)
    precision = intersection / (pred_sum + 1e-6)
    recall = intersection / (gt_sum + 1e-6)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }
