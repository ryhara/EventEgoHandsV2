"""Shared YOLO / ground-truth detection helpers for EventEgoHandsV2 scripts."""

import os

import cv2
import numpy as np
import torch


class GTBoxes:
    def __init__(self, xyxy: torch.Tensor, cls: torch.Tensor):
        self.xyxy = xyxy
        self.cls = cls

    def __len__(self):
        return self.xyxy.shape[0]


class GTMasks:
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return self.data.shape[0]


class GTResult:
    def __init__(self, boxes: GTBoxes = None, masks: GTMasks = None):
        self.boxes = boxes
        self.masks = masks


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
            x_min = int(np.min(polygon_np[:, 0]))
            x_max = int(np.max(polygon_np[:, 0]))
            y_min = int(np.min(polygon_np[:, 1]))
            y_max = int(np.max(polygon_np[:, 1]))

            detections.append(
                {
                    "class_id": class_id,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "polygon": polygon_np,
                }
            )

    return detections


def build_gt_yolo_results(batch, use_mask: bool, device):
    """Build YOLO-like results from GT labels for each sample in batch."""
    lnes = batch["lnes"]
    batch_size = lnes.shape[0]
    height, width = lnes.shape[2], lnes.shape[3]
    label_paths = batch.get("gt_yolo_label_path", [""] * batch_size)

    yolo_results = []
    for i in range(batch_size):
        detections = parse_gt_yolo_label(label_paths[i], width, height)
        if len(detections) == 0:
            yolo_results.append(GTResult(boxes=None, masks=None))
            continue

        boxes = torch.tensor(
            [det["bbox"] for det in detections],
            dtype=torch.float32,
            device=device,
        )
        classes = torch.tensor(
            [det["class_id"] for det in detections],
            dtype=torch.float32,
            device=device,
        )
        boxes_obj = GTBoxes(xyxy=boxes, cls=classes)

        if use_mask:
            masks = []
            for det in detections:
                mask = np.zeros((height, width), dtype=np.float32)
                cv2.fillPoly(mask, [det["polygon"]], 1.0)
                masks.append(mask)
            masks_tensor = torch.tensor(np.stack(masks), dtype=torch.float32, device=device)
            masks_obj = GTMasks(data=masks_tensor)
        else:
            masks_obj = None

        yolo_results.append(GTResult(boxes=boxes_obj, masks=masks_obj))

    return yolo_results


def predict_with_yolo(yolo_model, event_frame, cfg, device, predict_batch_size):
    batch_size = event_frame.shape[0]
    yolo_input_list = []

    for i in range(batch_size):
        frame = event_frame[i].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        yolo_input_list.append(frame)

    return yolo_model.predict(
        source=yolo_input_list,
        task="segment",
        save=False,
        verbose=False,
        batch=predict_batch_size,
        conf=cfg.yolo_conf_threshold if hasattr(cfg, "yolo_conf_threshold") else 0.25,
        iou=cfg.yolo_iou_threshold if hasattr(cfg, "yolo_iou_threshold") else 0.45,
        device=device,
    )


def get_yolo_results(batch, yolo_model, cfg, device, predict_batch_size):
    """Return detection results: GT labels when use_gt_mask, otherwise YOLO inference."""
    use_gt_mask = getattr(cfg, "use_gt_mask", False)
    use_mask = getattr(cfg, "use_mask", True)

    if use_gt_mask:
        return build_gt_yolo_results(batch, use_mask=use_mask, device=device)

    event_frame = batch["event_frame"].to(device)
    return predict_with_yolo(yolo_model, event_frame, cfg, device, predict_batch_size)
