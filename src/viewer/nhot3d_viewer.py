#!/usr/bin/env python3
"""
NHOT3D Dataset Viewer using Rerun.

Visualizes RGB, event frames, 3D mesh/joints, 2D joint projections, and YOLO annotations.
Usage:
    python nhot3d_viewer.py <sequence_id> --ip <rerun_server_ip> --port <rerun_server_port>
    Example:
        python nhot3d_viewer.py P0001_15c4300c --ip 127.0.0.1 --port 9876
        # Web viewer (open in browser):
        python nhot3d_viewer.py P0001_15c4300c --web-viewer
        # Save to .rrd (serve it later with src/viewer/serve_rrd.sh):
        python nhot3d_viewer.py P0001_15c4300c --recording out.rrd
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Constants
ORIGINAL_WIDTH = 346
ORIGINAL_HEIGHT = 260
EVENT_DISPLAY_SIZE = 346  # Square display size for event frames
IMAGE_SIZE = 512
CROP_SIZE = 224  # Cropped event frame size

# Hand color configuration (RGB format, light pastel colors)
# Left hand: light red/pink pastel, Right hand: light blue pastel
# 3D mesh/joints colors (0.0-1.0 range for Rerun)
HAND_COLOR_LEFT_3D = [1.0, 0.75, 0.75]  # light pastel pink
HAND_COLOR_RIGHT_3D = [0.75, 0.85, 1.0]  # light pastel blue
# 2D visualization colors (0-255 range for OpenCV/images)
HAND_COLOR_LEFT_2D = [255, 190, 190]  # light pastel pink
HAND_COLOR_RIGHT_2D = [190, 215, 255]  # light pastel blue

# Crop offset for center crop from EVENT_DISPLAY_SIZE to CROP_SIZE
CROP_OFFSET = (EVENT_DISPLAY_SIZE - CROP_SIZE) // 2  # 61

# Directory configuration
BASE_DIR = "/path/to/N-HOT3D/Aria"
RGB_BASE_DIR = "/path/to/N-HOT3D/RGB"
SPLITS = ["test", "val", "train"]


def resize_to_square(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resize frame to square (target_size x target_size)."""
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def crop_center(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop frame from center to specified size."""
    h, w = frame.shape[:2]
    y_offset = (h - crop_size) // 2
    x_offset = (w - crop_size) // 2
    return frame[y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]


def find_sequence_dir(sequence_id: str) -> tuple[Path, Path] | None:
    """Find sequence directory in test/val/train splits."""
    for split in SPLITS:
        seq_dir = Path(BASE_DIR) / split / sequence_id
        rgb_dir = Path(RGB_BASE_DIR) / sequence_id
        if seq_dir.exists():
            return seq_dir, rgb_dir
    return None


def get_events_from_h5(h5_file: Path) -> np.ndarray:
    """Load events from h5 file. Returns array with columns [t, x, y, p]."""
    with h5py.File(h5_file, "r") as f:
        events = np.array(f["events"], dtype=np.float32)
    return events


def create_event_frame(events: np.ndarray) -> np.ndarray:
    """Create RGB event frame from events (346x346 square)."""
    frame = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3), dtype=np.uint8)
    if len(events) == 0:
        return resize_to_square(frame, EVENT_DISPLAY_SIZE)
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    p = events[:, 3].astype(np.int32)
    frame[y[p == 1], x[p == 1], 1] = 255  # positive -> green (RGB index 1)
    frame[y[p == 0], x[p == 0], 0] = 255  # negative -> red (RGB index 0)
    return resize_to_square(frame, EVENT_DISPLAY_SIZE)


def create_lnes_frame(events: np.ndarray, window_size: float = 1.0 / 30.0) -> np.ndarray:
    """Create LNES frame from events (2 channels, 346x346 square)."""
    frame = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 2), dtype=np.float32)
    if len(events) == 0:
        return resize_to_square(frame, EVENT_DISPLAY_SIZE)
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    p = events[:, 3].astype(np.int32)
    t = events[:, 0].astype(np.float32)

    t_max = t.max()
    dt = t_max - t
    valid_mask = dt < window_size
    if not np.any(valid_mask):
        return resize_to_square(frame, EVENT_DISPLAY_SIZE)

    t_norm = 1.0 - (dt[valid_mask] / window_size)
    x_v, y_v, p_v = x[valid_mask], y[valid_mask], p[valid_mask]
    frame[y_v[p_v == 1], x_v[p_v == 1], 0] = t_norm[p_v == 1]  # positive channel
    frame[y_v[p_v == 0], x_v[p_v == 0], 1] = t_norm[p_v == 0]  # negative channel
    return resize_to_square(frame, EVENT_DISPLAY_SIZE)


def lnes_to_rgb(lnes: np.ndarray) -> np.ndarray:
    """Convert 2-channel LNES to RGB image."""
    h, w = lnes.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 1] = (lnes[:, :, 0] * 255).astype(np.uint8)  # positive -> green (RGB index 1)
    rgb[:, :, 0] = (lnes[:, :, 1] * 255).astype(np.uint8)  # negative -> red (RGB index 0)
    return rgb


def compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute vertex normals from mesh vertices and triangles."""
    vertex_normals = np.zeros_like(vertices)

    # Compute face normals and accumulate to vertices
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Accumulate face normals to each vertex
    for i, tri in enumerate(triangles):
        vertex_normals[tri[0]] += face_normals[i]
        vertex_normals[tri[1]] += face_normals[i]
        vertex_normals[tri[2]] += face_normals[i]

    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vertex_normals = vertex_normals / norms

    return vertex_normals


def load_gt_file(gt_path: Path) -> dict | None:
    """Load ground truth JSON file."""
    if not gt_path.exists():
        return None
    with open(gt_path, "r") as f:
        return json.load(f)


def load_joints_file(joints_path: Path) -> dict | None:
    """Load joints JSON file."""
    if not joints_path.exists():
        return None
    with open(joints_path, "r") as f:
        return json.load(f)


def load_yolo_bbox(bbox_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO bbox annotations. Returns list of (class_id, cx, cy, w, h)."""
    if not bbox_path.exists():
        return []
    bboxes = []
    with open(bbox_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                bboxes.append((class_id, cx, cy, w, h))
    return bboxes


def load_yolo_seg(seg_path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """Load YOLO segmentation annotations. Returns list of (class_id, polygon)."""
    if not seg_path.exists():
        return []
    segments = []
    with open(seg_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:  # class_id + at least 3 points (6 coords)
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                segments.append((class_id, polygon))
    return segments


def yolo_to_346_cropped_coords(
    cx: float, cy: float, w: float, h: float
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized bbox to 224x224 cropped frame coordinates.

    Flow: YOLO normalized (0-1) -> 346x346 pixels -> center crop to 224x224

    Args:
        cx, cy, w, h: YOLO normalized coordinates (0-1) based on 346x346

    Returns:
        (x_min, y_min, x_max, y_max) in 224x224 cropped frame coordinates
    """
    # Convert to pixel coordinates in 346x346
    px_cx = cx * EVENT_DISPLAY_SIZE
    px_cy = cy * EVENT_DISPLAY_SIZE
    px_w = w * EVENT_DISPLAY_SIZE
    px_h = h * EVENT_DISPLAY_SIZE

    # Apply center crop offset
    x_min = int(px_cx - px_w / 2 - CROP_OFFSET)
    y_min = int(px_cy - px_h / 2 - CROP_OFFSET)
    x_max = int(px_cx + px_w / 2 - CROP_OFFSET)
    y_max = int(px_cy + px_h / 2 - CROP_OFFSET)

    return x_min, y_min, x_max, y_max


def polygon_to_346_cropped_pixels(polygon: list[tuple[float, float]]) -> np.ndarray:
    """Convert normalized polygon to 224x224 cropped frame coordinates.

    Flow: YOLO normalized (0-1) -> 346x346 pixels -> center crop to 224x224
    """
    return np.array(
        [
            [
                int(x * EVENT_DISPLAY_SIZE - CROP_OFFSET),
                int(y * EVENT_DISPLAY_SIZE - CROP_OFFSET),
            ]
            for x, y in polygon
        ],
        dtype=np.int32,
    )


def get_frame_files(seq_dir: Path, rgb_dir: Path, frame_num: str) -> dict:
    """Get all file paths for a frame."""
    return {
        "h5": seq_dir / f"events_{frame_num}.h5",
        "rgb": rgb_dir / "rgb_images" / f"rgb_{frame_num}.jpg",
        "gt": seq_dir / "gt" / f"gt_{frame_num}.jsonl",
        "joints": seq_dir / "gt" / f"joints_{frame_num}.jsonl",
        "bbox": seq_dir / "gt_hand_bbox" / f"frame_{frame_num}.txt",
        "seg": seq_dir / "gt_hand_seg" / f"frame_{frame_num}.txt",
    }


def visualize_frame(seq_dir: Path, rgb_dir: Path, frame_num: str) -> bool:
    """Visualize a single frame."""
    files = get_frame_files(seq_dir, rgb_dir, frame_num)

    # Load event data
    if not files["h5"].exists():
        return False

    events = get_events_from_h5(files["h5"])
    event_frame = create_event_frame(events)  # 346x346
    lnes_frame = create_lnes_frame(events)  # 346x346
    lnes_rgb = lnes_to_rgb(lnes_frame)

    # Crop event frames to 224x224 (center crop)
    event_frame_cropped = crop_center(event_frame, CROP_SIZE)
    lnes_rgb_cropped = crop_center(lnes_rgb, CROP_SIZE)

    # Load YOLO annotations for overlay
    bboxes = load_yolo_bbox(files["bbox"])
    segments = load_yolo_seg(files["seg"])

    bbox_colors = {0: HAND_COLOR_LEFT_2D, 1: HAND_COLOR_RIGHT_2D}

    # Create overlay on event frame
    event_with_overlay = event_frame_cropped.copy()

    # Draw transparent segmentation masks
    if segments:
        mask_overlay = np.zeros_like(event_with_overlay)
        for class_id, polygon in segments:
            pts = polygon_to_346_cropped_pixels(polygon)
            color = bbox_colors.get(class_id, [255, 255, 255])
            cv2.fillPoly(mask_overlay, [pts], color)
        # Blend with transparency (alpha=0.4)
        alpha = 0.4
        event_with_overlay = cv2.addWeighted(event_with_overlay, 1.0, mask_overlay, alpha, 0)

    # Draw bboxes
    if bboxes:
        for class_id, cx, cy, w, h in bboxes:
            x_min, y_min, x_max, y_max = yolo_to_346_cropped_coords(cx, cy, w, h)
            color = bbox_colors.get(class_id, [255, 255, 255])
            cv2.rectangle(event_with_overlay, (x_min, y_min), (x_max, y_max), color, 2)

    # Create bbox/mask on black background
    gt_overlay = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

    if segments:
        mask_layer = np.zeros_like(gt_overlay)
        for class_id, polygon in segments:
            pts = polygon_to_346_cropped_pixels(polygon)
            color = bbox_colors.get(class_id, [255, 255, 255])
            cv2.fillPoly(mask_layer, [pts], color)
        gt_overlay = cv2.addWeighted(gt_overlay, 1.0, mask_layer, 0.4, 0)

    if bboxes:
        for class_id, cx, cy, w, h in bboxes:
            x_min, y_min, x_max, y_max = yolo_to_346_cropped_coords(cx, cy, w, h)
            color = bbox_colors.get(class_id, [255, 255, 255])
            cv2.rectangle(gt_overlay, (x_min, y_min), (x_max, y_max), color, 2)

    # Log event frames
    rr.log("event_frame", rr.Image(event_frame_cropped))
    rr.log("event_frame_overlay", rr.Image(event_with_overlay))
    rr.log("gt_overlay", rr.Image(gt_overlay))
    rr.log("lnes_frame", rr.Image(lnes_rgb_cropped))

    # Load and log RGB (resize to 346x346, then center crop to 224x224)
    if files["rgb"].exists():
        rgb_img = cv2.imread(str(files["rgb"]))
        if rgb_img is not None:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = resize_to_square(rgb_img, EVENT_DISPLAY_SIZE)
            rgb_img = crop_center(rgb_img, CROP_SIZE)
            rr.log("rgb", rr.Image(rgb_img))

    # Load GT for 3D mesh and joints
    gt_data = load_gt_file(files["gt"])
    joints_data = load_joints_file(files["joints"])

    hand_colors = {"left": HAND_COLOR_LEFT_3D, "right": HAND_COLOR_RIGHT_3D}

    for hand in ["left", "right"]:
        color = hand_colors[hand]

        # 3D Mesh with normals for proper shading
        if gt_data and gt_data.get(hand) and gt_data[hand].get("valid"):
            hand_data = gt_data[hand]
            vertices = np.array(hand_data["vertices"], dtype=np.float32)
            triangles = np.array(hand_data["triangles"], dtype=np.uint32)
            vertex_normals = compute_vertex_normals(vertices, triangles)
            rr.log(
                f"3d/mesh_{hand}",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=triangles,
                    vertex_normals=vertex_normals,
                    vertex_colors=[color] * len(vertices),
                ),
            )

            # 3D Joints from hand_landmarks
            if "hand_landmarks" in hand_data:
                joints_3d = np.array(hand_data["hand_landmarks"], dtype=np.float32)
                rr.log(
                    f"3d/joints_{hand}",
                    rr.Points3D(joints_3d, colors=[color] * len(joints_3d), radii=0.005),
                )

        # 2D Joint projections
        if joints_data and joints_data.get(hand):
            hand_joints = joints_data[hand]
            key_2d = "joints_2d" if hand == "left" else "joints2d"
            if key_2d in hand_joints:
                joints_2d = np.array(hand_joints[key_2d], dtype=np.float32)
                rr.log(
                    f"joints_2d_{hand}",
                    rr.Points2D(joints_2d, colors=[color] * len(joints_2d), radii=3),
                )

    return True


def get_all_frame_numbers(seq_dir: Path) -> list[str]:
    """Get all available frame numbers from h5 files."""
    frame_nums = []
    for f in sorted(seq_dir.glob("events_*.h5")):
        num = f.stem.split("_")[1]
        frame_nums.append(num)
    return frame_nums


def create_blueprint() -> rrb.Blueprint:
    """Create a fixed layout blueprint for the viewer."""
    return rrb.Blueprint(
        rrb.Horizontal(
            # Left side: 2D image views
            rrb.Vertical(
                rrb.Spatial2DView(name="Event Frame", origin="/event_frame"),
                rrb.Spatial2DView(name="Event Overlay", origin="/event_frame_overlay"),
                rrb.Spatial2DView(name="GT Overlay", origin="/gt_overlay"),
                row_shares=[1, 1, 1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="LNES", origin="/lnes_frame"),
                rrb.Spatial2DView(name="RGB", origin="/rgb"),
                row_shares=[1, 1],
            ),
            # Right side: 3D view
            rrb.Spatial3DView(name="3D Hand", origin="/3d"),
            column_shares=[1, 1, 2],
        ),
        rrb.TimePanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
    )


def main():
    parser = argparse.ArgumentParser(description="NHOT3D Dataset Viewer")
    parser.add_argument("id", type=str, help="Sequence ID (e.g., P0000000001)")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Rerun server IP")
    parser.add_argument("--port", type=int, default=9876, help="Rerun server port")
    parser.add_argument(
        "--web-viewer",
        action="store_true",
        help="Serve a web viewer instead of connecting to a native viewer",
    )
    parser.add_argument("--web-viewer-port", type=int, default=9090)
    parser.add_argument(
        "--recording", type=Path, default=None, help="Save to .rrd file instead"
    )
    args = parser.parse_args()

    # Find sequence directory
    result = find_sequence_dir(args.id)
    if result is None:
        print(f"ERROR: Sequence {args.id} not found in {SPLITS}")
        return 1

    seq_dir, rgb_dir = result
    print(f"Found sequence: {seq_dir}")

    # Initialize Rerun
    rr.init(f"nhot3d_{args.id}", spawn=False)
    if args.recording:
        rr.save(str(args.recording))
    elif args.web_viewer:
        rr.serve_web(open_browser=True, web_port=args.web_viewer_port)
    else:
        rr.connect_tcp(f"{args.ip}:{args.port}")

    # Send blueprint to fix layout
    blueprint = create_blueprint()
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    # Get all frame numbers
    frame_nums = get_all_frame_numbers(seq_dir)
    print(f"Found {len(frame_nums)} frames")

    # Visualize all frames
    for i, frame_num in enumerate(frame_nums):
        rr.set_time_sequence("frame", i)
        if not visualize_frame(seq_dir, rgb_dir, frame_num):
            print(f"Warning: Failed to visualize frame {frame_num}")

    print("Done")

    # Keep the process alive so the web viewer stays reachable
    if args.web_viewer:
        print("Keeping web viewer alive (Ctrl+C to exit)...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    exit(main())
