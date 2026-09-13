#!/usr/bin/env python3
"""
EventEgoHands Dataset MANO Viewer using Rerun.

Visualizes MANO vertices, joints, mocap joints, event frames, and images.
Usage:
    python eehr_viewer.py <sequence_id> --ip <ip> --port <port> --fps <30|120>
    Example:
        python eehr_viewer.py P04_01 --ip 127.0.0.1 --port 9876 --fps 120
        # Web viewer (open in browser):
        python eehr_viewer.py P04_01 --fps 120 --web-viewer
        # Save to .rrd (serve it later with src/viewer/serve_rrd.sh):
        python eehr_viewer.py P04_01 --fps 120 --recording out.rrd
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Dataset root directory
DATASET_ROOT = Path("/path/to/EEH-R")  # Change this to your EEH-R dataset root
YOLO_MASK_ROOT = Path("/path/to/EEH-R/YOLO")  # Change this to your YOLO segmentation labels root
YOLO_SPLITS = ["train", "val", "test"]

# Image size constants
ORIGINAL_WIDTH = 346
ORIGINAL_HEIGHT = 260
DISPLAY_SIZE = 346  # Square display size
CROP_SIZE = 260  # Center cropped size

# Entity holding the YOLO mask / bbox view
MASK_ENTITY = "mask"
MASK_ALPHA = 0.4
# Entity root for the world-space 3D view (the "3d" root stays camera-space)
WORLD_ROOT = "3d_world"
HEAD_AXIS_LENGTH = 0.1
# YOLO labels are normalized against a plain center crop of the 346x260 source,
# while the displayed frames are resized to a square before cropping.
YOLO_CROP_OFFSET_X = (ORIGINAL_WIDTH - CROP_SIZE) // 2
YOLO_CROP_OFFSET_Y = (ORIGINAL_HEIGHT - CROP_SIZE) // 2
DISPLAY_CROP_OFFSET = (DISPLAY_SIZE - CROP_SIZE) // 2

# Hand colors (matching nhot3d_viewer.py)
# 3D mesh/joints colors (0.0-1.0 range for Rerun)
HAND_COLOR_LEFT_3D = [1.0, 0.75, 0.75]  # light pastel pink
HAND_COLOR_RIGHT_3D = [0.75, 0.85, 1.0]  # light pastel blue
# 2D visualization colors (0-255 range for OpenCV/images)
HAND_COLOR_LEFT_2D = [255, 190, 190]  # light pastel pink
HAND_COLOR_RIGHT_2D = [190, 215, 255]  # light pastel blue
# Mocap joints colors (slightly different for distinction)
MOCAP_COLOR_LEFT_3D = [1.0, 0.5, 0.5]  # darker pink
MOCAP_COLOR_RIGHT_3D = [0.5, 0.7, 1.0]  # darker blue
# Head (camera rigid body) color
HEAD_COLOR_3D = [1.0, 1.0, 0.4]  # yellow

# MANO skeleton connections (21 joints)
# 0: wrist, 1-4: index, 5-8: middle, 9-12: ring, 13-16: pinky, 17-20: thumb
MANO_CONNECTIONS = [
    # Index finger
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Middle finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Ring finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Pinky finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Thumb
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Mocap skeleton connections (16 joints)
# 15: wrist, 0-2: index, 3-5: middle, 6-8: pinky, 9-11: ring, 12-14: thumb
MOCAP_CONNECTIONS = [
    # Index finger
    (15, 0), (0, 1), (1, 2),
    # Middle finger
    (15, 3), (3, 4), (4, 5),
    # Pinky finger
    (15, 6), (6, 7), (7, 8),
    # Ring finger
    (15, 9), (9, 10), (10, 11),
    # Thumb
    (15, 12), (12, 13), (13, 14),
]


def resize_to_square(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resize frame to square (target_size x target_size)."""
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def crop_center(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop frame from center to specified size."""
    h, w = frame.shape[:2]
    y_offset = (h - crop_size) // 2
    x_offset = (w - crop_size) // 2
    return frame[y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]


def get_mano_dir(fps: int) -> Path:
    """Get MANO annotations directory based on fps."""
    return DATASET_ROOT / f"mano_annotations_{fps}fps_v2"


def get_mocap_dir(fps: int) -> Path:
    """Get mocap annotations directory based on fps."""
    return DATASET_ROOT / f"annotations_{fps}fps"


def get_data_dir(sequence_id: str) -> Path:
    """Get data directory for a sequence."""
    return DATASET_ROOT / "data" / sequence_id


def load_sync_mapping() -> dict:
    """Load event start frame mapping from CSV."""
    mapping = {}
    csv_path = DATASET_ROOT / "mapping_sync_event_start_frame.csv"
    if not csv_path.exists():
        return mapping

    with open(csv_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                seq_id = parts[0]
                try:
                    mapping[seq_id] = int(parts[1])
                except ValueError:
                    mapping[seq_id] = None
    return mapping


def get_mano_frame_files(mano_dir: Path, sequence_id: str) -> list[tuple[int, Path]]:
    """Get all MANO frame files sorted by frame_id."""
    seq_dir = mano_dir / sequence_id
    if not seq_dir.exists():
        return []

    frames = []
    for f in sorted(seq_dir.glob("mano_params_*.json")):
        frame_num = int(f.stem.split("_")[-1])
        frames.append((frame_num, f))
    return frames


def load_mano_params(json_path: Path) -> dict | None:
    """Load MANO parameters from JSON file."""
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def load_mocap_params(mocap_dir: Path, sequence_id: str, frame_id: int) -> dict | None:
    """Load mocap parameters from JSONL file."""
    json_path = mocap_dir / sequence_id / "mocap_data_camera" / f"frame_{frame_id:010d}.jsonl"
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def load_world_mocap_params(
    mocap_dir: Path, sequence_id: str, frame_id: int
) -> dict | None:
    """Load world-space mocap parameters (hand joints and head/camera pose)."""
    json_path = mocap_dir / sequence_id / "mocap_data" / f"frame_{frame_id:010d}.jsonl"
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / norm
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def get_head_pose(world_data: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the head (camera rigid body) pose in meters from world mocap data.

    Returns (t_wc in meters, R_wc) or None when the pose is missing.
    """
    camera = world_data.get("camera")
    if not camera or "position" not in camera or "rotation" not in camera:
        return None
    t_wc = np.array(camera["position"], dtype=np.float32) / 1000.0
    r_wc = quat_xyzw_to_matrix(camera["rotation"])
    return t_wc, r_wc


def camera_to_world(points: np.ndarray, t_wc: np.ndarray, r_wc: np.ndarray) -> np.ndarray:
    """Transform camera-space points (meters) into world space (meters).

    Inverse of world_to_camera in export_rerun_json.py: P_w = R_wc @ P_c + t_wc.
    """
    return points @ r_wc.T + t_wc


def load_event_frame(data_dir: Path, frame_idx: int) -> np.ndarray | None:
    """Load event frame image (resized to 346x346, then center cropped to 260x260)."""
    path = data_dir / "event_frames" / f"event_frame_{frame_idx:010d}.jpg"
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_to_square(img, DISPLAY_SIZE)
        img = crop_center(img, CROP_SIZE)
    return img


def load_image_frame(data_dir: Path, frame_idx: int) -> np.ndarray | None:
    """Load image frame (resized to 346x346, then center cropped to 260x260)."""
    path = data_dir / "images" / f"frame_{frame_idx:010d}.png"
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_to_square(img, DISPLAY_SIZE)
        img = crop_center(img, CROP_SIZE)
    return img


def find_yolo_label(sequence_id: str, frame_idx: int) -> Path | None:
    """Find YOLO segmentation label file across train/val/test splits."""
    filename = f"{sequence_id}_frame_{frame_idx:010d}.txt"
    for split in YOLO_SPLITS:
        path = YOLO_MASK_ROOT / "labels" / split / filename
        if path.exists():
            return path
    return None


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


def yolo_polygon_to_display(polygon: list[tuple[float, float]]) -> np.ndarray:
    """Map normalized YOLO label coords onto the displayed frame coords.

    Labels live in the plain center crop of the ORIGINAL_WIDTH x ORIGINAL_HEIGHT
    source, while the displayed frames are resized to DISPLAY_SIZE square first
    and then center cropped. Points may fall outside the crop because the
    vertical stretch pushes part of the label off screen.
    """
    pts = np.array(polygon, dtype=np.float32)
    x_src = pts[:, 0] * CROP_SIZE + YOLO_CROP_OFFSET_X
    y_src = pts[:, 1] * CROP_SIZE + YOLO_CROP_OFFSET_Y
    x = x_src * DISPLAY_SIZE / ORIGINAL_WIDTH - DISPLAY_CROP_OFFSET
    y = y_src * DISPLAY_SIZE / ORIGINAL_HEIGHT - DISPLAY_CROP_OFFSET
    return np.stack([x, y], axis=1)


def create_gt_overlay(segments: list[tuple[int, list[tuple[float, float]]]]) -> np.ndarray:
    """Draw mask and bbox on a black background (same style as nhot3d_viewer)."""
    bbox_colors = {0: HAND_COLOR_LEFT_2D, 1: HAND_COLOR_RIGHT_2D}
    overlay = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

    # Draw transparent segmentation masks
    mask_layer = np.zeros_like(overlay)
    for class_id, polygon in segments:
        pts = yolo_polygon_to_display(polygon).round().astype(np.int32)
        color = bbox_colors.get(class_id, [255, 255, 255])
        cv2.fillPoly(mask_layer, [pts], color)
    overlay = cv2.addWeighted(overlay, 1.0, mask_layer, MASK_ALPHA, 0)

    # Draw bounding boxes derived from polygon extents
    for class_id, polygon in segments:
        pts = yolo_polygon_to_display(polygon)
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)
        color = bbox_colors.get(class_id, [255, 255, 255])
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)

    return overlay


def log_yolo_annotations(
    segments: list[tuple[int, list[tuple[float, float]]]],
) -> None:
    """Log the YOLO mask / bbox overlay as a plain image."""
    if not segments:
        rr.log(MASK_ENTITY, rr.Clear(recursive=True))
        return

    rr.log(MASK_ENTITY, rr.Image(create_gt_overlay(segments)))


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normals for a mesh."""
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    for i, face in enumerate(faces):
        normals[face[0]] += face_normals[i]
        normals[face[1]] += face_normals[i]
        normals[face[2]] += face_normals[i]

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def get_mano_faces() -> np.ndarray:
    """Get MANO face indices (standard MANO topology)."""
    # MANO has 778 vertices and 1538 faces
    # Load from manotorch if available, otherwise use pre-computed
    try:
        from manotorch.manolayer import ManoLayer
        mano = ManoLayer(
            mano_assets_root="src/mano",
            side="right",
            use_pca=False,
            flat_hand_mean=True,
        )
        return mano.th_faces.cpu().numpy().astype(np.int32)
    except Exception as e:
        print(f"WARNING: Could not load MANO faces: {e}")
        return np.array([], dtype=np.int32)


def visualize_skeleton(
    path: str,
    joints: np.ndarray,
    connections: list[tuple[int, int]],
    color: list[float],
) -> None:
    """Visualize skeleton lines connecting joints."""
    lines = []
    for start_idx, end_idx in connections:
        if start_idx < len(joints) and end_idx < len(joints):
            lines.append([joints[start_idx], joints[end_idx]])

    if lines:
        rr.log(
            f"{path}/skeleton",
            rr.LineStrips3D(
                strips=lines,
                colors=[color] * len(lines),
                radii=0.002,
            ),
        )


def visualize_hand(
    path: str,
    hand_data: dict,
    faces: np.ndarray,
    color: list[float],
) -> None:
    """Visualize a single hand mesh and joints with skeleton."""
    if not hand_data.get("valid", False):
        return

    vertices = np.array(hand_data["vertices"], dtype=np.float32)
    joints = np.array(hand_data["joints"], dtype=np.float32)

    # Log mesh
    if len(faces) > 0:
        normals = compute_vertex_normals(vertices, faces)
        rr.log(
            f"{path}/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_normals=normals,
                vertex_colors=[color] * len(vertices),
            ),
        )

    # Log joints
    rr.log(
        f"{path}/joints",
        rr.Points3D(
            positions=joints,
            colors=[color] * len(joints),
            radii=0.005,
        ),
    )

    # Log skeleton lines
    visualize_skeleton(path, joints, MANO_CONNECTIONS, color)


def visualize_mocap_joints(
    path: str,
    joints: np.ndarray,
    color: list[float],
) -> None:
    """Visualize mocap joints with skeleton."""
    # Scale mocap joints from mm to meters
    joints_m = joints / 1000.0

    # Log joints
    rr.log(
        f"{path}/joints",
        rr.Points3D(
            positions=joints_m,
            colors=[color] * len(joints_m),
            radii=0.006,
        ),
    )

    # Log skeleton lines
    visualize_skeleton(path, joints_m, MOCAP_CONNECTIONS, color)


def transform_hand_data(hand_data: dict, t_wc: np.ndarray, r_wc: np.ndarray) -> dict:
    """Return a copy of hand_data with vertices/joints moved into world space."""
    if not hand_data.get("valid", False):
        return hand_data
    transformed = dict(hand_data)
    for key in ("vertices", "joints"):
        points = np.array(hand_data[key], dtype=np.float32)
        transformed[key] = camera_to_world(points, t_wc, r_wc)
    return transformed


def visualize_head_pose(path: str, t_wc: np.ndarray, r_wc: np.ndarray) -> None:
    """Log the head (camera rigid body) pose as a transform with local axes."""
    rr.log(
        path,
        rr.Transform3D(translation=t_wc, mat3x3=r_wc, axis_length=HEAD_AXIS_LENGTH),
    )
    rr.log(f"{path}/center", rr.Points3D(positions=[[0, 0, 0]], colors=[HEAD_COLOR_3D], radii=0.01))


def visualize_world_frame(
    mano_data: dict,
    world_data: dict | None,
    faces: np.ndarray,
) -> None:
    """Visualize hands and the head pose in world space."""
    if world_data is None:
        rr.log(WORLD_ROOT, rr.Clear(recursive=True))
        return

    head = get_head_pose(world_data)
    if head is None:
        rr.log(f"{WORLD_ROOT}/head", rr.Clear(recursive=True))
    else:
        t_wc, r_wc = head
        visualize_head_pose(f"{WORLD_ROOT}/head", t_wc, r_wc)

        # MANO is stored in camera space, so lift it into world space
        for side, color in (
            ("left", HAND_COLOR_LEFT_3D),
            ("right", HAND_COLOR_RIGHT_3D),
        ):
            hand_data = mano_data.get(f"{side}_hand")
            if hand_data is not None:
                visualize_hand(
                    f"{WORLD_ROOT}/mano/{side}",
                    transform_hand_data(hand_data, t_wc, r_wc),
                    faces,
                    color,
                )

    # Mocap joints are already stored in world space
    for side, color in (("left", MOCAP_COLOR_LEFT_3D), ("right", MOCAP_COLOR_RIGHT_3D)):
        joints = world_data.get(side, {}).get("world", {}).get("joints_3d")
        if joints is not None:
            visualize_mocap_joints(
                f"{WORLD_ROOT}/mocap/{side}",
                np.array(joints, dtype=np.float32),
                color,
            )


def visualize_frame(
    mano_data: dict,
    mocap_data: dict | None,
    world_data: dict | None,
    data_dir: Path,
    sequence_id: str,
    mano_frame_id: int,
    image_frame_idx: int,
    faces: np.ndarray,
) -> None:
    """Visualize a single frame."""
    # Visualize MANO hands
    if "left_hand" in mano_data:
        visualize_hand("3d/mano/left", mano_data["left_hand"], faces, HAND_COLOR_LEFT_3D)

    if "right_hand" in mano_data:
        visualize_hand("3d/mano/right", mano_data["right_hand"], faces, HAND_COLOR_RIGHT_3D)

    # Visualize the same pose in world space, together with the head pose
    visualize_world_frame(mano_data, world_data, faces)

    # Visualize mocap joints
    if mocap_data is not None:
        if "left_joints_3d" in mocap_data:
            left_joints = np.array(mocap_data["left_joints_3d"], dtype=np.float32)
            visualize_mocap_joints("3d/mocap/left", left_joints, MOCAP_COLOR_LEFT_3D)

        if "right_joints_3d" in mocap_data:
            right_joints = np.array(mocap_data["right_joints_3d"], dtype=np.float32)
            visualize_mocap_joints("3d/mocap/right", right_joints, MOCAP_COLOR_RIGHT_3D)

    # Load and log images
    event_frame = load_event_frame(data_dir, image_frame_idx)
    if event_frame is not None:
        rr.log("event_frame", rr.Image(event_frame))

    image_frame = load_image_frame(data_dir, image_frame_idx)
    if image_frame is not None:
        rr.log("image", rr.Image(image_frame))

    # Load and log YOLO mask / bbox GT
    yolo_label_path = find_yolo_label(sequence_id, image_frame_idx)
    segments = load_yolo_seg(yolo_label_path) if yolo_label_path is not None else []
    log_yolo_annotations(segments)


def create_blueprint() -> rrb.Blueprint:
    """Create a fixed layout blueprint for the viewer."""
    return rrb.Blueprint(
        rrb.Horizontal(
            # Left side: 2D image views
            rrb.Vertical(
                rrb.Spatial2DView(name="Image", origin="/image"),
                rrb.Spatial2DView(name="Event Frame", origin="/event_frame"),
                rrb.Spatial2DView(name="Mask / BBox", origin=f"/{MASK_ENTITY}"),
                row_shares=[1, 1, 1],
            ),
            # Right side: 3D views (camera space / world space)
            rrb.Tabs(
                rrb.Spatial3DView(name="Camera", origin="/3d"),
                rrb.Spatial3DView(name="World", origin=f"/{WORLD_ROOT}"),
            ),
            column_shares=[1, 2],
        ),
        rrb.TimePanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
    )


def main():
    parser = argparse.ArgumentParser(description="EventEgoHands MANO Viewer")
    parser.add_argument("id", type=str, help="Sequence ID (e.g., P04_01)")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Rerun server IP")
    parser.add_argument("--port", type=int, default=9876, help="Rerun server port")
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        choices=[30, 120],
        help="MANO annotation fps (30 or 120)",
    )
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

    # Check directories
    mano_dir = get_mano_dir(args.fps)
    mocap_dir = get_mocap_dir(args.fps)
    data_dir = get_data_dir(args.id)

    if not mano_dir.exists():
        print(f"ERROR: MANO directory not found: {mano_dir}")
        return 1

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    seq_mano_dir = mano_dir / args.id
    if not seq_mano_dir.exists():
        print(f"ERROR: Sequence not found in MANO annotations: {seq_mano_dir}")
        return 1

    print(f"Sequence: {args.id}")
    print(f"MANO dir: {seq_mano_dir}")
    print(f"Mocap dir: {mocap_dir / args.id}")
    print(f"Data dir: {data_dir}")
    print(f"FPS: {args.fps}")

    # Load sync mapping
    sync_mapping = load_sync_mapping()
    event_start_frame = sync_mapping.get(args.id)
    if event_start_frame is None:
        print(f"WARNING: No sync mapping found for {args.id}, using 0")
        event_start_frame = 0
    else:
        print(f"Event start frame: {event_start_frame}")

    # Get MANO frame files
    mano_frames = get_mano_frame_files(mano_dir, args.id)
    print(f"Found {len(mano_frames)} MANO frames")

    if not mano_frames:
        print("ERROR: No MANO frames found")
        return 1

    # Log first and last frame_id
    first_frame_id = mano_frames[0][0]
    last_frame_id = mano_frames[-1][0]
    print(f"Frame ID range: {first_frame_id} - {last_frame_id}")

    # Load MANO faces (shared for all frames)
    faces = get_mano_faces()
    if len(faces) == 0:
        print("WARNING: Could not load MANO faces, mesh visualization disabled")

    # Initialize Rerun
    rr.init(f"eventegohands_{args.id}_{args.fps}fps", spawn=False)
    if args.recording:
        rr.save(str(args.recording))
    elif args.web_viewer:
        rr.serve_web(open_browser=True, web_port=args.web_viewer_port)
    else:
        rr.connect_tcp(f"{args.ip}:{args.port}")

    # Send blueprint to fix layout
    blueprint = create_blueprint()
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    # Set coordinate system (Left=+X, Up=+Y, Forward=+Z)
    rr.log("3d", rr.ViewCoordinates.LUF, static=True)
    rr.log(WORLD_ROOT, rr.ViewCoordinates.LUF, static=True)

    # Log origin axes
    for root in ("3d", WORLD_ROOT):
        rr.log(
            f"{root}/origin_axes",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB for XYZ
                labels=["X", "Y", "Z"],
            ),
            static=True,
        )

    # Visualize all frames
    for i, (mano_frame_id, mano_path) in enumerate(mano_frames):
        # Calculate corresponding image frame index
        # mano_frame_id is at args.fps
        # images are at 30fps
        if args.fps == 120:
            # 120fps: frame_9513.jsonl -> event_frame_2378.jpg (9513 // 4 = 2378)
            image_frame_idx = mano_frame_id // 4 + event_start_frame
        else:
            # 30fps: frame_2379.jsonl -> event_frame_2379.jpg (direct mapping)
            image_frame_idx = mano_frame_id + event_start_frame

        print(f"mano_frame_id: {mano_frame_id}, image_frame_idx: {image_frame_idx}")

        # Set time
        rr.set_time_sequence("mano_frame_id", mano_frame_id)
        rr.set_time_sequence("image_frame", image_frame_idx)
        rr.set_time_sequence("index", i)

        # Load MANO data
        mano_data = load_mano_params(mano_path)
        if mano_data is None:
            print(f"WARNING: Failed to load {mano_path}")
            continue

        # Load mocap data (camera space) and world-space data with the head pose
        mocap_data = load_mocap_params(mocap_dir, args.id, mano_frame_id)
        world_data = load_world_mocap_params(mocap_dir, args.id, mano_frame_id)

        # Visualize frame
        visualize_frame(
            mano_data,
            mocap_data,
            world_data,
            data_dir,
            args.id,
            mano_frame_id,
            image_frame_idx,
            faces,
        )

        if i % 100 == 0:
            print(f"Processed {i + 1}/{len(mano_frames)} frames")

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
