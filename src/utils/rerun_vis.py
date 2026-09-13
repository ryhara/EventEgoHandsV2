"""Shared Rerun helpers for the prediction-vs-ground-truth viewers.

`src/v1/vis_nhot3d.py`, `src/v1/vis_eehr.py` and `src/v2/vis.py` all log the same
entity layout, so colours, skeleton tables, image conversion and the CLI/blueprint
handling live here.

Entity layout used by every viewer::

    2D/<name>                 input / intermediate images
    3D/gt/<hand>/{mesh,joints,skeleton}
    3D/pred/<hand>/{mesh,joints,skeleton}
"""

import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

# Ground truth is drawn in a saturated tone and the prediction in the pastel
# colours the dataset viewers use, so both fit in a single 3D view.
GT_COLOR = {"left": [0.85, 0.30, 0.30], "right": [0.25, 0.45, 0.85]}
PRED_COLOR = {"left": [1.0, 0.75, 0.75], "right": [0.75, 0.85, 1.0]}
# 2D overlays are drawn with OpenCV, so they need 0-255 RGB.
PRED_COLOR_2D = {"left": (255, 190, 190), "right": (190, 215, 255)}

HANDS = ("left", "right")

# HOT3D landmark order (20 joints). Index 5 is the wrist, 0-4 are the fingertips.
HOT3D_CONNECTIONS = [
    (5, 8), (8, 9), (9, 10), (10, 1),      # Index
    (5, 11), (11, 12), (12, 13), (13, 2),  # Middle
    (5, 14), (14, 15), (15, 16), (16, 3),  # Ring
    (5, 17), (17, 18), (18, 19), (19, 4),  # Pinky
    (5, 6), (6, 7), (7, 0),                # Thumb
    (8, 11), (11, 14), (14, 17),           # Palm arch
]

# Standard MANO order (21 joints, wrist at 0).
MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 17),        # Index
    (0, 4), (4, 5), (5, 6), (6, 18),        # Middle
    (0, 10), (10, 11), (11, 12), (12, 19),  # Ring
    (0, 7), (7, 8), (8, 9), (9, 20),        # Pinky
    (0, 13), (13, 14), (14, 15), (15, 16),  # Thumb
    (1, 4), (4, 10), (10, 7),               # Palm arch
]

# EEH-R mocap annotations (16 joints, wrist at 15).
MOCAP_CONNECTIONS = [
    (15, 0), (0, 1), (1, 2),       # Index
    (15, 3), (3, 4), (4, 5),       # Middle
    (15, 6), (6, 7), (7, 8),       # Pinky
    (15, 9), (9, 10), (10, 11),    # Ring
    (15, 12), (12, 13), (13, 14),  # Thumb
    (0, 3), (3, 9), (9, 6),        # Palm arch
]


def get_skeleton_connections(num_joints: int):
    """Pick the skeleton table matching the joint count."""
    if num_joints == 21:
        return MANO_CONNECTIONS
    if num_joints == 16:
        return MOCAP_CONNECTIONS
    return HOT3D_CONNECTIONS


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, used when the GT provides no normals."""
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, triangles[:, 0], face_normals)
    np.add.at(vertex_normals, triangles[:, 1], face_normals)
    np.add.at(vertex_normals, triangles[:, 2], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vertex_normals / norms


def project_3d_to_2d(joints_3d: np.ndarray, transl: np.ndarray, focal_length: float):
    """Weak perspective projection used by the 2D joint overlays."""
    joints_cam = joints_3d + transl
    return joints_cam[:, :2] / (joints_cam[:, 2:3] + 1e-6) * focal_length


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------


def _to_numpy_hwc(tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    if array.ndim == 3 and array.shape[0] in (1, 2, 3) and array.shape[0] < array.shape[-1]:
        array = np.transpose(array, (1, 2, 0))
    return array.astype(np.float32)


def _denormalize(array: np.ndarray) -> np.ndarray:
    """Map to [0, 1].

    v1 datasets normalise frames to [-1, 1] while the v2 datasets keep them in
    [0, 1], so the range is detected instead of configured.
    """
    if array.size and array.min() < -0.01:
        array = array * 0.5 + 0.5
    return np.clip(array, 0.0, 1.0)


def event_frame_to_rgb(tensor) -> np.ndarray:
    """3-channel event frame tensor -> uint8 RGB image."""
    array = _denormalize(_to_numpy_hwc(tensor))
    return (array * 255).astype(np.uint8)


def lnes_to_rgb(tensor) -> np.ndarray:
    """2-channel LNES tensor -> uint8 RGB image (positive green, negative red)."""
    array = _denormalize(_to_numpy_hwc(tensor))
    height, width = array.shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 1] = (array[:, :, 0] * 255).astype(np.uint8)  # positive -> green
    rgb[:, :, 0] = (array[:, :, 1] * 255).astype(np.uint8)  # negative -> red
    return rgb


def event_cloud_to_rgb(events, height: int, width: int) -> np.ndarray:
    """Render a filtered event cloud as an RGB image.

    Args:
        events: [N, 5] tensor/array of (x, y, t, p_positive, p_negative) in pixel
            coordinates, i.e. before `pc_normalize_batch`.
    """
    array = events.detach().cpu().numpy() if torch.is_tensor(events) else np.asarray(events)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if array.size == 0:
        return image

    x = array[:, 0].astype(np.int32)
    y = array[:, 1].astype(np.int32)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[inside], y[inside]
    positive = array[inside, 3] > 0
    negative = array[inside, 4] > 0
    image[y[positive], x[positive], 1] = 255  # positive -> green
    image[y[negative], x[negative], 0] = 255  # negative -> red
    return image


def mask_to_rgb(mask) -> np.ndarray:
    """Binary mask (H, W) -> uint8 grayscale RGB image."""
    array = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    array = np.squeeze(array)
    return cv2.cvtColor((array > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)


def overlay_mask(image: np.ndarray, mask, color, alpha: float = 0.4) -> np.ndarray:
    """Blend a binary mask over an RGB image."""
    array = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    array = np.squeeze(array) > 0
    if array.shape[:2] != image.shape[:2]:
        array = cv2.resize(
            array.astype(np.uint8), (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    layer = np.zeros_like(image)
    layer[array] = color
    return cv2.addWeighted(image, 1.0, layer, alpha, 0)


def draw_joints_2d(image, joints_2d, color, radius: int = 2, thickness: int = 1):
    """Draw joints and their skeleton on an RGB image."""
    annotated = image.copy()
    joints_2d = np.asarray(joints_2d, dtype=np.float32)
    if not np.isfinite(joints_2d).all():
        return annotated

    for start, end in get_skeleton_connections(len(joints_2d)):
        if start < len(joints_2d) and end < len(joints_2d):
            cv2.line(
                annotated,
                tuple(map(int, joints_2d[start])),
                tuple(map(int, joints_2d[end])),
                color,
                thickness,
            )
    for point in joints_2d:
        cv2.circle(annotated, tuple(map(int, point)), radius, color, -1)
    return annotated


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------


def log_hand(root: str, hand: str, color, vertices=None, triangles=None, joints=None):
    """Log mesh / joints / skeleton for one hand under ``<root>/<hand>``."""
    if vertices is not None and triangles is not None:
        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.uint32)
        rr.log(
            f"{root}/{hand}/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                vertex_normals=compute_vertex_normals(vertices, triangles),
                vertex_colors=[color] * len(vertices),
            ),
        )

    if joints is None:
        return

    joints = np.asarray(joints, dtype=np.float32)
    rr.log(f"{root}/{hand}/joints", rr.Points3D(joints, colors=[color] * len(joints), radii=0.004))
    strips = [
        [joints[start], joints[end]]
        for start, end in get_skeleton_connections(len(joints))
        if start < len(joints) and end < len(joints)
    ]
    if strips:
        rr.log(
            f"{root}/{hand}/skeleton",
            rr.LineStrips3D(strips, colors=[color] * len(strips), radii=0.0015),
        )


def clear_hand(root: str, hand: str):
    """Remove a hand from the current frame (invalid GT or undetected hand)."""
    rr.log(f"{root}/{hand}", rr.Clear(recursive=True))


# ---------------------------------------------------------------------------
# CLI / blueprint
# ---------------------------------------------------------------------------


def add_rerun_args(parser):
    """Add the output options shared with the dataset viewers."""
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Rerun server IP")
    parser.add_argument("--port", type=int, default=9876, help="Rerun server port")
    parser.add_argument(
        "--web-viewer",
        action="store_true",
        help="Serve a web viewer instead of connecting to a native viewer",
    )
    parser.add_argument("--web-viewer-port", type=int, default=9090)
    parser.add_argument(
        "--recording", type=str, default=None, help="Save to .rrd file instead"
    )
    return parser


def build_blueprint(image_views, name_3d: str = "3D (GT + Pred)") -> rrb.Blueprint:
    """Lay out the 2D views in columns of three next to a single 3D view.

    Args:
        image_views: ``(name, entity_path)`` pairs, in display order.
    """
    columns = []
    for start in range(0, len(image_views), 3):
        chunk = image_views[start : start + 3]
        columns.append(
            rrb.Vertical(
                *[rrb.Spatial2DView(name=name, origin=origin) for name, origin in chunk],
                row_shares=[1] * len(chunk),
            )
        )
    columns.append(rrb.Spatial3DView(name=name_3d, origin="/3D", background=[255, 255, 255]))

    return rrb.Blueprint(
        rrb.Horizontal(*columns, column_shares=[1] * (len(columns) - 1) + [2]),
        rrb.TimePanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
    )


def init_rerun(app_id: str, args, blueprint: rrb.Blueprint):
    """Start a recording, a web viewer or a connection, matching the dataset viewers."""
    rr.init(app_id, spawn=False)
    if args.recording:
        rr.save(str(args.recording))
        print(f"Saving rerun recording to: {args.recording}")
    elif args.web_viewer:
        rr.serve_web(open_browser=False, web_port=args.web_viewer_port)
        print(f"Web viewer on port {args.web_viewer_port}")
    else:
        rr.connect_tcp(f"{args.ip}:{args.port}")
    rr.send_blueprint(blueprint, make_active=True, make_default=True)


def keep_alive_if_web(args):
    """Block so the served page stays reachable after the last frame."""
    if not args.web_viewer:
        return
    print("Keeping web viewer alive (Ctrl+C to exit)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def sequence_subset(dataset, sequence_ids, key_fn):
    """Restrict a dataset to the given sequences without breaking index alignment.

    Args:
        key_fn: maps a dataset index to the string the sequence id is matched against.
    """
    if not sequence_ids:
        return dataset

    wanted = set(sequence_ids)
    indices = [
        i for i in range(len(dataset))
        if any(seq in key_fn(i) for seq in wanted)
    ]
    if not indices:
        raise ValueError(f"No frames found for sequences: {sorted(wanted)}")
    print(f"Sequence filter kept {len(indices)}/{len(dataset)} frames")
    return torch.utils.data.Subset(dataset, indices)
