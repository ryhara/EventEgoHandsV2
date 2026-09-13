#!/usr/bin/env python3
"""
EEH-R (real) dataset class for PyTorch.

Dataset class for loading EEH-R data with MANO annotations.
Compatible with NHOT3DYOLODataset interface for training.
"""

import csv
import os
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.representation import get_v2_transform


def load_light_dark_mapping(dataset_root):
    """Load mapping_id_to_scene.csv and return {id: 'light' or 'dark'} dict."""
    mapping = {}
    csv_path = os.path.join(dataset_root, "mapping_id_to_scene.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: mapping_id_to_scene.csv not found at {csv_path}")
        return mapping
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["id"]] = row["light_or_dark"]
    return mapping

# Image size constants
ORIGINAL_WIDTH = 346
ORIGINAL_HEIGHT = 260
DISPLAY_SIZE = 346  # Square display size

# LNES constants
# lnes_frames/*.png are pre-cropped to 260x260 (center crop on width)
LNES_CROP_SIZE = 260
LNES_CROP_X_OFFSET = (ORIGINAL_WIDTH - LNES_CROP_SIZE) // 2  # 43
# LNES integration window used when the stored PNGs were generated.
# Verified against lnes_frames/*.png: reconstructing from events/*.h5 with this
# window reproduces the stored PNG up to +/-1 (rounding).
DEFAULT_LNES_WINDOW_SEC = 1.0 / 30.0
# Base event-frame rate of the recordings (events/*.h5, lnes_frames/*.png)
BASE_EVENT_FPS = 30
# Number of raw event chunks kept in memory per worker (sequential access reuses them)
EVENT_CHUNK_CACHE_SIZE = 8

# HOT3D joint mapping: selects 20 joints from MANO 21 joints (drops thumb CMC at index 13)
# Same mapping as hot3d/hot3d/data_loaders/mano_layer.py
MANOANNOTATION_TO_HOT3D_JOINT_MAPPING = [
    4, 8, 12, 16, 20,  # Fingertips: thumb, index, middle, ring, pinky
    0,                    # Wrist
    2, 3,              # Thumb: MCP, IP
    5, 6, 7,             # Index: MCP, PIP, DIP
    9, 10, 11,             # Middle: MCP, PIP, DIP
    13, 14, 15,          # Ring: MCP, PIP, DIP
    17, 18, 19,             # Pinky: MCP, PIP, DIP
]

NUM_HOT3D_JOINTS = 20
NUM_MOCAP_JOINTS = 16
NUM_MANO_VERTICES = 778


def resize_to_square(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resize frame to square (target_size x target_size)."""
    return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def crop_center(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop frame from center to specified size."""
    h, w = frame.shape[:2]
    y_offset = (h - crop_size) // 2
    x_offset = (w - crop_size) // 2
    return frame[y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]


class EEHRDataset(Dataset):
    """
    Dataset class for EventEgoHands data with MANO annotations.

    Loads event frames and MANO hand parameters.
    Compatible with NHOT3DYOLODataset interface.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        fps: int = 30,
        output_size: tuple = (224, 224),
        use_center_crop: bool = True,
        ignore_files_count: int = 50,
        dataset_divide: int = 1,
        return_raw_annotations: bool = False,
        annotation_source: str = "mano",
        annotation_fps: int = 30,
        use_gt_mask: bool = False,
        gt_mask_root: Optional[str] = None,
        load_h5: bool = False,
        event_number_threshold: int = 2048,
        load_grayscale_image: bool = False,
        lnes_from_h5: bool = False,
        lnes_window_sec: float = DEFAULT_LNES_WINDOW_SEC,
        allow_subframe_gt_mask: bool = False,
    ):
        """
        Initialize EEHRDataset.

        Args:
            dataset_root: Root directory of the dataset
            split: Data split ("train", "valid", "test")
            fps: Annotation FPS (30 or 120)
            output_size: Output image size (H, W)
            use_center_crop: If True, resize to square then center crop
            ignore_files_count: Number of frames to skip from start/end
            dataset_divide: Divide dataset size by this factor
            return_raw_annotations: If True, return raw MANO JSON data
            annotation_source: "mano" or "mocap"
            annotation_fps: Annotation FPS (default: 30)
            use_gt_mask: If True, use GT mask labels as detection GT source
            gt_mask_root: Root path containing YOLO-format labels
            lnes_from_h5: If True, rebuild LNES / event frames from events/*.h5
                instead of reading the pre-rendered lnes_frames/*.png.
                Forced on when annotation_fps > 30, because the stored PNGs only
                exist at the base event rate (30fps).
            lnes_window_sec: LNES integration window in seconds (default: 1/30).
                The window length is kept constant across sub-frames so that the
                representation statistics match the 30fps training data.
            allow_subframe_gt_mask: Opt out of the guard that rejects
                use_gt_mask together with annotation_fps > 30. GT mask labels only
                exist on the 30fps grid, so every sub-frame of one event frame
                would share the same label while being up to 3/120s apart.
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.fps = fps
        self.annotation_source = annotation_source.lower()
        self.annotation_fps = annotation_fps if annotation_fps is not None else fps
        self.output_size = output_size
        self.use_center_crop = use_center_crop
        self.ignore_files_count = ignore_files_count
        self.dataset_divide = dataset_divide
        self.return_raw_annotations = return_raw_annotations
        self.use_gt_mask = use_gt_mask
        self.gt_mask_root = Path(gt_mask_root) if gt_mask_root else None
        self.load_h5 = load_h5
        self.event_number_threshold = event_number_threshold
        self.load_grayscale_image = load_grayscale_image
        self.lnes_window_sec = float(lnes_window_sec)

        # Number of annotation frames per stored event frame (30fps -> 1, 120fps -> 4)
        self.sub_frame_count = max(1, int(round(self.annotation_fps / BASE_EVENT_FPS)))
        # Sub-frames other than #0 have no pre-rendered PNG, so they must be
        # rebuilt from the raw events.
        self.lnes_from_h5 = bool(lnes_from_h5) or self.sub_frame_count > 1
        # Per-worker cache of raw event chunks: one sample touches up to 3 chunks
        # and consecutive samples share most of them.
        self._event_chunk_cache = OrderedDict()

        if self.annotation_source not in ["mano", "mocap"]:
            raise ValueError(
                f"Invalid annotation_source: {self.annotation_source}. "
                "Expected 'mano' or 'mocap'."
            )

        # GT mask labels are keyed by the 30fps event frame index only. At a higher
        # annotation rate all sub-frames of one event frame would reuse that single
        # label, and sub-frames 1..N-1 are up to (N-1)/annotation_fps seconds away
        # from the moment the label was drawn for.
        if self.use_gt_mask and self.sub_frame_count > 1 and not allow_subframe_gt_mask:
            stale_ms = (self.sub_frame_count - 1) / self.annotation_fps * 1000
            raise ValueError(
                f"use_gt_mask=True is not supported with annotation_fps="
                f"{self.annotation_fps}: GT mask labels only exist at "
                f"{BASE_EVENT_FPS}fps, so {self.sub_frame_count} annotation frames "
                f"would share one label and the last one would be misaligned by "
                f"{stale_ms:.1f}ms. Set annotation_fps={BASE_EVENT_FPS}, or pass "
                "allow_subframe_gt_mask=True if the misalignment is intended."
            )

        # Load sync mapping
        self.sync_mapping = self._load_sync_mapping()

        # Load sequence IDs from split file
        self.sequence_ids = self._load_split_file()
        print(f"Loaded {len(self.sequence_ids)} sequences from {split} split")

        # Resolve GT mask labels dir
        self.gt_mask_label_dir = None
        if self.use_gt_mask:
            self.gt_mask_label_dir = self._resolve_gt_mask_label_dir()

        # Build frame list
        self.frame_list = self._build_frame_list()
        print(f"Total frames before filtering: {len(self.frame_list)}")

        # Apply dataset_divide to randomly sample subset
        if self.dataset_divide > 1 and len(self.frame_list) > 0:
            original_length = len(self.frame_list)
            target_length = max(1, original_length // self.dataset_divide)
            self.frame_list = random.sample(self.frame_list, target_length)
            print(
                f"Dataset reduced by divide={self.dataset_divide}: "
                f"{original_length} -> {len(self.frame_list)}"
            )

        # Transform for event frames (3-channel)
        self.transform_3channel = get_v2_transform(
            self.output_size, self.use_center_crop, normalize=False
        )
        # Transform for LNES (2-channel)
        self.transform_2channel = get_v2_transform(
            self.output_size, self.use_center_crop, normalize=False
        )

        if self.lnes_from_h5:
            print(
                f"LNES source: events/*.h5 (annotation_fps={self.annotation_fps}, "
                f"sub_frames={self.sub_frame_count}, "
                f"window={self.lnes_window_sec * 1000:.2f}ms)"
            )
        else:
            print("LNES source: lnes_frames/*.png")

        print(f"Dataset length: {len(self)}")

    def __len__(self):
        return len(self.frame_list)

    def _load_sync_mapping(self) -> dict:
        """Load event start frame mapping from CSV."""
        mapping = {}
        csv_path = self.dataset_root / "mapping_sync_event_start_frame.csv"
        if not csv_path.exists():
            print(f"WARNING: Sync mapping file not found: {csv_path}")
            return mapping

        with open(csv_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    seq_id = parts[0]
                    try:
                        mapping[seq_id] = int(parts[1])
                    except ValueError:
                        mapping[seq_id] = None
        return mapping

    def _resolve_gt_mask_label_dir(self) -> Path:
        """Resolve GT mask label directory with valid->val fallback."""
        if self.gt_mask_root is None:
            raise ValueError("use_gt_mask=True requires gt_mask_root")

        split_candidates = [self.split]
        if self.split == "valid":
            split_candidates = ["val", "valid"]

        for split_name in split_candidates:
            label_dir = self.gt_mask_root / "labels" / split_name
            if label_dir.exists():
                print(f"Using GT mask labels from: {label_dir}")
                return label_dir

        searched = [str(self.gt_mask_root / "labels" / s) for s in split_candidates]
        raise FileNotFoundError(
            f"GT mask label directory not found. Searched: {searched}"
        )

    def _load_split_file(self) -> list:
        """Load sequence IDs from split file."""
        split_file = self.dataset_root / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            sequence_ids = [line.strip() for line in f if line.strip()]
        return sequence_ids

    def _get_mano_dir(self) -> Path:
        """Get MANO annotations directory based on fps."""
        return self.dataset_root / f"mano_annotations_{self.annotation_fps}fps"

    def _get_mocap_dir(self) -> Path:
        """Get MoCap annotations directory based on fps."""
        return self.dataset_root / f"annotations_{self.annotation_fps}fps"

    def _get_data_dir(self, sequence_id: str) -> Path:
        """Get data directory for a sequence."""
        return self.dataset_root / "data" / sequence_id

    def _make_gt_label_path(self, sequence_id: str, event_frame_idx: int) -> Optional[Path]:
        """Get GT label path for a sequence/frame if GT mask mode is enabled."""
        if not self.use_gt_mask:
            return None
        filename = f"{sequence_id}_frame_{event_frame_idx:010d}.txt"
        return self.gt_mask_label_dir / filename

    def _build_mano_frame_list(self) -> list:
        """
        Build list of valid MANO frames from all sequences.

        Each entry is a tuple:
        (sequence_id, annotation_frame_id, annotation_path, event_frame_idx,
         gt_label_path, sub_frame_idx)
        """
        frame_list = []
        mano_dir = self._get_mano_dir()

        for seq_id in self.sequence_ids:
            seq_mano_dir = mano_dir / seq_id
            if not seq_mano_dir.exists():
                print(f"WARNING: MANO directory not found for {seq_id}")
                continue

            # Get all MANO files sorted by frame_id
            mano_files = sorted(seq_mano_dir.glob("mano_params_*.json"))
            if not mano_files:
                print(f"WARNING: No MANO files found for {seq_id}")
                continue

            # Extract frame IDs
            frame_ids = []
            for f in mano_files:
                frame_num = int(f.stem.split("_")[-1])
                event_frame_idx = self._get_event_frame_idx(seq_id, frame_num)
                sub_frame_idx = self._get_sub_frame_idx(frame_num)
                gt_label_path = self._make_gt_label_path(seq_id, event_frame_idx)
                if gt_label_path is not None and not gt_label_path.exists():
                    continue
                frame_ids.append(
                    (frame_num, f, event_frame_idx, gt_label_path, sub_frame_idx)
                )

            # Apply ignore_files_count
            total_frames = len(frame_ids)
            print(f"Total frames for {seq_id}: {total_frames}")
            if total_frames == 0:
                print(f"WARNING: No valid frames for {seq_id}, skipping")
                continue
            if self.ignore_files_count > 0 and total_frames < 2 * self.ignore_files_count:
                print(
                    f"WARNING: Not enough frames for {seq_id} "
                    f"({total_frames} < {2 * self.ignore_files_count})"
                )
                continue

            # Filter out first and last ignore_files_count frames
            if self.ignore_files_count > 0:
                valid_frames = frame_ids[self.ignore_files_count : -self.ignore_files_count]
            else:
                valid_frames = frame_ids

            for (
                mano_frame_id,
                mano_path,
                event_frame_idx,
                gt_label_path,
                sub_frame_idx,
            ) in valid_frames:
                frame_list.append(
                    (
                        seq_id,
                        mano_frame_id,
                        mano_path,
                        event_frame_idx,
                        gt_label_path,
                        sub_frame_idx,
                    )
                )

        return frame_list

    def _build_mocap_frame_list(self) -> list:
        """
        Build list of valid MoCap frames from all sequences.

        Each entry is a tuple:
        (sequence_id, annotation_frame_id, annotation_path, event_frame_idx,
         gt_label_path, sub_frame_idx)
        """
        frame_list = []
        mocap_dir = self._get_mocap_dir()

        for seq_id in self.sequence_ids:
            seq_mocap_dir = mocap_dir / seq_id / "mocap_data_camera"
            if not seq_mocap_dir.exists():
                print(f"WARNING: MoCap directory not found for {seq_id}")
                continue

            mocap_files = sorted(seq_mocap_dir.glob("frame_*.jsonl"))
            if not mocap_files:
                print(f"WARNING: No MoCap files found for {seq_id}")
                continue

            frame_ids = []
            for f in mocap_files:
                frame_num = int(f.stem.split("_")[-1])
                event_frame_idx = self._get_event_frame_idx(seq_id, frame_num)
                sub_frame_idx = self._get_sub_frame_idx(frame_num)
                gt_label_path = self._make_gt_label_path(seq_id, event_frame_idx)
                if gt_label_path is not None and not gt_label_path.exists():
                    continue
                frame_ids.append(
                    (frame_num, f, event_frame_idx, gt_label_path, sub_frame_idx)
                )

            total_frames = len(frame_ids)
            print(f"Total frames for {seq_id}: {total_frames}")
            if total_frames == 0:
                print(f"WARNING: No valid frames for {seq_id}, skipping")
                continue
            if self.ignore_files_count > 0 and total_frames < 2 * self.ignore_files_count:
                print(
                    f"WARNING: Not enough frames for {seq_id} "
                    f"({total_frames} < {2 * self.ignore_files_count})"
                )
                continue

            if self.ignore_files_count > 0:
                valid_frames = frame_ids[self.ignore_files_count : -self.ignore_files_count]
            else:
                valid_frames = frame_ids
            for (
                frame_id,
                mocap_path,
                event_frame_idx,
                gt_label_path,
                sub_frame_idx,
            ) in valid_frames:
                frame_list.append(
                    (
                        seq_id,
                        frame_id,
                        mocap_path,
                        event_frame_idx,
                        gt_label_path,
                        sub_frame_idx,
                    )
                )

        return frame_list

    def _build_frame_list(self) -> list:
        """
        Build list of valid frames from all sequences.
        """
        if self.annotation_source == "mocap":
            return self._build_mocap_frame_list()
        return self._build_mano_frame_list()

    def _load_mano_params(self, mano_path: Path) -> dict:
        """Load MANO parameters from JSON file."""
        with open(mano_path, "r") as f:
            return json.load(f)

    def _load_mocap_params(self, mocap_path: Path) -> dict:
        """Load MoCap parameters from JSONL file."""
        with open(mocap_path, "r") as f:
            content = f.read().strip()
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            first_line = content.splitlines()[0]
            return json.loads(first_line)

    def _get_event_frame_idx(self, sequence_id: str, mano_frame_id: int) -> int:
        """
        Calculate event frame index from MANO frame ID.

        30fps: event_frame_idx = mano_frame_id + event_start_frame
        120fps: event_frame_idx = mano_frame_id // 4 + event_start_frame
        """
        event_start_frame = self.sync_mapping.get(sequence_id, 0)
        if event_start_frame is None:
            event_start_frame = 0

        return mano_frame_id // self.sub_frame_count + event_start_frame

    def _get_sub_frame_idx(self, mano_frame_id: int) -> int:
        """
        Sub-frame offset inside one stored (30fps) event frame.

        Verified on the dataset: MANO frame 4n at 120fps is bit-identical to MANO
        frame n at 30fps, so sub-frame 0 is time-aligned with the 30fps frame and
        sub-frames 1..3 are 1/120s, 2/120s and 3/120s later.
        """
        return mano_frame_id % self.sub_frame_count

    def _load_event_frame(self, data_dir: Path, frame_idx: int) -> np.ndarray:
        """Load event frame image."""
        path = data_dir / "event_frames" / f"event_frame_{frame_idx:010d}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Event frame not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _convert_lnes_to_3channel(self, lnes_frame: np.ndarray) -> np.ndarray:
        """
        Convert 2-channel LNES frame to 3-channel image.

        Args:
            lnes_frame: numpy array with shape (H, W, 2) with values in 0-255 range

        Returns:
            lnes_3channel: numpy array with shape (H, W, 3) as uint8
        """
        h, w = lnes_frame.shape[:2]
        lnes_3channel = np.zeros((h, w, 3), dtype=np.float32)
        lnes_3channel[:, :, 0] = 0
        lnes_3channel[:, :, 1] = lnes_frame[:, :, 0]  # positive events
        lnes_3channel[:, :, 2] = lnes_frame[:, :, 1]  # negative events
        lnes_3channel = np.clip(lnes_3channel, 0, 255).astype(np.uint8)
        return lnes_3channel

    def _load_lnes_frame(self, data_dir: Path, frame_idx: int) -> np.ndarray:
        """
        Load LNES frame image and convert to 2-channel format.

        LNES PNG files are stored as BGR where:
        - B channel: 0 (unused)
        - G channel: positive events
        - R channel: negative events

        Returns:
            lnes: (H, W, 2) numpy array where [0]=positive, [1]=negative
        """
        path = data_dir / "lnes_frames" / f"frame_{frame_idx:010d}.png"
        if not path.exists():
            raise FileNotFoundError(f"LNES frame not found: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load LNES image: {path}")

        # Extract G (positive) and R (negative) channels
        # OpenCV reads as BGR, so G=1, R=2
        g_channel = img[:, :, 1]  # positive events
        r_channel = img[:, :, 2]  # negative events

        # Stack to (H, W, 2)
        lnes = np.stack([g_channel, r_channel], axis=-1)
        return lnes

    # ------------------------------------------------------------------
    # LNES / event reconstruction from raw events (needed for >30fps)
    # ------------------------------------------------------------------

    def _read_event_chunk(self, data_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
        """
        Read one raw event chunk (events/events_*.h5) as [N, 4] (t, x, y, p).

        Returns None when the chunk does not exist or is empty. Timestamps are
        kept in float64 (seconds) so that sub-frame windows stay accurate.
        """
        if frame_idx < 0:
            return None

        key = (str(data_dir), frame_idx)
        if key in self._event_chunk_cache:
            self._event_chunk_cache.move_to_end(key)
            return self._event_chunk_cache[key]

        path = data_dir / "events" / f"events_{frame_idx:010d}.h5"
        events = None
        if path.exists():
            with h5py.File(str(path), "r") as f:
                raw = np.asarray(f["events"], dtype=np.float64)
            if raw.ndim == 2 and raw.shape[0] > 0:
                events = raw

        self._event_chunk_cache[key] = events
        if len(self._event_chunk_cache) > EVENT_CHUNK_CACHE_SIZE:
            self._event_chunk_cache.popitem(last=False)
        return events

    def _load_window_events(
        self, data_dir: Path, frame_idx: int, sub_frame_idx: int
    ) -> Optional[tuple]:
        """
        Collect the raw events falling inside the LNES window of one sub-frame.

        The stored event frame `frame_idx` covers [T1 - W, T1] where T1 is its
        chunk boundary and W = lnes_window_sec. Sub-frame k is 1/annotation_fps * k
        later, so its window is [T1 + k/fps - W, T1 + k/fps).

        Chunks are contiguous and follow a repeating 30/30/40ms pattern (mean
        exactly 1/30s), so a window can reach into the previous and the next
        chunk; both are pulled in as needed.

        Returns:
            (events [N, 4] sorted by t, t_start, t_end) or None when no events
            are available.
        """
        current = self._read_event_chunk(data_dir, frame_idx)
        # The chunk boundary is the first timestamp of the next chunk; fall back
        # to the last timestamp of the current chunk at the end of a sequence.
        next_chunk = self._read_event_chunk(data_dir, frame_idx + 1)
        if next_chunk is not None:
            t_boundary = float(next_chunk[:, 0].min())
        elif current is not None:
            t_boundary = float(current[:, 0].max())
        else:
            return None

        t_end = t_boundary + sub_frame_idx / float(self.annotation_fps)
        t_start = t_end - self.lnes_window_sec

        chunks = []
        if current is not None and float(current[:, 0].min()) > t_start:
            previous = self._read_event_chunk(data_dir, frame_idx - 1)
            if previous is not None:
                chunks.append(previous)
        if current is not None:
            chunks.append(current)
        if next_chunk is not None and float(next_chunk[:, 0].min()) < t_end:
            chunks.append(next_chunk)

        if not chunks:
            return None

        events = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        mask = (events[:, 0] >= t_start) & (events[:, 0] < t_end)
        events = events[mask]
        if events.shape[0] == 0:
            return None

        order = np.argsort(events[:, 0], kind="stable")
        return events[order], t_start, t_end

    def _build_lnes_from_events(
        self, events: Optional[np.ndarray], t_start: float, t_end: float
    ) -> np.ndarray:
        """
        Rasterize raw events into a 2-channel LNES frame.

        Matches the pre-rendered lnes_frames/*.png: per pixel and polarity the
        most recent event wins, its normalized timestamp is stored in 0-255, and
        the 346-wide sensor is center-cropped to 260x260.

        Args:
            events: [N, 4] (t, x, y, p) sorted by t, or None
            t_start / t_end: LNES window bounds in seconds

        Returns:
            lnes: (260, 260, 2) uint8 where [..., 0]=positive, [..., 1]=negative
        """
        lnes = np.zeros((LNES_CROP_SIZE, LNES_CROP_SIZE, 2), dtype=np.uint8)
        if events is None or events.shape[0] == 0:
            return lnes

        duration = t_end - t_start
        if duration <= 0:
            return lnes

        x = events[:, 1].astype(np.int32) - LNES_CROP_X_OFFSET
        y = events[:, 2].astype(np.int32)
        keep = (
            (x >= 0) & (x < LNES_CROP_SIZE) & (y >= 0) & (y < LNES_CROP_SIZE)
        )
        if not np.any(keep):
            return lnes

        x, y = x[keep], y[keep]
        t = events[keep, 0]
        polarity = events[keep, 3]

        values = np.clip(np.round((t - t_start) / duration * 255.0), 0, 255).astype(
            np.uint8
        )
        channel = np.where(polarity == 1, 0, 1)
        # events are time-sorted, so later events overwrite earlier ones
        lnes[y, x, channel] = values
        return lnes

    def _events_to_pointcloud(self, events: Optional[np.ndarray]) -> torch.Tensor:
        """
        Convert raw [N, 4] (t, x, y, p) events into the ev2hands point cloud.

        Same aggregation as _load_ev2hands_events, but on an already selected set
        of events so that sub-frame windows can be used.

        Returns:
            [5, event_number_threshold] tensor of (x, y, t, n_pos, n_neg)
        """
        if events is None or events.shape[0] == 0:
            return torch.zeros(5, self.event_number_threshold, dtype=torch.float32)

        t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        # Anchor timestamps at the window start; downstream normalization is
        # affine per sample, so this only improves float32 precision.
        t = (t - t.min()).astype(np.float32)

        event_grid = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3), dtype=np.float32)
        count_grid = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH), dtype=np.float32)

        xi, yi = x.astype(np.int32), y.astype(np.int32)
        np.add.at(event_grid, (yi, xi, 0), t)
        np.add.at(event_grid, (yi, xi, 1), p == 1)
        np.add.at(event_grid, (yi, xi, 2), p != 1)
        np.add.at(count_grid, (yi, xi), 1)

        nz_y, nz_x = np.nonzero(count_grid)
        t_avg = event_grid[nz_y, nz_x, 0] / count_grid[nz_y, nz_x]
        p_evn = event_grid[nz_y, nz_x, 1]
        n_evn = event_grid[nz_y, nz_x, 2]

        aggregated = np.hstack(
            [
                nz_x[:, np.newaxis],
                nz_y[:, np.newaxis],
                t_avg[:, np.newaxis],
                p_evn[:, np.newaxis],
                n_evn[:, np.newaxis],
            ]
        ).astype(np.float32)

        return self._resample_pointcloud(aggregated)

    def _resample_pointcloud(self, events: np.ndarray) -> torch.Tensor:
        """Resample aggregated [N, 5] events to event_number_threshold and to [5, N]."""
        if events.shape[0] == 0:
            return torch.zeros(5, self.event_number_threshold, dtype=torch.float32)

        if events.shape[0] < self.event_number_threshold:
            sampled_indices = np.random.choice(
                events.shape[0], self.event_number_threshold - events.shape[0]
            )
            events = np.concatenate([events, events[sampled_indices]], 0)
        elif events.shape[0] > self.event_number_threshold:
            sampled_indices = np.random.choice(
                events.shape[0], self.event_number_threshold
            )
            events = events[sampled_indices]

        return torch.tensor(events, dtype=torch.float32).permute(1, 0)

    def _load_ev2hands_events(self, data_dir: Path, frame_idx: int) -> torch.Tensor:
        """
        Load raw events from H5 and convert to ev2hands point cloud format.

        Reads [N, 4] (t, x, y, p) events from H5, aggregates per pixel,
        and returns [5, event_number_threshold] point cloud tensor.

        Reference: src/datasets/nhot3d_dataset.py (create_ev2hands_event)
        """
        return self._events_to_pointcloud(self._read_event_chunk(data_dir, frame_idx))

    def _get_default_hand_data(self) -> dict:
        """Return zero-filled hand data dict for invalid hands."""
        return {
            "shape": torch.zeros(10, dtype=torch.float32),
            "trans": torch.zeros(3, dtype=torch.float32),
            "global_orient": torch.zeros(3, dtype=torch.float32),
            "hand_pose": torch.zeros(15, dtype=torch.float32),
            "valid": False,
        }

    def _get_default_joints_data(self, annotation_format: str = "mano20") -> dict:
        """Return zero-filled joints data dict for invalid hands."""
        if annotation_format == "mocap16":
            return {
                "joints_3d": torch.zeros(NUM_MOCAP_JOINTS, 3, dtype=torch.float32),
                "vertices_3d": torch.zeros(NUM_MANO_VERTICES, 3, dtype=torch.float32),
            }
        return {
            "joints_3d": torch.zeros(NUM_HOT3D_JOINTS, 3, dtype=torch.float32),
            "vertices_3d": torch.zeros(NUM_MANO_VERTICES, 3, dtype=torch.float32),
        }

    def _apply_joint_mapping(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Convert MANO 21 joints to HOT3D 20 joints format.

        Args:
            joints: (21, 3) tensor in standard MANO joint order

        Returns:
            (20, 3) tensor in HOT3D joint order
        """
        if joints.shape[0] == 21:
            return joints[MANOANNOTATION_TO_HOT3D_JOINT_MAPPING]
        elif joints.shape[0] == NUM_HOT3D_JOINTS:
            return joints
        else:
            raise ValueError(
                f"Unexpected joint count: {joints.shape[0]}. "
                f"Expected 21 (MANO) or {NUM_HOT3D_JOINTS} (HOT3D)."
            )

    def _process_hand_data(self, hand_data: dict, hand_key: str) -> tuple:
        """
        Process raw MANO hand data into tensor format.

        Args:
            hand_data: Raw hand data from JSON
            hand_key: "left_hand" or "right_hand"

        Returns:
            Tuple of (hand_pose_dict, joints_dict)
        """
        if hand_data is None or not hand_data.get("valid", False):
            return self._get_default_hand_data(), self._get_default_joints_data()

        joints_raw = torch.tensor(hand_data["joints"], dtype=torch.float32)
        joints_mapped = self._apply_joint_mapping(joints_raw)
        vertices = torch.tensor(hand_data["vertices"], dtype=torch.float32)

        hand_pose_dict = {
            "shape": torch.tensor(hand_data["betas"], dtype=torch.float32),
            "trans": torch.tensor(hand_data["transl"], dtype=torch.float32),
            "global_orient": torch.tensor(
                hand_data["global_orient"], dtype=torch.float32
            ),
            "hand_pose": torch.tensor(hand_data["hand_pose"], dtype=torch.float32),
            "valid": hand_data["valid"],
        }

        joints_dict = {
            "joints_3d": joints_mapped,
            "vertices_3d": vertices,
        }

        return hand_pose_dict, joints_dict

    def _extract_mocap_joints(self, mocap_data: dict, hand_prefix: str):
        """Extract 16 joints from mocap annotation."""
        key = f"{hand_prefix}_joints_3d"
        joints = None
        if key in mocap_data and mocap_data[key] is not None:
            joints = np.array(mocap_data[key], dtype=np.float32)
        elif hand_prefix in mocap_data and isinstance(mocap_data[hand_prefix], dict):
            hand_dict = mocap_data[hand_prefix]
            if "hand_landmarks" in hand_dict:
                joints = np.array(hand_dict["hand_landmarks"], dtype=np.float32)
            elif "joint_points" in hand_dict:
                joints = np.array(hand_dict["joint_points"], dtype=np.float32)
            elif "joints_3d" in hand_dict:
                joints = np.array(hand_dict["joints_3d"], dtype=np.float32)

        if joints is None:
            return None
        if joints.ndim != 2 or joints.shape[0] != NUM_MOCAP_JOINTS or joints.shape[1] != 3:
            return None
        # Convert mm -> meters when likely needed.
        if np.max(np.abs(joints)) > 1.0:
            joints = joints / 1000.0
        return joints

    def _process_mocap_hand_data(self, mocap_data: dict, hand_prefix: str):
        """Process mocap hand annotation to tensor format."""
        joints_np = self._extract_mocap_joints(mocap_data, hand_prefix)
        if joints_np is None:
            return self._get_default_hand_data(), self._get_default_joints_data("mocap16")

        hand_pose_dict = {
            "shape": torch.zeros(10, dtype=torch.float32),
            "trans": torch.zeros(3, dtype=torch.float32),
            "global_orient": torch.zeros(3, dtype=torch.float32),
            "hand_pose": torch.zeros(15, dtype=torch.float32),
            "valid": True,
        }
        joints_dict = {
            "joints_3d": torch.tensor(joints_np, dtype=torch.float32),
            "vertices_3d": torch.zeros(NUM_MANO_VERTICES, 3, dtype=torch.float32),
        }
        return hand_pose_dict, joints_dict

    @classmethod
    def to_device(cls, v, device):
        """Move data to device. Keep 'events' on CPU (for PointNet++)."""
        if isinstance(v, torch.Tensor):
            return v.to(device)
        elif isinstance(v, dict):
            nv = {}
            for k, _v in v.items():
                if k == "events":
                    nv[k] = _v
                else:
                    nv[k] = cls.to_device(_v, device)
            return nv
        elif isinstance(v, (list, tuple)):
            return [cls.to_device(item, device) for item in v]
        elif isinstance(v, (str, float, int, bool, type(None))):
            return v
        elif isinstance(v, np.ndarray):
            return v
        else:
            raise TypeError(f"Type {type(v)} is not supported: {v}")

    def __getitem__(self, idx):
        (
            sequence_id,
            annotation_frame_id,
            annotation_path,
            event_frame_idx,
            gt_label_path,
            sub_frame_idx,
        ) = self.frame_list[idx]

        # Load LNES and create event_frame from it
        data_dir = self._get_data_dir(sequence_id)
        window_events = None
        if self.lnes_from_h5:
            # Rebuild the LNES for this sub-frame from the raw events. The stored
            # PNGs only exist at the base 30fps rate, so sub-frames 1..N-1 have no
            # pre-rendered counterpart.
            window = self._load_window_events(
                data_dir, event_frame_idx, sub_frame_idx
            )
            if window is None:
                lnes_np = np.zeros(
                    (LNES_CROP_SIZE, LNES_CROP_SIZE, 2), dtype=np.uint8
                )
            else:
                window_events, t_start, t_end = window
                lnes_np = self._build_lnes_from_events(window_events, t_start, t_end)
        else:
            lnes_np = self._load_lnes_frame(data_dir, event_frame_idx)
        event_frame_np = self._convert_lnes_to_3channel(lnes_np)

        # Apply transform
        event_frame = self.transform_3channel(event_frame_np)

        # Transform LNES 2-channel if requested
        lnes = self.transform_2channel(lnes_np)  # (2, H, W) tensor

        if self.annotation_source == "mocap":
            mocap_data = self._load_mocap_params(annotation_path)
            left_hand, left_joints = self._process_mocap_hand_data(mocap_data, "left")
            right_hand, right_joints = self._process_mocap_hand_data(mocap_data, "right")
            annotation_format = "mocap16"
        else:
            mano_data = self._load_mano_params(annotation_path)
            # Process hand data (returns tuple of (hand_pose_dict, joints_dict))
            left_hand, left_joints = self._process_hand_data(
                mano_data.get("left_hand"), "left_hand"
            )
            right_hand, right_joints = self._process_hand_data(
                mano_data.get("right_hand"), "right_hand"
            )
            annotation_format = "mano20"

        # Build handedness tensor
        left_valid = 1 if left_hand.get("valid") else 0
        right_valid = 1 if right_hand.get("valid") else 0
        handedness = torch.tensor([left_valid, right_valid], dtype=torch.int32)

        # Build gt_hand_poses dict (compatible with CustomLoss)
        gt_hand_poses = {
            "left": left_hand,
            "right": right_hand,
            "handedness": handedness,
        }

        # Build gt_joints dict (always provide both hands for CustomLoss)
        gt_joints = {
            "left": left_joints,
            "right": right_joints,
        }

        # Load ev2hands events if requested (same time window as the LNES above)
        ev2hands_events = None
        if self.load_h5:
            if self.lnes_from_h5:
                ev2hands_events = self._events_to_pointcloud(window_events)
            else:
                ev2hands_events = self._load_ev2hands_events(data_dir, event_frame_idx)

        # Build result dict
        image_path = str(
            data_dir / "event_frames" / f"event_frame_{event_frame_idx:010d}.jpg"
        )
        result = {
            "event_frame": event_frame,
            "lnes": lnes,
            "gt_hand_poses": gt_hand_poses,
            "gt_joints": gt_joints,
            "image_file_path": image_path,

            "sequence_name": sequence_id,
            "frame_number": str(annotation_frame_id),
            "event_frame_idx": event_frame_idx,
            "sub_frame_idx": sub_frame_idx,
            "annotation_format": annotation_format,
            "gt_yolo_label_path": str(gt_label_path) if gt_label_path is not None else "",
        }

        # Load grayscale image if requested
        if self.load_grayscale_image:
            gray_path = str(
                data_dir / "images" / f"frame_{event_frame_idx:010d}.png"
            )
            if os.path.exists(gray_path):
                gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is not None:
                    # Center crop to 260x260
                    gray_img = crop_center(gray_img, 260)
                    result["grayscale_image"] = gray_img
                    result["grayscale_image_path"] = gray_path

        if ev2hands_events is not None:
            result["events"] = ev2hands_events

        # Optionally include raw annotations
        if self.return_raw_annotations:
            if self.annotation_source == "mocap":
                result["raw_mocap_params"] = mocap_data
            else:
                result["raw_mano_params"] = mano_data

        return result


if __name__ == "__main__":
    # Example usage and verification
    print("Testing EEHRDataset...")

    dataset = EEHRDataset(
        dataset_root="/path/to/EEH-R",
        split="valid",
        fps=30,
        output_size=(224, 224),
        use_center_crop=True,
        ignore_files_count=50,
        return_raw_annotations=True,
    )

    print(f"\nDataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Event frame shape: {sample['event_frame'].shape}")
        print(f"Event frame dtype: {sample['event_frame'].dtype}")
        print(
            f"Event frame range: [{sample['event_frame'].min():.4f}, "
            f"{sample['event_frame'].max():.4f}]"
        )

        # Check LNES
        if sample["lnes"] is not None:
            print(f"LNES shape: {sample['lnes'].shape}")
            print(f"LNES dtype: {sample['lnes'].dtype}")
            print(
                f"LNES range: [{sample['lnes'].min():.4f}, "
                f"{sample['lnes'].max():.4f}]"
            )
        else:
            print("LNES: None")

        print(f"Sequence name: {sample['sequence_name']}")
        print(f"Frame number: {sample['frame_number']}")
        print(f"Image file path: {sample['image_file_path']}")

        # Check gt_hand_poses
        gt_poses = sample["gt_hand_poses"]
        print(f"\ngt_hand_poses keys: {gt_poses.keys()}")
        print(f"Handedness: {gt_poses['handedness']}")

        for hand_key in ["left", "right"]:
            hand = gt_poses[hand_key]
            print(f"\n{hand_key} hand (valid={hand['valid']}):")
            print(f"  shape (betas): {hand['shape'].shape}")
            print(f"  trans: {hand['trans'].shape}")
            print(f"  global_orient: {hand['global_orient'].shape}")
            print(f"  hand_pose: {hand['hand_pose'].shape}")

        # Check gt_joints
        gt_joints = sample["gt_joints"]
        print(f"\ngt_joints keys: {gt_joints.keys()}")
        for key in ["left", "right"]:
            print(f"  {key} joints_3d: {gt_joints[key]['joints_3d'].shape}")
            print(f"  {key} vertices_3d: {gt_joints[key]['vertices_3d'].shape}")

        # Check raw annotations
        if sample.get("raw_mano_params"):
            print(f"\nraw_mano_params keys: {sample['raw_mano_params'].keys()}")

    print("\nTest completed!")
