import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import json
import random
from PIL import Image
from torchvision.transforms import v2
import blosc2

from hot3d.hot3d.data_loaders.loader_object_library import load_object_library
from hot3d.hot3d.data_loaders.mano_layer import MANOHandModel
from hot3d.hot3d.data_loaders.pytorch3d_rotation.rotation_conversions import (
    quaternion_to_axis_angle,
)

from utils.representation import (
    get_v2_transform,
)

ORIGINAL_WIDTH = 346
ORIGINAL_HEIGHT = 260


class NHOT3DYOLODataset(Dataset):
    """
    Dataset class that loads event frames from YOLO format images
    and ground truth from NHOT3D format.

    YOLO image filename format: P0001_15c4300c_frame_0000000001.png
    -> sequence: P0001_15c4300c, frame_number: 0000000001
    """

    def __init__(
        self,
        yolo_image_dir: str,
        gt_file_dir: str,
        object_library_path: str,
        mano_hand_model_path: str,
        is_get_image: bool = False,
        rgb_dir: str = "",
        mode: str = "",
        output_size: tuple = (224, 224),
        use_center_crop: bool = True,
        augmentation: bool = False,
        ignore_files_count: int = 50,
        dataset_divide: int = 1,
        lnes_blosc2_dir: str = "",
    ):
        self.yolo_image_dir = yolo_image_dir
        self.gt_file_dir = gt_file_dir
        self.rgb_dir = rgb_dir
        self.is_get_image = is_get_image
        self.mode = mode
        self.output_size = output_size
        self.use_center_crop = use_center_crop
        self.augmentation = augmentation
        self.ignore_files_count = ignore_files_count
        self.dataset_divide = dataset_divide
        self.lnes_blosc2_dir = lnes_blosc2_dir

        # Get all YOLO image files
        all_image_files = self._get_all_image_files()
        print(f"Found {len(all_image_files)} YOLO image files")

        # Count files per sequence (for ignore_files_count filtering)
        self.sequence_file_counts = self._count_files_per_sequence(all_image_files)
        print(f"Found {len(self.sequence_file_counts)} sequences")

        # Build mapping from image files to GT files (only valid ones)
        self.image_to_gt_mapping, self.image_files = self._build_gt_mapping(
            all_image_files
        )
        print(f"Valid image files with GT: {len(self.image_files)}")

        # Apply dataset_divide to randomly sample subset of data
        if self.dataset_divide > 1:
            original_length = len(self.image_files)
            target_length = original_length // self.dataset_divide
            self.image_files = random.sample(self.image_files, target_length)
            print(
                f"Dataset reduced by divide={self.dataset_divide}: {original_length} -> {len(self.image_files)}"
            )

        # Hand model
        self.object_library = load_object_library(
            object_library_folderpath=object_library_path
        )
        self.mano_hand_model = MANOHandModel(mano_hand_model_path)

        # Transforms
        self.transform_3channel = get_v2_transform(
            self.output_size, self.use_center_crop, normalize=False
        )
        self.transform_2channel = get_v2_transform(
            self.output_size, self.use_center_crop, normalize=False
        )

        print(f"Dataset length: {len(self)}")

    def __len__(self):
        return len(self.image_files)

    def _get_all_image_files(self):
        """Get all PNG image files from YOLO image directory."""
        all_files = []
        for file in os.listdir(self.yolo_image_dir):
            if file.endswith(".png") or file.endswith(".jpg"):
                all_files.append(os.path.join(self.yolo_image_dir, file))
        all_files.sort()
        return all_files

    def _count_files_per_sequence(self, all_image_files: list) -> dict:
        """
        Count number of files per sequence for ignore_files_count filtering.

        Returns:
            dict: {sequence_name: file_count}
        """
        sequence_counts = {}
        for image_file in all_image_files:
            try:
                sequence_name, _ = self._parse_yolo_filename(image_file)
                if sequence_name not in sequence_counts:
                    sequence_counts[sequence_name] = 0
                sequence_counts[sequence_name] += 1
            except ValueError:
                continue
        return sequence_counts

    def _parse_yolo_filename(self, filename: str) -> tuple:
        """
        Parse YOLO image filename to extract sequence name and frame number.

        Args:
            filename: e.g., "P0001_15c4300c_frame_0000000001.png"

        Returns:
            tuple: (sequence_name, frame_number_str)
                   e.g., ("P0001_15c4300c", "0000000001")
        """
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]

        # Split by "_frame_"
        parts = name_without_ext.split("_frame_")
        if len(parts) != 2:
            raise ValueError(f"Invalid YOLO filename format: {filename}")

        sequence_name = parts[0]
        # Keep frame number as 10 digits
        frame_number_str = parts[1]

        return sequence_name, frame_number_str

    def _build_gt_mapping(self, all_image_files: list):
        """Build mapping from image files to GT file paths.
        Only includes entries where GT files exist.
        Filters out files at the beginning and end of each sequence based on ignore_files_count.
        """
        mapping = {}
        valid_image_files = []

        for image_file in all_image_files:
            try:
                sequence_name, frame_number_str = self._parse_yolo_filename(image_file)

                # Apply ignore_files_count filtering
                frame_int = int(frame_number_str)
                sequence_file_count = self.sequence_file_counts.get(sequence_name, 0)

                # Skip files at the beginning and end of sequence
                if frame_int < self.ignore_files_count or frame_int > (
                    sequence_file_count - self.ignore_files_count
                ):
                    continue

                # Build GT file paths with correct structure
                sequence_dir = os.path.join(self.gt_file_dir, sequence_name)

                # GT file is in 'gt' subdirectory
                gt_file = os.path.join(
                    sequence_dir, "gt", f"gt_{frame_number_str}.jsonl"
                )
                # Joints file is in 'joints_gt' subdirectory
                joints_file = os.path.join(
                    sequence_dir, "joints_gt", f"joints_{frame_number_str}.jsonl"
                )
                mask_left_file = os.path.join(
                    sequence_dir,
                    "gt_hand_mask",
                    f"gt_{frame_number_str}_left.jpg",
                )
                mask_right_file = os.path.join(
                    sequence_dir,
                    "gt_hand_mask",
                    f"gt_{frame_number_str}_right.jpg",
                )

                # Only add if GT file exists
                if not os.path.exists(gt_file):
                    continue

                mapping[image_file] = {
                    "sequence_name": sequence_name,
                    "frame_number": frame_number_str,
                    "gt_file": gt_file,
                    "joints_file": joints_file,
                    "mask_left_file": mask_left_file,
                    "mask_right_file": mask_right_file,
                }
                valid_image_files.append(image_file)

            except ValueError as e:
                print(f"Warning: {e}")
                continue

        return mapping, valid_image_files

    def load_event_frame(self, image_path: str):
        """Load event frame from YOLO image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_lnes_from_blosc2(self, image_file: str) -> np.ndarray:
        """Load LNES data from blosc2 file.

        Args:
            image_file: Path to the YOLO image file (used to derive blosc2 path)

        Returns:
            lnes: (H, W, 2) numpy array or None if file not found
        """
        basename = os.path.basename(image_file)
        name_without_ext = os.path.splitext(basename)[0]
        blosc2_filename = f"{name_without_ext}.bl2"

        blosc2_path = os.path.join(self.lnes_blosc2_dir, blosc2_filename)

        if not os.path.exists(blosc2_path):
            return None
        lnes = blosc2.load_array(blosc2_path)  # (H, W, 2)
        return lnes

    def get_hand_pose_from_gt_file(
        self, gt_file: str, frame_number_str: str, sequence_name: str = ""
    ):
        """Load hand pose ground truth from GT file."""
        if not os.path.exists(gt_file):
            return None

        with open(gt_file, "r") as f:
            gt_hand_poses = json.load(f)

        for key in ["left", "right"]:
            if gt_hand_poses.get(key) is None:
                continue
            hand_pose = gt_hand_poses[key]
            global_orient = quaternion_to_axis_angle(torch.tensor(hand_pose["quat"]))
            gt_hand_poses[key] = {
                "shape": torch.tensor(hand_pose["betas"], dtype=torch.float32),
                "quat": torch.tensor(hand_pose["quat"], dtype=torch.float32),
                "trans": torch.tensor(hand_pose["translation"], dtype=torch.float32),
                "global_orient": global_orient,
                "hand_pose": torch.tensor(
                    hand_pose["joint_angles"], dtype=torch.float32
                ),
                "vertices": torch.tensor(hand_pose["vertices"], dtype=torch.float32),
                "triangles": torch.tensor(hand_pose["triangles"], dtype=torch.int32),
                "vertex_normals": torch.tensor(
                    hand_pose["vertex_normals"], dtype=torch.float32
                ),
                "hand_landmarks": torch.tensor(
                    hand_pose["hand_landmarks"], dtype=torch.float32
                ),
                "joint_points": hand_pose["joint_points"],
                "valid": hand_pose["valid"],
            }

        handedness = [1, 1]
        if gt_hand_poses.get("left") is None or not gt_hand_poses.get("left").get(
            "valid"
        ):
            handedness[0] = 0
        if gt_hand_poses.get("right") is None or not gt_hand_poses.get("right").get(
            "valid"
        ):
            handedness[1] = 0
        gt_hand_poses["handedness"] = torch.tensor(handedness, dtype=torch.int32)

        if self.is_get_image and self.rgb_dir:
            image_data = self._get_rgb_image(sequence_name, frame_number_str)
            if image_data is not None:
                gt_hand_poses["image"] = image_data
            else:
                # Create dummy image with same shape as transformed output
                gt_hand_poses["image"] = np.zeros(
                    (self.output_size[0], self.output_size[1], 3), dtype=np.uint8
                )

        return gt_hand_poses

    def _get_rgb_image(self, sequence_name: str, frame_number_str: str):
        """Get RGB image corresponding to the frame."""
        if not self.rgb_dir:
            return None
        rgb_image_path = os.path.join(
            self.rgb_dir, sequence_name, "rgb_images", f"rgb_{frame_number_str}.jpg"
        )
        if not os.path.exists(rgb_image_path):
            return None
        image = cv2.imread(rgb_image_path)
        if image is None:
            return None

        # Apply the same transformation as event_frame
        image_transformed = self.transform_3channel(image)

        # Convert back to numpy array in uint8 format for visualization
        # transform_3channel outputs [C, H, W] tensor in [0, 1] range
        image_np = image_transformed.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)

        return image_np

    def get_joints_from_gt_file(self, joints_file: str):
        """Load joints ground truth from joints file."""
        if not os.path.exists(joints_file):
            return None

        with open(joints_file, "r") as f:
            joints_data = json.load(f)

        joints_data_res = {}
        for key in ["left", "right"]:
            if joints_data.get(key) is None:
                continue
            hand_joints = joints_data[key]

            # Load joints_2d (originally in 512x512 coordinate system)
            joints_2d_original = np.array(
                hand_joints["joints_2d" if key == "left" else "joints2d"],
                dtype=np.float32,
            )

            # Apply same transformation as event_frame to match coordinate systems
            # Original: 512x512 -> Resize to (346, 346) -> CenterCrop to output_size
            joints_2d_transformed = self._transform_joints_2d(joints_2d_original)

            joints_data_res[key] = {
                "joints_3d": torch.tensor(
                    hand_joints["joints_3d"], dtype=torch.float32
                ),
                "joints_2d": torch.tensor(joints_2d_transformed, dtype=torch.float32),
                "vertices_3d": torch.tensor(
                    hand_joints["vertices_3d"], dtype=torch.float32
                ),
            }

        return joints_data_res

    def _transform_joints_2d(self, joints_2d: np.ndarray) -> np.ndarray:
        """
        Transform joints_2d from 512x512 coordinate system to match event_frame transformation.

        Applies the same transformation as get_v2_transform:
        1. Resize from 512x512 to (346, 346)
        2. CenterCrop to output_size

        Args:
            joints_2d: [N, 2] array in 512x512 coordinate system

        Returns:
            joints_2d_transformed: [N, 2] array in output_size coordinate system
        """
        if self.use_center_crop:
            # Step 1: Resize from 512x512 to (346, 346)
            scale_factor = ORIGINAL_WIDTH / 512.0  # 346 / 512
            joints_2d_resized = joints_2d * scale_factor

            # Step 2: CenterCrop from (346, 346) to output_size
            crop_h, crop_w = self.output_size
            crop_offset_x = (ORIGINAL_WIDTH - crop_w) / 2.0
            crop_offset_y = (ORIGINAL_WIDTH - crop_h) / 2.0

            joints_2d_transformed = joints_2d_resized.copy()
            joints_2d_transformed[:, 0] -= crop_offset_x
            joints_2d_transformed[:, 1] -= crop_offset_y
        else:
            # Direct resize from 512x512 to output_size
            scale_x = self.output_size[1] / 512.0
            scale_y = self.output_size[0] / 512.0
            joints_2d_transformed = joints_2d.copy()
            joints_2d_transformed[:, 0] *= scale_x
            joints_2d_transformed[:, 1] *= scale_y

        return joints_2d_transformed

    def load_mask(self, mask_file: str):
        """Load mask from file."""
        if not os.path.exists(mask_file):
            return np.zeros((self.output_size[0], self.output_size[1]), dtype=np.uint8)

        mask_image = Image.open(mask_file).convert("L")
        mask_image = v2.ToImage()(mask_image)
        mask_image = v2.Resize((ORIGINAL_WIDTH, ORIGINAL_WIDTH))(mask_image)
        if self.use_center_crop:
            mask_image = v2.CenterCrop(self.output_size)(mask_image)
        else:
            mask_image = mask_image.resize((self.output_size[1], self.output_size[0]))
        mask_image = np.array(mask_image)
        mask_image = mask_image.squeeze(0)
        return mask_image

    def mix_mask(self, mask_left, mask_right):
        """Mix left and right hand masks."""
        mix_mask = np.maximum(mask_left, mask_right)
        if mix_mask is None:
            mix_mask = np.zeros(
                (self.output_size[0], self.output_size[1]), dtype=np.uint8
            )
        return mix_mask

    @classmethod
    def to_device(cls, v, device):
        """Move data to device."""
        if isinstance(v, torch.Tensor):
            nv = v.to(device)
        elif isinstance(v, dict):
            nv = dict()
            for _k, _v in v.items():
                nv[_k] = cls.to_device(_v, device)
        elif isinstance(v, (list, tuple)):
            nv = list()
            for i in range(0, len(v)):
                nv.append(cls.to_device(v[i], device))
        elif isinstance(v, str):
            nv = v
        elif isinstance(v, float):
            nv = v
        elif isinstance(v, int):
            nv = v
        elif isinstance(v, np.ndarray):
            nv = v
        else:
            raise TypeError(f"Type {type(v)} is not supported {v}")
        return nv

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mapping = self.image_to_gt_mapping[image_file]

        # Load LNES from blosc2 if directory is specified
        lnes = None
        if self.lnes_blosc2_dir:
            lnes_np = self.load_lnes_from_blosc2(image_file)
            if lnes_np is None:
                # Skip to next valid index if blosc2 file not found
                return self.__getitem__((idx + 1) % len(self))
            lnes = self.transform_2channel(lnes_np)  # (2, H, W) tensor

        # Load event frame from YOLO image
        event_frame_np = self.load_event_frame(image_file)

        # Apply transform
        event_frame = self.transform_3channel(event_frame_np)

        # Load GT
        gt_hand_poses = self.get_hand_pose_from_gt_file(
            mapping["gt_file"], mapping["frame_number"], mapping["sequence_name"]
        )
        gt_joints = self.get_joints_from_gt_file(mapping["joints_file"])

        # Load masks if needed
        if self.mode == "mask" or self.mode == "multi_mask":
            mask_left = self.load_mask(mapping["mask_left_file"])
            mask_right = self.load_mask(mapping["mask_right_file"])

            if self.mode == "mask":
                mix_mask = self.mix_mask(mask_left, mask_right)
                mix_mask = mix_mask / 255.0
                mix_mask = np.clip(mix_mask, 0, 1)
                mix_mask = torch.tensor(mix_mask, dtype=torch.float32).unsqueeze(0)
                event_frame = event_frame * mix_mask
            elif self.mode == "multi_mask":
                mask_left = np.clip(mask_left.astype(np.float32) / 255.0, 0.0, 1.0)
                mask_right = np.clip(mask_right.astype(np.float32) / 255.0, 0.0, 1.0)
                event_frame_left = event_frame * mask_left
                event_frame_right = event_frame * mask_right

        data = {
            "mano_gt": 1.0,
            "event_frame": event_frame,
            "lnes": lnes,
            "gt_hand_poses": gt_hand_poses,
            "gt_joints": gt_joints,
            "image_file_path": image_file,
            "sequence_name": mapping["sequence_name"],
            "frame_number": mapping["frame_number"],
        }

        if self.mode == "mask":
            data["mask"] = mix_mask
        elif self.mode == "multi_mask":
            data["mask_left"] = mask_left
            data["mask_right"] = mask_right
            data["event_frame_left"] = event_frame_left
            data["event_frame_right"] = event_frame_right

        return data


if __name__ == "__main__":

    # Example usage
    dataset = NHOT3DYOLODataset(
        yolo_image_dir="/path/to/NHOT3D_yolo_event_frame/images/train",
        gt_file_dir="/path/to/N-HOT3D/Aria/train",
        object_library_path="/path/to/N-HOT3D/assets",
        mano_hand_model_path="src/mano/models",
        is_get_image=False,
        output_size=(224, 224),
        mode="multi_mask",
        use_center_crop=True,
    )
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        data = dataset[0]
        print(f"Data keys: {data.keys()}")
        print(f"Event frame shape: {data['event_frame'].shape}")
        if data.get("gt_hand_poses"):
            print(f"GT hand poses keys: {data['gt_hand_poses'].keys()}")
