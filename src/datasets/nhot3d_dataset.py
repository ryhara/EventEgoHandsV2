import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
import numpy as np
import cv2
import h5py
import json
import random
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from PIL import Image

from utils import  mask_event_cloud_one, bbox_event_one

from hot3d.hot3d.dataset_api import Hot3dDataProvider
from hot3d.hot3d.data_loaders.loader_object_library import load_object_library
from hot3d.hot3d.data_loaders.mano_layer import MANOHandModel
from hot3d.hot3d.data_loaders.hand_common import LANDMARK_CONNECTIVITY
from hot3d.hot3d.data_loaders.pytorch3d_rotation.rotation_conversions import (
    quaternion_to_axis_angle,
)

from projectaria_tools.core import data_provider, calibration

OUTPUT_WIDTH = 346
OUTPUT_HEIGHT = 260


class NHOT3DDataset(Dataset):
    def __init__(
        self,
        event_input_dir: str,
        gt_file_dir: str,
        object_library_path: str,
        mano_hand_model_path: str,
        is_get_image: bool = False,
        augmentation: bool = False,
        is_normalize: bool = True,
        is_rotate: bool = False,
        mode: str = "",
        output_size: tuple = (OUTPUT_HEIGHT, OUTPUT_WIDTH),
        event_number_threshold: int = 2048,
        time_bin: int = 1,
        is_sequence: bool = False,
        dataset_divide: int = 1,
        lnes_version: int = 1,
        no_resample: bool = False,
    ):
        self.event_input_dir = event_input_dir
        self.is_get_image = is_get_image
        self.augmentation = augmentation
        self.is_normalize = is_normalize
        self.is_rotate = is_rotate
        self.mode = mode
        self.output_size = output_size
        self.event_number_threshold = event_number_threshold
        self.time_bin = time_bin
        self.is_sequence = is_sequence
        self.dataset_divide = dataset_divide
        self.lnes_version = lnes_version  # 1: original, 2: window-based (generate_lnes_from_h5 style)
        self.no_resample = no_resample

        # subdirs
        self.sub_dirs = [
            os.path.join(gt_file_dir, dir)
            for dir in os.listdir(gt_file_dir)
            if os.path.isdir(os.path.join(gt_file_dir, dir))
        ]
        self.sub_dirs.sort()

        self.dataset_len = int(self.count_all_event_files(event_input_dir) / self.dataset_divide)
        self.data_len_per_subdir = int(self.dataset_len / len(self.sub_dirs))

        # event files
        self.event_input_files = self.get_all_event_files()

        # Hand gt
        self.object_library = load_object_library(
            object_library_folderpath=object_library_path
        )
        self.mano_hand_model = MANOHandModel(mano_hand_model_path)
        self.gt_files = self.get_all_gt_files()

        assert (
            len(self.event_input_files) == len(self.gt_files)
        ), f"len(event_input_files): {len(self.event_input_files)}, len(gt_files): {len(self.gt_files)}"

        # Hand mask gt
        self.mask_files = self.get_all_mask_files()

        assert (
            len(self.event_input_files) * 2 == len(self.mask_files)
        ), f"len(event_input_files) * 2 != len(mask_files) : {len(self.event_input_files) * 2} != {len(self.mask_files)}"


        # Joints
        self.joints_files = self.get_all_joints_files()

        assert (
            len(self.event_input_files) == len(self.joints_files)
        ), f"len(event_input_files) != len(joints_files) : {len(self.event_input_files)} != {len(self.joints_files)}"


        # device pose gt
        self.device_pose_gt_files = self.get_all_device_pose_gt_files()

        # Hand Bbox
        # self.bbox_files = self.get_all_bbox_files(gt_file_dir)
        # self.bbox_files.sort()

        # assert (
        #     len(self.event_input_files) == len(self.bbox_files)
        # ), f"len(event_input_files) != len(bbox_files) : {len(self.event_input_files)} != {len(self.bbox_files)}"

        # Define transforms
        self.transform_2 = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
            ]
        )
        self.transform_3 = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(self.output_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        print(f"Dataset length: {len(self)}")

    def __len__(self):
        return len(self.event_input_files)

    def get_index_from_event_file(self, event_file):
        return self.event_input_files.index(event_file)

    def count_all_event_files(self, directory):
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith("events_") and file.endswith(".h5") and "data" not in file:
                    file_number = file.split("_")[1].split(".")[0]
                    file_int = int(file_number)
                    if file_int < 50:
                        continue
                    all_files.append(os.path.join(root, file))
        return len(all_files)

    def get_all_event_files(self):
        all_files = []
        for sub_dir in self.sub_dirs:
            subdir_files = []
            for root, _, files in os.walk(sub_dir):
                for file in files:
                    if file.startswith("events_") and file.endswith(".h5") and "data" not in file:
                        file_number = file.split("_")[1].split(".")[0]
                        file_int = int(file_number)
                        if file_int < 50:
                            continue
                        subdir_files.append(os.path.join(root, file))
            subdir_files.sort()
            selected_files = subdir_files[:self.data_len_per_subdir]
            all_files.extend(selected_files)
        all_files.sort()
        return all_files

    def get_all_gt_files(self):
        all_files = []
        for sub_dir in self.sub_dirs:
            subdir_files = []
            for root, _, files in os.walk(sub_dir):
                for file in files:
                    if file.startswith("gt_") and file.endswith(".jsonl"):
                        file_number = file.split("_")[1].split(".")[0]
                        file_int = int(file_number)
                        if file_int < 50:
                            continue
                        subdir_files.append(os.path.join(root, file))
            subdir_files.sort()
            selected_files = subdir_files[:self.data_len_per_subdir]
            all_files.extend(selected_files)
        all_files.sort()
        return all_files

    def get_all_mask_files(self):
        all_files = []
        for sub_dir in self.sub_dirs:
            subdir_files = []
            for root, _, files in os.walk(sub_dir):
                for file in files:
                    if file.startswith("gt_") and (
                        file.endswith("left.jpg") or file.endswith("right.jpg")
                    ) and "gt_hand_mask" in root:
                        file_number = file.split("_")[1]
                        file_int = int(file_number)
                        if file_int < 50:
                            continue
                        subdir_files.append(os.path.join(root, file))
            subdir_files.sort()
            selected_files = subdir_files[:self.data_len_per_subdir * 2]
            all_files.extend(selected_files)
        all_files.sort()
        return all_files

    def get_all_bbox_files(self):
        all_files = []
        for sub_dir in self.sub_dirs:
            subdir_files = []
            for root, _, files in os.walk(sub_dir):
                for file in files:
                    if file.startswith("bbox_gt_") and file.endswith(".jsonl"):
                        file_number = file.split("_")[2].split(".")[0]
                        file_int = int(file_number)
                        if file_int < 50:
                            continue
                        subdir_files.append(os.path.join(root, file))
            subdir_files.sort()
            selected_files = subdir_files[:self.data_len_per_subdir]
            all_files.extend(selected_files)
        all_files.sort()
        return all_files

    def get_all_device_pose_gt_files(self):
        all_files = []
        for sub_dir in self.sub_dirs:
            subdir_files = []
            for root, _, files in os.walk(sub_dir):
                for file in files:
                    if file.startswith("device_pose_") and file.endswith(".jsonl"):
                        file_number = file.split("_")[2].split(".")[0]
                        file_int = int(file_number)
                        if file_int < 50:
                            continue
                        subdir_files.append(os.path.join(root, file))
            subdir_files.sort()
            selected_files = subdir_files[:self.data_len_per_subdir]
            all_files.extend(selected_files)
        all_files.sort()
        return all_files

    def get_all_joints_files(self):
        all_files = []
        for sub_dir in self.sub_dirs:
            subdir_files = []
            for root, _, files in os.walk(sub_dir):
                for file in files:
                    if file.startswith("joints_") and file.endswith(".jsonl"):
                        file_number = file.split("_")[1].split(".")[0]
                        file_int = int(file_number)
                        if file_int < 50:
                            continue
                        subdir_files.append(os.path.join(root, file))
            subdir_files.sort()
            selected_files = subdir_files[:self.data_len_per_subdir]
            all_files.extend(selected_files)
        all_files.sort()
        return all_files

    def get_events_from_h5(self, h5_file):
        # start_time = time.perf_counter()
        data = h5py.File(h5_file, "r")
        events = data["events"]
        events = np.array(events, np.float32)
        data.close()
        # print(f"get_events_from_h5 time: {time.perf_counter() - start_time}")
        return events

    def create_event_frame(self, events):
        # start_time = time.perf_counter()
        event_frame = np.zeros(
            (self.output_size[0], self.output_size[1], 3), dtype=np.uint8
        )
        scale_x = self.output_size[1] / OUTPUT_WIDTH  # original width is 346
        scale_y = self.output_size[0] / OUTPUT_HEIGHT  # original height is 260
        x = (events[:, 1] * scale_x).astype(int)
        y = (events[:, 2] * scale_y).astype(int)

        valid_indices = (x >= 0) & (x < self.output_size[1]) & (y >= 0) & (y < self.output_size[0])
        x = x[valid_indices]
        y = y[valid_indices]
        p = events[:, 3][valid_indices]

        event_frame[y[p == 1], x[p == 1], 1] = 255
        event_frame[y[p == 0], x[p == 0], -1] = 255
        # print(f"create_event_frame time: {time.perf_counter() - start_time}")
        return event_frame

    def increase_event(self, events):
        new_events = list(events)
        event_len = len(events)
        time_offset = events[-1][0] - events[0][0]

        num_new_events = self.event_number_threshold - event_len

        if num_new_events > 0:
            random_indices = np.random.randint(0, event_len, size=num_new_events)
            random_events = np.array(events)[random_indices]
            new_events_array = random_events.copy()

            new_events_array[:, 0] += time_offset
            new_events.extend(new_events_array.tolist())

        return np.array(new_events)


    def sample_event(self, events):
        new_events = np.array(events)
        random_indices = np.random.choice(
            len(events), self.event_number_threshold, replace=False
        )
        new_events = new_events[random_indices]
        return new_events

    def event_normalize(self, events):
        t_min = events[:, 0].min()
        t_max = events[:, 0].max()
        t_diff = t_max - t_min
        if t_diff == 0:
            t_diff = t_max

        events[:, 0] = (events[:, 0] - t_min) / t_diff
        events[:, 1] = events[:, 1] / self.output_size[1]
        events[:, 2] = events[:, 2] / self.output_size[0]

        return events

    def get_image(
        self,
        device_data_provider,
        timestamp_ns,
        rgb_image_stream_id,
        sequence_path,
        undistorted=False,
    ):
        image_data = device_data_provider.get_image(timestamp_ns, rgb_image_stream_id)

        if undistorted:
            vrs_path = os.path.join(sequence_path, "recording.vrs")
            vrs_provider = data_provider.create_vrs_data_provider(vrs_path)
            calib = vrs_provider.get_device_calibration().get_camera_calib("camera-rgb")
            pinhole = calibration.get_linear_camera_calibration(512, 512, 150)
            undistorted_image = calibration.distort_by_calibration(
                image_data, pinhole, calib
            )
            undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
            undistorted_image = cv2.rotate(undistorted_image, cv2.ROTATE_90_CLOCKWISE)
            return undistorted_image
        else:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            image_data = cv2.rotate(image_data, cv2.ROTATE_90_CLOCKWISE)
            return image_data

    def convert_to_mesh(self, hand_pose, hand_data_provider):
        hand_mesh_vertices = hand_data_provider.get_hand_mesh_vertices(hand_pose)
        hand_triangles, hand_vertex_normals = (
            hand_data_provider.get_hand_mesh_faces_and_normals(hand_pose)
        )
        return hand_mesh_vertices, hand_triangles, hand_vertex_normals

    def convert_to_joints(self, hand_pose, hand_data_provider):
        hand_landmarks = hand_data_provider.get_hand_landmarks(hand_pose)
        # convert landmarks to connected lines for display
        # (i.e retrieve points along the HAND LANDMARK_CONNECTIVITY as a list)
        points = [
            connections
            for connectivity in LANDMARK_CONNECTIVITY
            for connections in [
                [hand_landmarks[it].numpy().tolist() for it in connectivity]
            ]
        ]

        return hand_landmarks, points

    def convert_from_hand_pose(self, hand_pose):
        handedness = hand_pose.handedness_label()
        # How to Use the wrist pose : https://github.com/facebookresearch/hot3d/blob/main/hot3d/data_loaders/ManoHandDataProvider.py#L62
        # T_world_wrist : https://github.com/facebookresearch/hot3d/blob/main/hot3d/HOT3D_Tutorial.ipynb
        wrist_pose = (
            hand_pose.wrist_pose
        )  # like mano global orientation, <class '_core_pybinds.sophus.SE3'>
        quat_and_translation = wrist_pose.to_quat_and_translation()
        quat = quat_and_translation[0][0:4]
        translation = quat_and_translation[0][4:]
        joint_angles = hand_pose.joint_angles
        return handedness, quat, translation, joint_angles

    def fix_event_distribution(self, events):
        amplitude = 0.005
        noise = np.random.uniform(0, amplitude, len(events))
        new_timestamps = np.linspace(
            events[:, 0].min(), events[:, 0].max(), len(events)
        )
        new_timestamps += noise
        events[:, 0] = new_timestamps
        return events

    def re_getitem(self):
        idx = random.randint(0, len(self) - 1)
        return self.__getitem__(idx)

    def previous_getitem(self, idx):
        return self.__getitem__(idx - 1)

    def get_device_pose_from_gt_file(self, index):
        device_pose_gt_file = self.device_pose_gt_files[index]
        with open(device_pose_gt_file, "r") as f:
            device_pose_gt = json.load(f)
        device_pose_gt_res = {
            "world_camera_pose" : {
                "quat" : torch.tensor(device_pose_gt["world_camera_pose"]["quat"], dtype=torch.float32),
                "translation" : torch.tensor(device_pose_gt["world_camera_pose"]["translation"], dtype=torch.float32)
            },
            "intrinsics" : {
                # "principal_point" : torch.tensor(device_pose_gt["intrinsics"]["principal_point"], dtype=torch.float32),
                # "focal_length" : torch.tensor(device_pose_gt["intrinsics"]["focal_length"], dtype=torch.float32),
                "projection_params" : torch.tensor(device_pose_gt["intrinsics"]["projection_params"], dtype=torch.float32),
                # "T_device_quat" : torch.tensor(device_pose_gt["intrinsics"]["T_Device_Camera"]["quat"], dtype=torch.float32),
                # "T_device_translation" : torch.tensor(device_pose_gt["intrinsics"]["T_Device_Camera"]["translation"], dtype=torch.float32),
            }
        }

        return device_pose_gt_res

    def get_joints_from_gt_file(self, index):
        joints_file = self.joints_files[index]
        # if not os.path.exists(joints_file):
        #     print(f"joints_file: {joints_file} does not exist.")
        #     return None
        with open(joints_file, "r") as f:
            joints_data = json.load(f)

        joints_data_res = {}
        for key in ["left", "right"]:
            if joints_data.get(key) is None:
                continue
            hand_joints = joints_data[key]
            joints_data_res[key] = {
                "joints_3d" : torch.tensor(hand_joints["joints_3d"], dtype=torch.float32),
                "joints_2d" : torch.tensor(hand_joints["joints_2d" if key =="left" else "joints2d"], dtype=torch.float32),
                "vertices_3d" : torch.tensor(hand_joints["vertices_3d"], dtype=torch.float32),
                # "vertices_2d" : torch.tensor(hand_joints["vertices_2d"], dtype=torch.float32),
            }

        return joints_data_res

    def get_hand_pose_from_gt_file(self, event_file, index):
        event_file_dir = os.path.dirname(event_file)
        event_file_number_str = (
            os.path.basename(event_file).split("_")[-1].split(".")[0]
        )
        event_file_number = int(event_file_number_str)
        # gt_dir = os.path.join(event_file_dir, "gt")
        # gt_file = os.path.join(gt_dir, f"gt_{event_file_number_str}.jsonl")
        gt_file = self.gt_files[index]
        # if not os.path.exists(gt_file):
        #     print(f"gt_file: {gt_file} does not exist.")
        #     return None
        with open(gt_file, "r") as f:
            gt_hand_poses = json.load(f)
        for key in ["left", "right"]:
            if gt_hand_poses.get(key) is None:
                continue
            hand_pose = gt_hand_poses[key]
            global_orient = quaternion_to_axis_angle(torch.tensor(hand_pose["quat"]))
            handedness = [0, 0]
            gt_hand_poses[key] = {
                "shape": torch.tensor(
                    hand_pose["betas"], dtype=torch.float32
                ),  # mano shape parameters, <class 'torch.Tensor'> torch.Size([10])
                "quat": torch.tensor(
                    hand_pose["quat"], dtype=torch.float32
                ),  # original <class 'numpy.ndarray'> (4,)
                "trans": torch.tensor(
                    hand_pose["translation"], dtype=torch.float32
                ),  # original <class 'numpy.ndarray'> (3,)　`trans`
                "global_orient": global_orient,  # type: <class 'torch.Tensor'> shape: torch.Size([3])
                "hand_pose": torch.tensor(
                    hand_pose["joint_angles"], dtype=torch.float32
                ),  # mano pose parameters,  <class 'list'> len=15 `hand_pose`
                "vertices": torch.tensor(
                    hand_pose["vertices"], dtype=torch.float32
                ),  # <class 'torch.Tensor'> torch.Size([778, 3])
                "triangles": torch.tensor(
                    hand_pose["triangles"], dtype=torch.int32
                ),  # <class 'numpy.ndarray'> (1538, 3)
                "vertex_normals": torch.tensor(
                    hand_pose["vertex_normals"], dtype=torch.float32
                ),  # <class 'numpy.ndarray'> (778, 3)
                "hand_landmarks": torch.tensor(
                    hand_pose["hand_landmarks"], dtype=torch.float32
                ),  # <class 'torch.Tensor'> torch.Size([20, 3]) real xyz
                "joint_points": hand_pose[
                    "joint_points"
                ],  # <class 'list'> len=23 [[[x,y,z], [x,y,z]], ...] for display
                "valid": hand_pose["valid"],
            }

        handedness = [1, 1]
        if not gt_hand_poses.get("left").get("valid"):
            handedness[0] = 0
        if not gt_hand_poses.get("right").get("valid"):
            handedness[1] = 0
        gt_hand_poses["handedness"] = torch.tensor(handedness, dtype=torch.int32)

        if self.is_get_image:
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=event_file_dir,
                object_library=self.object_library,
                mano_hand_model=self.mano_hand_model,
            )
            device_data_provider = hot3d_data_provider.device_data_provider
            image_stream_ids = device_data_provider.get_image_stream_ids()
            rgb_image_stream_id = image_stream_ids[0]
            timestamps = device_data_provider.get_sequence_timestamps()
            if event_file_number < len(timestamps):
                timestamp_ns = timestamps[event_file_number]
            else:
                timestamp_ns = timestamps[event_file_number - 1]
            image_data = self.get_image(
                device_data_provider,
                timestamp_ns,
                rgb_image_stream_id,
                sequence_path=event_file_dir,
                undistorted=True,
            )
            gt_hand_poses["image"] = image_data  # <class 'numpy.ndarray'> (512, 512, 3)

        return gt_hand_poses

    def get_hand_bbox_from_gt_file(self, index):
        gt_file = self.bbox_files[index]
        # if not os.path.exists(gt_file):
        #     print(f"gt_file: {gt_file} does not exist.")
        #     return None
        with open(gt_file, "r") as f:
            load_data = json.load(f)
        gt_bbox = {}
        for key in ["left", "right"]:
            if load_data.get(key) is None:
                continue
            hand_bbox = load_data[key]

            # resize bbox
            original_width = 512
            original_height = 512

            scale_x = self.output_size[1] / original_width
            scale_y = self.output_size[0] / original_height

            hand_bbox["left"] *= scale_x
            hand_bbox["top"] *= scale_y
            hand_bbox["right"] *= scale_x
            hand_bbox["bottom"] *= scale_y
            hand_bbox["width"] *= scale_x
            hand_bbox["height"] *= scale_y

            gt_bbox[key] = {
                "left": hand_bbox["left"],
                "top": hand_bbox["top"],
                "right": hand_bbox["right"],
                "bottom": hand_bbox["bottom"],
                "width": hand_bbox["width"],
                "height": hand_bbox["height"],
                "visibility_ratio": hand_bbox["visibility_ratio"],
                "is_right": hand_bbox["is_right"],
                "valid" : True
            }

        for key in ["left", "right"]:
            if gt_bbox.get(key) is None:
                gt_bbox[key] = {
                    "left": 0,
                    "top": 0,
                    "right": 0,
                    "bottom": 0,
                    "width": 0,
                    "height": 0,
                    "visibility_ratio": 0,
                    "is_right": False,
                    "valid" : False
                }
        return gt_bbox

    def create_event_cloud(self, row_events):
        # start_time = time.perf_counter()
        t, x, y, p = (
            row_events[:, 0],
            row_events[:, 1],
            row_events[:, 2],
            row_events[:, 3],
        )

        n_evn = (p != 1).astype(np.float32)
        p_evn = (p == 1).astype(np.float32)

        events = np.hstack(
            [x[:, np.newaxis], y[:, np.newaxis], t[:, np.newaxis],  p_evn[:, np.newaxis], n_evn[:, np.newaxis]]
        )

        events = events.astype(dtype=np.float32)

        if self.mode == "all":
            if events.shape[0] < self.event_number_threshold:
                sampled_indices = np.random.choice(
                    events.shape[0], self.event_number_threshold - events.shape[0]
                )
                try:
                    events = np.concatenate([events, events[sampled_indices]], 0)
                except TypeError:
                    return self.re_getitem()
            elif events.shape[0] > self.event_number_threshold:
                try:
                    sampled_indices = np.random.choice(
                        events.shape[0], self.event_number_threshold
                    )
                    events = events[sampled_indices]
                except TypeError:
                    return self.re_getitem()

        events = torch.tensor(events, dtype=torch.float32)

        # print(f"create_event_cloud time: {time.perf_counter() - start_time}")
        return events
    def create_ev2hands_event(self, row_events):
        t, x, y, p = (
            row_events[:, 0],
            row_events[:, 1],
            row_events[:, 2],
            row_events[:, 3],
        )
        event_grid = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.float32)
        count_grid = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.float32)

        x, y = x.astype(dtype=np.int32), y.astype(dtype=np.int32)

        np.add.at(event_grid, (y, x, 0), t)
        np.add.at(event_grid, (y, x, 1), p == 1)
        np.add.at(event_grid, (y, x, 2), p != 1)
        np.add.at(count_grid, (y, x), 1)

        yi, xi = np.nonzero(count_grid)
        t_avg = (event_grid[yi, xi, 0] / count_grid[yi, xi]) # * 1e-6  # ns to ms
        p_evn = event_grid[yi, xi, 1]
        n_evn = event_grid[yi, xi, 2]

        events = np.hstack(
            [
                xi[:, np.newaxis],
                yi[:, np.newaxis],
                t_avg[:, np.newaxis],
                p_evn[:, np.newaxis],
                n_evn[:, np.newaxis],
            ]
        )

        events = events.astype(dtype=np.float32)

        # indices = np.argsort(events[:, 2])
        # events = events[indices]

        if not self.no_resample:
            if events.shape[0] < self.event_number_threshold:
                sampled_indices = np.random.choice(
                    events.shape[0], self.event_number_threshold - events.shape[0]
                )
                try:
                    events = np.concatenate([events, events[sampled_indices]], 0)
                except TypeError:
                    return self.re_getitem()
            elif events.shape[0] > self.event_number_threshold:
                try:
                    sampled_indices = np.random.choice(
                        events.shape[0], self.event_number_threshold
                    )
                    events = events[sampled_indices]
                except TypeError:
                    return self.re_getitem()

        events = torch.tensor(events, dtype=torch.float32)

        # if self.is_normalize:
        #     # events[:, 2] -= events[0, 2]  # normalize ts
        #     events[:, :3] = self.pc_normalize(events[:, :3])

        # if torch.isnan(events).any():
        #     print("nan in events")
        #     exit()

        return events

    def plot_row_events(self, events, output_file="row_events_plot.png"):
        t, x, y, p = (
            events[:, 0],
            events[:, 1],
            events[:, 2],
            events[:, 3],
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[p == 1], y[p == 1], t[p == 1], c="b", s=2, alpha=0.5)
        ax.scatter(x[p != 1], y[p != 1], t[p != 1], c="r", s=2, alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("T")
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=-90, azim=-90)

        plt.savefig(output_file)

    def plot_point_cloud(self, events, output_file="events_plot.png"):
        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        positive = events[:, 3]
        negative = events[:, 4]

        positive_indices = positive > 0
        negative_indices = negative > 0

        positive_x = x[positive_indices]
        positive_y = y[positive_indices]
        positive_t = t[positive_indices]

        negative_x = x[negative_indices]
        negative_y = y[negative_indices]
        negative_t = t[negative_indices]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(positive_x, positive_y, positive_t, c="b", s=2, alpha=0.5)
        ax.scatter(negative_x, negative_y, negative_t, c="r", s=2, alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("T")
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=-90, azim=-90)

        plt.savefig(output_file)

    def event_point_cloud_to_image(self, events):
        event_frame = np.zeros(
            (self.output_size[0], self.output_size[1], 3), dtype=np.uint8
        )
        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        positive = events[:, 3]
        negative = events[:, 4]

        positive_indices = positive > 0
        negative_indices = negative > 0

        positive_x = x[positive_indices]
        positive_y = y[positive_indices]
        positive_t = t[positive_indices]

        negative_x = x[negative_indices]
        negative_y = y[negative_indices]
        negative_t = t[negative_indices]

        for x, y, t in zip(positive_x, positive_y, positive_t):
            x, y = int(x), int(y)
            if x < 0 or x >= self.output_size[1] or y < 0 or y >= self.output_size[0]:
                continue
            event_frame[y, x, 1] = 255
        for x, y, t in zip(negative_x, negative_y, negative_t):
            x, y = int(x), int(y)
            if x < 0 or x >= self.output_size[1] or y < 0 or y >= self.output_size[0]:
                continue
            event_frame[y, x, -1] = 255

        return event_frame

    def pc_normalize(self, pc):
        pc[:, 0] /= OUTPUT_WIDTH
        pc[:, 1] /= OUTPUT_HEIGHT
        # TODO : remove this normalization
        pc[:, :2] = 2 * pc[:, :2] - 1

        ts = pc[:, 2:]

        t_max = ts.max(0).values
        t_min = ts.min(0).values
        diff = t_max - t_min
        if diff == 0:
            diff =  t_max + 1e-6

        ts = 2 * ((ts - t_min) / (diff)) - 1

        pc[:, 2:] = ts

        return pc

    @classmethod
    def to_device(cls, v, device):
        if isinstance(v, torch.Tensor):
            nv = v.to(device)

        elif isinstance(v, dict):
            nv = dict()
            for _k, _v in v.items():
                if _k == "events":
                    nv[_k] = _v
                else:
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

    def load_mask(self, mask_file):
        mask_image = Image.open(mask_file).convert("L")
        mask_image = mask_image.resize((self.output_size[1], self.output_size[0]))
        mask_image = np.array(mask_image)
        return mask_image

    def mix_mask(self, mask_left, mask_right):
        mix_mask = np.maximum(mask_left, mask_right)
        if mix_mask is None:
            mix_mask = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.uint8)
        return mix_mask
    def create_time_surface(self, events, tau=50e-3):
        event_frame = np.zeros(
            (self.output_size[0], self.output_size[1]), dtype=np.float32
        )
        scale_x = self.output_size[1] / OUTPUT_WIDTH  # original width is 346
        scale_y = self.output_size[0] / OUTPUT_HEIGHT  # original height is 260
        x = (events[:, 1] * scale_x).astype(int)
        y = (events[:, 2] * scale_y).astype(int)

        valid_indices = (x >= 0) & (x < self.output_size[1]) & (y >= 0) & (y < self.output_size[0])

        x = x[valid_indices]
        y = y[valid_indices]
        p = events[:, 3][valid_indices]
        t = events[:, 0][valid_indices]

        t_ref = t[-1]
        decay_values = np.exp(-(t_ref - t) / tau)

        event_frame[y[p == 1], x[p == 1]] = decay_values[p == 1]
        event_frame[y[p == 0], x[p == 0]] = -decay_values[p == 0]

        event_frame = np.clip(event_frame, -1, 1)

        return event_frame

    def create_time_surface_channel(self, events_list, tau=50e-3):
        event_frame_in_channel = np.zeros(
            (self.output_size[0], self.output_size[1], self.time_bin), dtype=np.float32
        )
        for i in range(self.time_bin):
            if len(events_list[i]) == 0:
                continue
            events_frame = self.create_time_surface(events_list[i], tau=tau)
            event_frame_in_channel[:, :, i] = events_frame

        # # visualize
        # event_frame_in_channel_vis = ((event_frame_in_channel + 1) / 2 * 255).astype(np.uint8)
        # cv2.imwrite("event_frame_in_channel_vis.jpg", event_frame_in_channel_vis)

        return event_frame_in_channel

    def is_white_area_significant(self, mask_image, threshold=0.15):
        white_pixel_count = np.sum(mask_image > 127)
        total_pixel_count = mask_image.size
        white_ratio = white_pixel_count / total_pixel_count
        return white_ratio >= threshold or white_pixel_count == 0

    def create_lnes_frame(self, events):
        # start_time = time.perf_counter()
        event_frame = np.zeros(
            (self.output_size[0], self.output_size[1], 2), dtype=np.float32
        )
        scale_x = self.output_size[1] / OUTPUT_WIDTH  # original width is 346
        scale_y = self.output_size[0] / OUTPUT_HEIGHT  # original height is 260
        x = (events[:, 1] * scale_x).astype(int)
        y = (events[:, 2] * scale_y).astype(int)

        valid_indices = (x >= 0) & (x < self.output_size[1]) & (y >= 0) & (y < self.output_size[0])
        x = x[valid_indices]
        y = y[valid_indices]
        p = events[:, 3][valid_indices]
        t = events[:, 0][valid_indices]

        t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-6) + 1

        event_frame[y[p == 1], x[p == 1], 0] = t_normalized[p == 1]
        event_frame[y[p == 0], x[p == 0], 1] = t_normalized[p == 0]
        # print(f"create_lnes_frame time: {time.perf_counter() - start_time}")

        return event_frame

    def create_lnes_frame_v2(self, events, window_size=1.0 / 30.0, unit=1.0):
        """
        Create LNES frame from events (2 channels).
        Only use events where dt < window_size * unit to ensure values are in [0, 1] range.
        Based on generate_lnes_from_h5.py implementation.

        Args:
            events: numpy array of events with shape (N, 4), columns: [t, x, y, p]
            window_size: time window size
            unit: time unit

        Returns:
            lnes_frame: numpy array with shape (H, W, 2) with values in [0, 1]
        """
        event_frame = np.zeros(
            (self.output_size[0], self.output_size[1], 2), dtype=np.float32
        )
        if len(events) == 0:
            return event_frame

        scale_x = self.output_size[1] / OUTPUT_WIDTH  # original width is 346
        scale_y = self.output_size[0] / OUTPUT_HEIGHT  # original height is 260
        x = (events[:, 1] * scale_x).astype(np.int32)
        y = (events[:, 2] * scale_y).astype(np.int32)
        p = events[:, 3].astype(np.int32)
        t = events[:, 0].astype(np.float32)

        t_max = t.max()
        dt = t_max - t
        window = window_size * unit

        # Filter events where dt < window to ensure t_normalized is in [0, 1] range
        valid_mask = dt < window
        valid_mask &= (x >= 0) & (x < self.output_size[1]) & (y >= 0) & (y < self.output_size[0])

        if not np.any(valid_mask):
            return event_frame

        # Calculate normalized time for valid events: recent events -> value close to 1
        t_normalized = 1.0 - (dt[valid_mask] / window)

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        p_valid = p[valid_mask]

        event_frame[y_valid[p_valid == 1], x_valid[p_valid == 1], 0] = t_normalized[p_valid == 1]  # positive events
        event_frame[y_valid[p_valid == 0], x_valid[p_valid == 0], 1] = t_normalized[p_valid == 0]  # negative events

        return event_frame

    def __getitem__(self, idx):
        event_file = self.event_input_files[idx]
        event_file_number = event_file.split("/")[-1].split("_")[1].split(".")[0]

        # start_time = time.perf_counter()
        gt_hand_poses = self.get_hand_pose_from_gt_file(event_file, idx)
        # print(f"gt_hand_poses: {time.perf_counter() - start_time}")
        # start_time = time.perf_counter()
        gt_joints = self.get_joints_from_gt_file(idx)
        # print(f"gt_joints: {time.perf_counter() - start_time}")
        # start_time = time.perf_counter()
        # gt_device_pose = self.get_device_pose_from_gt_file(idx)
        # print(f"gt_device_pose: {time.perf_counter() - start_time}")

        # events = []
        # event_file_number_int = int(event_file_number)
        # if event_file_number_int < self.time_bin - 1:
        #     return self.__getitem__(idx + self.time_bin - 1)
        # for i in range(self.time_bin):
        #     event = self.get_events_from_h5(self.event_input_files[idx - i])
        #     events.append(event)
        # events = np.concatenate(events, 0)

        if self.is_sequence:
            # start_time = time.perf_counter()
            event_file_number_int = int(event_file_number)
            if event_file_number_int < self.time_bin - 1:
                return self.__getitem__(idx + self.time_bin - 1)
            events_list = []
            for i in range(self.time_bin):
                events = self.get_events_from_h5(self.event_input_files[idx - i])
                if len(events) != 0:
                    events_list.append(events)
            events = np.concatenate(events_list, 0)
            event_frame = self.create_event_frame(events[:(len(events) // self.time_bin)])
            event_frame = self.transform_3(event_frame)
            lnes_fn = self.create_lnes_frame_v2 if self.lnes_version == 2 else self.create_lnes_frame
            lnes = lnes_fn(events)
            # lnes = torch.tensor(lnes, dtype=torch.float32).permute(2, 0, 1)
            lnes = self.transform_2(lnes)
            # print(f"sequence event time: {time.perf_counter() - start_time}")
        else:
            events = self.get_events_from_h5(event_file)
            if len(events) == 0:
                return self.previous_getitem(idx)
            event_frame = self.create_event_frame(events)
            event_frame = self.transform_3(event_frame)
            lnes_fn = self.create_lnes_frame_v2 if self.lnes_version == 2 else self.create_lnes_frame
            lnes = lnes_fn(events)
            # lnes = torch.tensor(lnes, dtype=torch.float32).permute(2, 0, 1)
            lnes = self.transform_2(lnes)


        # if self.mode == "full":
        #     if len(events) < self.event_number_threshold:
        #         events = self.increase_event(events)
        #     elif len(events) > self.event_number_threshold:
        #         events = self.sample_event(events)

        # random_num = random.random()
        # if self.augmentation and random_num > 0.5:
        #     if random_num > 0.75:
        #         events = augment_events_flip(events)
        #     else:
        #         events = augment_events_time_shuffle(events)
        # assert len(events) == self.event_number_threshold, f"len(events): {len(events)}"

        # event_frame = rotate(event_frame, 90, reshape=True)


        # self.plot_row_events(events, "in_event_plot.png")
        if self.mode == "ev2hands":
            events = self.create_ev2hands_event(events)
        else:
            events = self.create_event_cloud(events)
        # self.plot_point_cloud(events, "in_point_cloud.png")

        ## bbox or mask segmentation
        if self.mode == "mask":
            mask_file_left = self.mask_files[idx * 2]
            mask_file_right = self.mask_files[idx * 2 + 1]
            event_file_number = event_file.split("/")[-1].split("_")[1].split(".")[0]
            # left_number = mask_file_left.split("/")[-1].split("_")[1]
            # right_number = mask_file_right.split("/")[-1].split("_")[1]
            # assert (
            #     left_number == right_number == event_file_number
            # ), f"left_number != right_number : {left_number} != {right_number}"
            mask_left = self.load_mask(mask_file_left)
            mask_right = self.load_mask(mask_file_right)
            mix_mask = self.mix_mask(mask_left, mask_right)
            if self.is_white_area_significant(mix_mask):
                random_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(random_idx)
            mix_mask = mix_mask / 255.0
            if self.is_rotate:
                event_frame = rotate(event_frame, 90, reshape=True)
                mix_mask = rotate(mix_mask, 90, reshape=True)
            mix_mask = np.clip(mix_mask, 0, 1)
            mix_mask = torch.tensor(mix_mask, dtype=torch.float32).unsqueeze(0)

            masked_events = mask_event_cloud_one(mix_mask, events, 2048)
            events = masked_events
            # visualize
            # frame = self.event_point_cloud_to_image(events)
            # mix_mask = mix_mask.squeeze(0).numpy()
            # mix_mask = (mix_mask * 255).astype(np.uint8)
            # mix_mask = cv2.cvtColor(mix_mask, cv2.COLOR_GRAY2BGR)
            # concat_image = np.concatenate([frame, mix_mask], 1)
            # cv2.imwrite("mask_visualize.png", concat_image)

            # self.plot_point_cloud(events, "masked_point_cloud.png")

            # events[:, 2] -= events[0, 2]  # normalize ts
            events[:, :3] = self.pc_normalize(events[:, :3])
        if self.mode == "bbox":
            bbox = self.get_hand_bbox_from_gt_file(idx)
            events = bbox_event_one(bbox, events, 2048)
            # events[:, 2] -= events[0, 2]  # normalize ts
            events[:, :3] = self.pc_normalize(events[:, :3])
        if self.mode == "full":
            pass

        # assert events.shape == (
        #     2048,
        #     5,
        # ), f"events.shape: {events.shape}"

        data = {
            "mano_gt": 1.0,
            "events": events.permute(1, 0),  # torch.Size([5, N])
            "event_frame": event_frame,  # <class 'torch.Tensor'> torch.Size([3, HEIGHT, WIDTH])
            # "event_frame_list": event_frame_list,
            "LNES": lnes,
            "gt_hand_poses": gt_hand_poses,
            "gt_joints": gt_joints,
            # "gt_device_pose": gt_device_pose,
            "event_file_path": event_file,
        }

        if self.no_resample:
            data["event_count"] = events.shape[0]

        if self.mode == "mask":
            data["mask"] = mix_mask
        elif self.mode == "bbox":
            data["bbox"] = bbox

        return data


if __name__ == "__main__":
    dataset = NHOT3DDataset(
        event_input_dir="/path/to/N-HOT3D/Aria/test",
        gt_file_dir="/path/to/N-HOT3D/Aria/test",
        object_library_path="/path/to/N-HOT3D/assets",
        mano_hand_model_path="src/mano/models",
        is_get_image=False,
        augmentation=False,
        output_size=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
        mode="mask",
        dataset_divide=10,
    )
    print(len(dataset))
    data = dataset[10000]
