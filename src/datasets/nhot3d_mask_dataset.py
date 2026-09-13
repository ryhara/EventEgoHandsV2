import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
import numpy as np
import random
import h5py
from PIL import Image
import json
from scipy.ndimage import rotate
import cv2

from hot3d.hot3d.dataset_api import Hot3dDataProvider
from datasets import augment_events_flip

OUTPUT_WIDTH = 346
OUTPUT_HEIGHT = 260


class NHOT3DDatasetForMask(Dataset):
    def __init__(
        self,
        event_input_dir: str,
        gt_file_dir: str,
        augmentation: bool = False,
        is_get_image: bool = False,
        is_rotate: bool = False,
        output_size: tuple = (OUTPUT_HEIGHT, OUTPUT_WIDTH),
        in_channels: int = 3,
        mode="default",
        time_bin: int = 1,
        is_sequence: bool = False,
        dataset_divide: int = 1,
    ):
        self.event_input_dir = event_input_dir
        self.augmentation = augmentation
        self.output_size = output_size
        self.is_get_image = is_get_image
        self.is_rotate = is_rotate
        self.in_channels = in_channels
        self.mode = mode
        self.dataset_divide = dataset_divide
        self.time_bin = time_bin
        self.is_sequence = is_sequence

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

        # Hand mask gt
        self.mask_files = self.get_all_mask_files()
        assert (
            len(self.event_input_files) * 2 == len(self.mask_files)
        ), f"len(event_input_files) * 2 != len(mask_files) : {len(self.event_input_files) * 2} != {len(self.mask_files)}"

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
                    if file.startswith("gt_") and file.endswith(".jsonl") and "bbox" not in file:
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

    def get_events_from_h5(self, h5_file):
        data = h5py.File(h5_file, "r")
        events = data["events"]
        events = np.array(events, np.float32)
        data.close()
        return events

    def create_event_frame(self, events):
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

        return event_frame

    def create_event_frame_single_channel(self, events):
        event_frame = np.zeros(
            (self.output_size[0], self.output_size[1]), dtype=np.uint8
        )
        scale_x = self.output_size[1] / OUTPUT_WIDTH
        scale_y = self.output_size[0] / OUTPUT_HEIGHT
        x = (events[:, 1] * scale_x).astype(int)
        y = (events[:, 2] * scale_y).astype(int)

        valid_indices = (x >= 0) & (x < self.output_size[1]) & (y >= 0) & (y < self.output_size[0])
        x = x[valid_indices]
        y = y[valid_indices]
        p = events[:, 3][valid_indices]

        event_frame[y[p == 1], x[p == 1]] = 1
        event_frame[y[p == 0], x[p == 0]] = -1

        return event_frame

    def create_event_frame_channel(self, events_list):
        event_frame_in_channel = np.zeros(
            (self.output_size[0], self.output_size[1], self.in_channels), dtype=np.uint8
        )
        for i in range(self.in_channels):
            if len(events_list[i]) == 0:
                continue
            events_frame = self.create_event_frame_single_channel(events_list[i])
            event_frame_in_channel[:, :, i] = events_frame

        return event_frame_in_channel

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

        # visualize
        # event_frame_vis = ((event_frame + 1) / 2 * 255).astype(np.uint8)
        # event_frame_vis = np.zeros((self.output_size[0], self.output_size[1], 3), dtype=np.uint8)

        # # color
        # positive_mask = event_frame > 0
        # negative_mask = event_frame < 0

        # event_frame_vis[..., 1][positive_mask] = 255
        # event_frame_vis[..., 2][negative_mask] = 255

        # cv2.imwrite("event_frame_vis.jpg", event_frame_vis)

        return event_frame

    def create_time_surface_channel(self, events_list, tau=50e-3):
        event_frame_in_channel = np.zeros(
            (self.output_size[0], self.output_size[1], self.in_channels), dtype=np.float32
        )
        for i in range(self.in_channels):
            if len(events_list[i]) == 0:
                continue
            events_frame = self.create_time_surface(events_list[i], tau=tau)
            event_frame_in_channel[:, :, i] = events_frame

        # # visualize
        # event_frame_in_channel_vis = ((event_frame_in_channel + 1) / 2 * 255).astype(np.uint8)
        # cv2.imwrite("event_frame_in_channel_vis.jpg", event_frame_in_channel_vis)

        return event_frame_in_channel

    def get_hand_pose_from_gt_file(self, event_file):
        event_file_dir = os.path.dirname(event_file)
        event_file_number_str = (
            os.path.basename(event_file).split("_")[-1].split(".")[0]
        )
        event_file_number = int(event_file_number_str)
        gt_dir = os.path.join(event_file_dir, "gt")
        gt_file = os.path.join(gt_dir, f"gt_{event_file_number_str}.jsonl")
        if not os.path.exists(gt_file):
            print(f"gt_file: {gt_file} does not exist.")
            return None
        with open(gt_file, "r") as f:
            gt_hand_poses = json.load(f)
        for key in ["left", "right"]:
            if gt_hand_poses.get(key) is None:
                continue
            hand_pose = gt_hand_poses[key]
            gt_hand_poses[key] = {
                "betas": torch.tensor(hand_pose["betas"], dtype=torch.float32),
                "quat": torch.tensor(hand_pose["quat"], dtype=torch.float32),
                "translation": torch.tensor(
                    hand_pose["translation"], dtype=torch.float32
                ),
                "joint_angles": torch.tensor(
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

    def rotation_augmentation(self, event_frame, mask_image, range=30):
        angle = np.random.randint(-range, range)
        event_frame = rotate(event_frame, angle, reshape=False)
        mask_image = rotate(mask_image, angle, reshape=False)
        return event_frame, mask_image

    def load_mask(self, mask_file):
        mask_image = Image.open(mask_file).convert("L")
        mask_image = mask_image.resize((self.output_size[1], self.output_size[0]))
        mask_image = np.array(mask_image)
        return mask_image

    def mix_mask(self, mask_left, mask_right):
        mix_mask = np.maximum(mask_left, mask_right)
        if mix_mask is None:
            mix_mask = np.zeros(
                (self.output_size[0], self.output_size[1]), dtype=np.uint8
            )
        return mix_mask

    def is_white_area_significant(self, mask_image, threshold=0.15):
        white_pixel_count = np.sum(mask_image > 127)
        total_pixel_count = mask_image.size
        white_ratio = white_pixel_count / total_pixel_count
        return white_ratio >= threshold or white_pixel_count == 0

    def previous_getitem(self, idx):
        return self.__getitem__(idx - 1)

    def create_lnes_frame(self, events):
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

        return event_frame

    def __getitem__(self, idx):
        event_file = self.event_input_files[idx]
        mask_file_left = self.mask_files[idx * 2]
        mask_file_right = self.mask_files[idx * 2 + 1]

        # check number
        event_file_number = event_file.split("/")[-1].split("_")[1].split(".")[0]

        # assert
        # left_number = mask_file_left.split("/")[-1].split("_")[1]
        # right_number = mask_file_right.split("/")[-1].split("_")[1]
        # assert (
        #     left_number == right_number == event_file_number
        # ), f"left_number != right_number : {left_number} != {right_number}"

        if self.is_sequence:
            event_file_number_int = int(event_file_number)
            if event_file_number_int < self.time_bin - 1:
                return self.__getitem__(idx + self.time_bin - 1)
            previous_event_file = self.event_input_files[idx - 1]
            previous_event_file_number = previous_event_file.split("/")[-1].split("_")[1].split(".")[0]
            if event_file_number_int - int(previous_event_file_number) != 1:
                random_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(random_idx)
            events_list = []
            for i in range(self.time_bin):
                events = self.get_events_from_h5(self.event_input_files[idx - i])
                if len(events) != 0:
                    events_list.append(events)
            events = np.concatenate(events_list, 0)
            random_num = random.random()
            if self.augmentation and random_num > 0.5:
                events = augment_events_flip(events)
            lnes = self.create_lnes_frame(events)
            event_frame = self.create_event_frame(events[:(len(events) // self.time_bin)])
        else:
            events = self.get_events_from_h5(event_file)
            if len(events) == 0:
                return self.previous_getitem(idx)
            random_num = random.random()
            if self.augmentation and random_num > 0.5:
                events = augment_events_flip(events)
            event_frame = self.create_event_frame(events)
            lnes = self.create_lnes_frame(events)

        mask_left = self.load_mask(mask_file_left)
        mask_right = self.load_mask(mask_file_right)
        mix_mask = self.mix_mask(mask_left, mask_right)
        if self.is_white_area_significant(mix_mask):
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)
        mix_mask = mix_mask / 255.0
        # if self.is_rotate:
        #     event_frame = rotate(event_frame, 90, reshape=True)
        #     mix_mask = rotate(mix_mask, 90, reshape=True)

        # # augmentation
        # if self.augmentation and random.random() > 0.5:
        #     event_frame, mix_mask = self.rotation_augmentation(
        #         event_frame, mix_mask, range=30
        #     )

        event_frame = self.transform_3(event_frame)
        # lnes = torch.tensor(lnes, dtype=torch.float32).permute(2, 0, 1)
        lnes = self.transform_2(lnes)
        mix_mask = np.clip(mix_mask, 0, 1)

        kernel = np.ones((2, 2), np.uint8)
        mix_mask = cv2.dilate(mix_mask, kernel, iterations=1)

        data = {
            # "events": torch.tensor(events, dtype=torch.float32),
            "event_frame": event_frame,
            "LNES": lnes,
            "gt_mask": torch.tensor(np.array(mix_mask), dtype=torch.float32).unsqueeze(
                0
            ),
        }

        return data


if __name__ == "__main__":
    import random

    dataset = NHOT3DDatasetForMask(
        event_input_dir="/path/to/N-HOT3D/Aria/test",
        gt_file_dir="/path/to/N-HOT3D/Aria/test",
        augmentation=False,
        is_get_image=False,
        output_size=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
    )
    print(len(dataset))
    idx = random.randint(0, len(dataset))
    sample = dataset[idx]
