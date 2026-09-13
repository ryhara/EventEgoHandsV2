import os
import argparse
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from datetime import datetime
import sys
import numpy as np
import random

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

# Ev2Hands
from datasets.nhot3d_dataset import NHOT3DDataset
from utils.joint_evaluation import evaluate_hand, get_auc

from models import load_unet
from factory import get_hand_model
from utils import HandEstimationConfig, mask_event_cloud_batch, pc_normalize_batch,  calc_PA_MPJPE, calc_MPJPE3d, calc_MPVPE3d, dilation_masks

log = logging.getLogger(os.path.basename(__file__))

os.environ["ERPC"] = "1"

def recursive_collate_fn(key, values):
    if key == 'events':
        events = [ev.clone().detach().permute(1, 0) for ev in values]
        padded_events = torch.nn.utils.rnn.pad_sequence(events, batch_first=True, padding_value=0) # batch_size, N, 5
        return padded_events
    elif isinstance(values[0], dict):
        return {k: recursive_collate_fn(k, [v[k] for v in values]) for k in values[0].keys()}
    elif isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    else:
        return values

def collate_fn(batch):
    batched_data = {key: recursive_collate_fn(key, [b[key] for b in batch]) for key in batch[0].keys()}
    return batched_data


def test_hand(net, segment_net, cfg: HandEstimationConfig, device, mode="default"):
    net.eval()
    test_dataset = NHOT3DDataset(
        event_input_dir=cfg.test.event_input_dir,
        gt_file_dir=cfg.test.gt_file_dir,
        object_library_path=cfg.test.object_library_path,
        mano_hand_model_path=cfg.mano_hand_model_path,
        is_get_image=cfg.test.is_get_image,
        augmentation=False,
        is_normalize=False,
        is_rotate=False,
        mode=cfg.dataset_mode,
        output_size=(cfg.output_height, cfg.output_width),
        event_number_threshold=cfg.event_number_threshold,
        time_bin=cfg.time_bin,
        is_sequence=cfg.is_sequence,
        dataset_divide=cfg.dataset_divide,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        num_steps = 100  # 0mm to 100mm

        joint_loss = 0
        absolute_pck3d = np.zeros(num_steps + 1)
        relative_pck3d = np.zeros(num_steps + 1)
        right_root_relative_pck3d = np.zeros(num_steps + 1)
        root_distance = []
        frame_index = 1
        pa_mpjpe_list = []
        mpjpe_list = []
        mpvpe_list = []

        for index, batch in enumerate(
            tqdm(
                test_loader,
                total=len(test_loader),
                desc="Evaluation",
                unit="batch",
            )
        ):
            batch = NHOT3DDataset.to_device(batch, device)
            events = batch["events"]
            event_frames = batch["LNES"]

            segment_outputs = segment_net(event_frames)
            segment_outputs = torch.sigmoid(segment_outputs)
            segment_outputs = segment_outputs > cfg.sigmoid_threshold
            batch_size = segment_outputs.shape[0]
            segment_outputs = dilation_masks(segment_outputs, batch_size=batch_size, kernel_size=cfg.kernel_size, itterations=cfg.itterations)
            segment_outputs = torch.tensor(segment_outputs, dtype=torch.float32).to(device)
            segment_outputs = segment_outputs.unsqueeze(1)
            masked_events = mask_event_cloud_batch(
                segment_outputs, events, 2048
            )

            masked_events[:, :, :3] = pc_normalize_batch(masked_events[:, :, :3])
            masked_events = masked_events.permute(0, 2, 1)
            masked_events = masked_events.to(device)

            # TODO: visualization
            # input_event_frame_one = event_frames[0].permute(1, 2, 0).cpu().numpy()
            # input_event_frame_one = (input_event_frame_one * 255).astype(np.uint8)
            # input_event_frame_one = cv2.rotate(
            #     input_event_frame_one, cv2.ROTATE_90_CLOCKWISE
            # )

            # segment_output_one = segment_outputs[0].permute(1, 2, 0).cpu().numpy()
            # segment_output_one = (segment_output_one * 255).astype(np.uint8)
            # segment_output_one = cv2.cvtColor(segment_output_one, cv2.COLOR_GRAY2BGR)

            # concat_frame = np.concatenate(
            #     [input_event_frame_one, segment_output_one, masked_event_frame_one],
            #     axis=1,
            # )
            # cv2.imwrite("masked.png", concat_frame)

            outputs = net(masked_events)

            gt_hand_poses = batch["gt_hand_poses"]
            gt_joints = batch["gt_joints"]


            pa_mpjpe = calc_PA_MPJPE(outputs, gt_joints, device, pred_key="j3d", coordinate="camera")
            mpjpe = calc_MPJPE3d(outputs, gt_joints, device, pred_key="j3d", coordinate="camera")
            mpvpe = calc_MPVPE3d(outputs, gt_joints, device, coordinate="camera")

            pa_mpjpe = pa_mpjpe * 1000
            mpjpe = mpjpe * 1000
            mpvpe = mpvpe * 1000
            pa_mpjpe_list.append(pa_mpjpe)
            mpjpe_list.append(mpjpe)
            mpvpe_list.append(mpvpe)


            # hot3d
            if mode == "NHOT3D":
                # world
                # left_gt_joints = gt_hand_poses["left"]["hand_landmarks"]
                # right_gt_joints = gt_hand_poses["right"]["hand_landmarks"]
                # camera
                left_gt_joints = gt_joints["left"]["joints_3d"]
                right_gt_joints = gt_joints["right"]["joints_3d"]

                j3d_gts = torch.cat(
                    [left_gt_joints.unsqueeze(1), right_gt_joints.unsqueeze(1)], dim=1
                )
            # manopth
            else:
                left_outputs = net.hands["left"](
                    global_orient=gt_hand_poses["left"]["global_orient"],
                    hand_pose=gt_hand_poses["left"]["hand_pose"],
                    betas=gt_hand_poses["left"]["shape"],
                    transl=gt_hand_poses["left"]["trans"],
                )
                right_outputs = net.hands["right"](
                    global_orient=gt_hand_poses["right"]["global_orient"],
                    hand_pose=gt_hand_poses["right"]["hand_pose"],
                    betas=gt_hand_poses["right"]["shape"],
                    transl=gt_hand_poses["right"]["trans"],
                )
                j3d_gts = torch.cat(
                    [left_outputs.joints.unsqueeze(1), right_outputs.joints.unsqueeze(1)],
                    dim=1,
                )

            frames = evaluate_hand(
                pred=outputs, j3d_gts=j3d_gts, device=device, num_steps=num_steps
            )
            for idx, frame in enumerate(frames):
                scores = frame["scores"]

                root_distance += scores["root_distance"]

                absolute_pck3d += scores["absolute_pck3d"]
                relative_pck3d += scores["relative_pck3d"]
                right_root_relative_pck3d += scores["right_root_relative_pck3d"]
                joint_loss += scores["joint_loss"]

                absolute_auc = get_auc(absolute_pck3d / frame_index)
                relative_auc = get_auc(relative_pck3d / frame_index)
                right_root_relative_auc = get_auc(
                    right_root_relative_pck3d / frame_index
                )

                # log.info(
                #     f"Absolute AUC: {absolute_auc} Relative AUC: {relative_auc} Right Root Relative AUC: {right_root_relative_auc} Joint Loss: {joint_loss / frame_index}"
                # )
                wandb.log(
                    {
                        "Absolute AUC": absolute_auc,
                        "Relative AUC": relative_auc,
                        "Right Root Relative AUC": right_root_relative_auc,
                        "Joint Loss": joint_loss / frame_index,
                    }
                )

                frame_index += 1

        joint_loss /= frame_index
        absolute_pck3d /= frame_index
        relative_pck3d /= frame_index
        right_root_relative_pck3d /= frame_index

        absolute_auc = get_auc(absolute_pck3d)
        relative_auc = get_auc(relative_pck3d)
        right_root_relative_auc = get_auc(right_root_relative_pck3d)

        log.info(
            f"[Average] Absolute AUC: {absolute_auc} Relative AUC: {relative_auc} Right Root Relative AUC: {right_root_relative_auc} Joint Loss: {joint_loss}"
        )
        wandb.log(
            {
                "Average Absolute AUC": absolute_auc,
                "Average Relative AUC": relative_auc,
                "Average Right Root Relative AUC": right_root_relative_auc,
                "Average Joint Loss": joint_loss,
            }
        )
        mean_pa_mpjpe = sum(pa_mpjpe_list) / len(pa_mpjpe_list)
        mean_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        mean_mpvpe = sum(mpvpe_list) / len(mpvpe_list)
        log.info(
            f"Mean PA-MPJPE: {mean_pa_mpjpe}, Mean MPJPE: {mean_mpjpe} Mean MPVPE: {mean_mpvpe}"
        )
        wandb.log(
            {
                "Mean PA-MPJPE": mean_pa_mpjpe,
                "Mean MPJPE": mean_mpjpe,
                "Mean MPVPE": mean_mpvpe,
            }
        )


def main(cfg: HandEstimationConfig) -> None:
    torch.cuda.empty_cache()
    # seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v1_test_nhot3d_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(
        f"cuda:{device_first}" if torch.cuda.is_available() else "cpu"
    )

    log.info(f"Loading segmentation model from: {cfg.segmentation_checkpoint_path}")
    segment_net = load_unet(
        cfg.segmentation_checkpoint_path,
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        device=device,
    )
    for param in segment_net.parameters():
        param.requires_grad = False

    net = get_hand_model(cfg)
    log.info(f"model parameters: {sum(p.numel() for p in net.parameters())}")
    net = net.to(device)
    if os.path.exists(cfg.test.checkpoint_path):
        checkpoint = torch.load(
            cfg.test.checkpoint_path, weights_only=True, map_location=device
        )
        net.load_state_dict(checkpoint)
        log.info(f"Load model from {cfg.test.checkpoint_path}")

    test_hand(net, segment_net, cfg, device, mode="NHOT3D")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_test_nhot3d.yaml"),
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
