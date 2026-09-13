import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from datasets.eehr_dataset import EEHRDataset
from models import BaseLoss, load_unet
from factory import get_hand_model
from utils import HandEstimationConfig, dilation_masks

log = logging.getLogger(os.path.basename(__file__))

os.environ["ERPC"] = "1"

# Event camera native resolution
SRC_EVENT_WIDTH = 346
SRC_EVENT_HEIGHT = 260
# lnes_frames/*.png are pre-cropped to 260x260 (center crop on width)
LNES_CROP_SIZE = 260
CROP_X_OFFSET = (SRC_EVENT_WIDTH - LNES_CROP_SIZE) // 2  # 43


def transform_event_coords_to_input(
    events: torch.Tensor, input_size: int
) -> torch.Tensor:
    """Map event (x, y) from source 346x260 coords to input_size x input_size coords.

    Matches lnes_frames preprocessing: center-crop on width to 260x260, then resize
    to input_size. Out-of-crop events get (-1, -1) so they are filtered by the mask.

    Args:
        events: [B, N, 5] tensor with channels (x, y, t, p_pos, p_neg) in last dim.
    """
    events = events.clone()
    x = events[..., 0]
    y = events[..., 1]
    x_cropped = x - CROP_X_OFFSET
    in_crop = (x_cropped >= 0) & (x_cropped < LNES_CROP_SIZE)
    scale = float(input_size) / LNES_CROP_SIZE
    events[..., 0] = torch.where(
        in_crop, x_cropped * scale, torch.full_like(x_cropped, -1.0)
    )
    events[..., 1] = y * scale
    return events


def mask_event_cloud_batch_local(
    masks_batch: torch.Tensor,
    events_batch: torch.Tensor,
    event_number_threshold: int = 2048,
) -> torch.Tensor:
    """Filter events by mask; both share the same (input_size x input_size) coord space.

    Args:
        masks_batch: [B, 1, H, W] mask tensor (bool or float, >0 means inside).
        events_batch: [B, N, 5] events with xy already mapped to mask coords.

    Returns:
        [B, N, 5] filtered events (channel-last, matching prior pipeline).
    """
    # Dataset keeps events on CPU; move to mask device for torch indexing.
    device = masks_batch.device
    events_batch = events_batch.to(device)
    B = events_batch.shape[0]
    _, _, H, W = masks_batch.shape

    out_list = []
    for i in range(B):
        ev = events_batch[i]  # [N, 5]
        mask = masks_batch[i, 0]  # [H, W]
        x = ev[:, 0].long()
        y = ev[:, 1].long()
        in_range = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x_c = x.clamp(0, W - 1)
        y_c = y.clamp(0, H - 1)
        inside = in_range & (mask[y_c, x_c] > 0)
        filt = ev[inside]  # [M, 5]

        M = filt.shape[0]
        N = ev.shape[0]
        if M == 0:
            idx = torch.randint(0, N, (event_number_threshold,), device=device)
            filt = ev[idx]
        elif M < event_number_threshold:
            idx = torch.randint(0, M, (event_number_threshold - M,), device=device)
            filt = torch.cat([filt, filt[idx]], dim=0)
        elif M > event_number_threshold:
            idx = torch.randperm(M, device=device)[:event_number_threshold]
            filt = filt[idx]
        out_list.append(filt.unsqueeze(0))  # [1, N, 5]
    return torch.cat(out_list, dim=0)


def pc_normalize_batch_local(pc: torch.Tensor, input_size: int) -> torch.Tensor:
    """Normalize xy to [-1, 1] using input_size and t to [-1, 1] per-sample.

    Args:
        pc: [B, N, 3] tensor with channels (x, y, t).
    """
    pc = pc.clone()
    pc[:, :, 0] = pc[:, :, 0] / input_size
    pc[:, :, 1] = pc[:, :, 1] / input_size
    pc[:, :, :2] = 2 * pc[:, :, :2] - 1

    ts = pc[:, :, 2:]
    t_max = ts.max(1).values
    t_min = ts.min(1).values
    diff = t_max - t_min
    if torch.any(diff == 0):
        diff = t_max + 1e-6
    pc[:, :, 2:] = 2 * ((ts - t_min.unsqueeze(1)) / diff.unsqueeze(1)) - 1
    return pc


def create_dataset(cfg, split: str):
    return EEHRDataset(
        dataset_root=cfg.real_dataset_root,
        split=split,
        fps=getattr(cfg, "real_dataset_fps", 30),
        output_size=(cfg.output_height, cfg.output_width),
        use_center_crop=getattr(cfg, "use_center_crop", False),
        ignore_files_count=cfg.ignore_files_count,
        dataset_divide=cfg.train.dataset_divide,
        load_h5=True,
        event_number_threshold=cfg.event_number_threshold,
        annotation_source=getattr(cfg, "annotation_source", "mano"),
        annotation_fps=getattr(cfg, "annotation_fps", 30),
        lnes_from_h5=getattr(cfg, "lnes_from_h5", False),
        lnes_window_sec=getattr(cfg, "lnes_window_sec", 1.0 / 30.0),
    )


def recursive_collate_fn(key, values):
    if key == "events":
        # (5, N) -> (N, 5), then pad to same length
        events = [ev.clone().detach().permute(1, 0) for ev in values]
        padded_events = torch.nn.utils.rnn.pad_sequence(
            events, batch_first=True, padding_value=0
        )  # (batch_size, N, 5)
        return padded_events
    elif isinstance(values[0], dict):
        return {
            k: recursive_collate_fn(k, [v[k] for v in values])
            for k in values[0].keys()
        }
    elif isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    elif isinstance(values[0], (str, bool, int, float, type(None))):
        return values
    else:
        return values


def collate_fn(batch):
    batched_data = {
        key: recursive_collate_fn(key, [b[key] for b in batch])
        for key in batch[0].keys()
    }
    return batched_data


def train_eehr(net, segment_net, cfg: HandEstimationConfig, device):
    epochs = cfg.train.epochs

    print("Using dataset type: real (full pipeline)")
    print(f"annotation_source: {getattr(cfg, 'annotation_source', 'mano')}")
    print(f"annotation_fps: {getattr(cfg, 'annotation_fps', 30)}")

    train_dataset = create_dataset(cfg, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    validation_dataset = create_dataset(cfg, "valid")
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    criterion = BaseLoss(device=device)
    iteration = 0

    for epoch in range(epochs):
        train_loss = 0
        net.train()
        for index, batch in enumerate(
            tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
            )
        ):
            batch = EEHRDataset.to_device(batch, device)
            events = batch["events"]
            event_frames = batch["lnes"]

            input_size = event_frames.shape[-1]
            events = transform_event_coords_to_input(events, input_size)

            with torch.no_grad():
                segment_outputs = segment_net(event_frames)
                segment_outputs = torch.sigmoid(segment_outputs)
                segment_outputs = segment_outputs > cfg.sigmoid_threshold
                batch_size = segment_outputs.shape[0]
                segment_outputs = dilation_masks(
                    segment_outputs,
                    batch_size=batch_size,
                    kernel_size=cfg.kernel_size,
                    itterations=cfg.itterations,
                )
                segment_outputs = (
                    torch.tensor(segment_outputs, dtype=torch.float32)
                    .to(device)
                    .unsqueeze(1)
                )

            masked_events = mask_event_cloud_batch_local(
                segment_outputs, events, 2048
            )
            masked_events[:, :, :3] = pc_normalize_batch_local(
                masked_events[:, :, :3], input_size=input_size
            )
            masked_events = masked_events.permute(0, 2, 1)  # (B, 5, N)
            masked_events = masked_events.to(device)

            outputs = net(masked_events)

            losses = criterion(outputs, batch)
            loss = sum(losses.values())
            losses["loss"] = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % cfg.train.log_interval == 0:
                log_dict = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in losses.items()
                }
                wandb.log(log_dict)

            train_loss += float(loss)
            iteration += 1

        train_loss /= len(train_loader)
        log.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}")
        wandb.log({"train_loss_mean": train_loss})

        # validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for index, batch in enumerate(
                tqdm(
                    validation_loader,
                    total=len(validation_loader),
                    desc="Validation",
                    unit="batch",
                )
            ):
                batch = EEHRDataset.to_device(batch, device)
                events = batch["events"]
                event_frames = batch["lnes"]

                input_size = event_frames.shape[-1]
                events = transform_event_coords_to_input(events, input_size)

                segment_outputs = segment_net(event_frames)
                segment_outputs = torch.sigmoid(segment_outputs)
                segment_outputs = segment_outputs > cfg.sigmoid_threshold
                batch_size = segment_outputs.shape[0]
                segment_outputs = dilation_masks(
                    segment_outputs,
                    batch_size=batch_size,
                    kernel_size=cfg.kernel_size,
                    itterations=cfg.itterations,
                )
                segment_outputs = (
                    torch.tensor(segment_outputs, dtype=torch.float32)
                    .to(device)
                    .unsqueeze(1)
                )

                masked_events = mask_event_cloud_batch_local(
                    segment_outputs, events, 2048
                )
                masked_events[:, :, :3] = pc_normalize_batch_local(
                    masked_events[:, :, :3], input_size=input_size
                )
                masked_events = masked_events.permute(0, 2, 1)
                masked_events = masked_events.to(device)

                outputs = net(masked_events)
                losses = criterion(outputs, batch)
                loss = sum(losses.values())
                val_loss += float(loss)

        val_loss /= len(validation_loader)
        wandb.log({"val_loss_mean": val_loss})
        log.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")

        # save model
        save_interval = getattr(cfg.train, "save_interval", 2)
        if (epoch + 1) % save_interval == 0:
            date_str = datetime.now().strftime("%Y-%m-%d/")
            save_dir = os.path.join(cfg.train.save_path, date_str)
            os.makedirs(save_dir, exist_ok=True)
            time_str = datetime.now().strftime("%H-%M-%S")
            save_path = os.path.join(
                save_dir,
                f"v1_eehr_hand_{epoch + 1}_{time_str}_val_loss_{val_loss:.4f}.pth",
            )
            log.info(f"Save model to {save_path}")
            torch.save(net.state_dict(), save_path)

    log.info("Finish training, iteration: {}".format(iteration))


def main(cfg: HandEstimationConfig) -> None:
    torch.cuda.empty_cache()

    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v1_train_eehr_{cfg.model}_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(
        f"cuda:{device_first}" if torch.cuda.is_available() else "cpu"
    )

    # Load segmentation model (frozen)
    print(f"Loading segmentation model from: {cfg.segmentation_checkpoint_path}")
    segment_net = load_unet(
        cfg.segmentation_checkpoint_path,
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        device=device,
    )
    for param in segment_net.parameters():
        param.requires_grad = False
    segment_net.eval()

    # Load hand estimation model
    net = get_hand_model(cfg)
    log.info(f"model parameters: {sum(p.numel() for p in net.parameters())}")
    net = net.to(device)

    if os.path.exists(cfg.train.checkpoint_path):
        state_dict = torch.load(
            cfg.train.checkpoint_path, weights_only=False, map_location=device
        )
        net.load_state_dict(state_dict)
        log.info(f"Load model from {cfg.train.checkpoint_path}")

    train_eehr(net, segment_net, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_train_eehr.yaml"),
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
