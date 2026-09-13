import os
import argparse
import logging
import torch
import torch.optim as optim
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

from datasets.nhot3d_dataset import NHOT3DDataset
from models import BaseLoss, load_unet
from factory import get_hand_model
from utils import HandEstimationConfig, mask_event_cloud_batch, pc_normalize_batch, dilation_masks

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

def train_hand(net, segment_net, cfg: HandEstimationConfig, device):
    epochs = cfg.train.epochs

    train_dataset = NHOT3DDataset(
        event_input_dir=cfg.train.event_input_dir,
        gt_file_dir=cfg.train.gt_file_dir,
        object_library_path=cfg.train.object_library_path,
        mano_hand_model_path=cfg.mano_hand_model_path,
        is_get_image=cfg.train.is_get_image,
        augmentation=cfg.train.augmentation,
        is_normalize=False,
        output_size=(cfg.output_height, cfg.output_width),
        mode=cfg.dataset_mode,
        event_number_threshold=cfg.event_number_threshold,
        time_bin=cfg.time_bin,
        is_sequence=cfg.is_sequence,
        dataset_divide=cfg.dataset_divide,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    validation_dataset = NHOT3DDataset(
        event_input_dir=cfg.valid.event_input_dir,
        gt_file_dir=cfg.valid.gt_file_dir,
        object_library_path=cfg.valid.object_library_path,
        mano_hand_model_path=cfg.mano_hand_model_path,
        is_get_image=cfg.valid.is_get_image,
        augmentation=False,
        is_normalize=False,
        output_size=(cfg.output_height, cfg.output_width),
        mode=cfg.dataset_mode,
        event_number_threshold=cfg.event_number_threshold,
        time_bin=cfg.time_bin,
        is_sequence=cfg.is_sequence,
        dataset_divide=cfg.dataset_divide,
    )
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
    # criterion = Loss()
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
            batch = NHOT3DDataset.to_device(batch, device)
            events = batch["events"]
            # train_dataset.plot_point_cloud(
            #     events[0].cpu().permute(1, 0), "events_plot.png"
            # )
            event_frames = batch["LNES"]

            with torch.no_grad():
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

            # visualize
            # masked_event_frame_one = train_dataset.event_point_cloud_to_image(
            #     masked_events[0].cpu()
            # )

            masked_events[:, :, :3] = pc_normalize_batch(masked_events[:, :, :3])
            masked_events = masked_events.permute(0, 2, 1)
            masked_events = masked_events.to(device) # batch_size, 5, N

            # event_original_frames = batch["event_frame"]
            # input_event_frame_one = event_original_frames[0].permute(1, 2, 0).cpu().numpy()
            # input_event_frame_one = (input_event_frame_one * 255).astype(np.uint8)

            # segment_output_one = segment_outputs[0].permute(1, 2, 0).cpu().numpy()
            # segment_output_one = (segment_output_one * 255).astype(np.uint8)
            # segment_output_one = cv2.cvtColor(segment_output_one, cv2.COLOR_GRAY2BGR)

            # concat_frame = np.concatenate(
            #     [input_event_frame_one, segment_output_one, masked_event_frame_one],
            #     axis=1,
            # )
            # cv2.imwrite("masked.png", concat_frame)

            outputs = net(masked_events)

            losses = criterion(outputs, batch)
            loss = sum(losses.values())

            losses["loss"] = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log(losses)
            train_loss += float(loss)
            iteration += 1

        # train loss
        train_loss /= len(train_loader)
        log.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}")
        wandb.log(
            {
                "train_loss_mean": train_loss,
            }
        )

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

                outputs = net(masked_events)
                losses = criterion(outputs, batch)
                loss = sum(losses.values())

                losses["loss"] = loss

                val_loss += float(loss)

                # val_losses = {f"val_{k}": v for k, v in losses.items()}
                # wandb.log(val_losses)

        # validation loss
        val_loss /= len(validation_loader)
        wandb.log(
            {
                "val_loss_mean": val_loss,
            }
        )
        log.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")

        # save model
        if (epoch + 1) % getattr(cfg.train, "save_interval", 2) == 0:
            date_str = datetime.now().strftime("%Y-%m-%d/")
            save_dir = os.path.join(cfg.train.save_path, date_str)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            time_str = datetime.now().strftime("%H-%M-%S")
            save_path = os.path.join(save_dir, f"v1_nhot3d_hand_{epoch + 1}_{time_str}_val_loss_{val_loss:.4f}.pth")
            log.info(f"Save model to {save_path}")
            torch.save(net.state_dict(), save_path)
    log.info("Finish training, iteration: {}".format(iteration))


def main(cfg: HandEstimationConfig) -> None:
    torch.cuda.empty_cache()
    # seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v1_train_nhot3d_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(
        f"cuda:{device_first}" if torch.cuda.is_available() else "cpu"
    )

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

    net = get_hand_model(cfg)
    log.info(f"model parameters: {sum(p.numel() for p in net.parameters())}")
    net = net.to(device)

    if os.path.exists(cfg.train.checkpoint_path):
        state_dict = torch.load(
            cfg.train.checkpoint_path, weights_only=False, map_location=device
        )
        net.load_state_dict(state_dict)
        log.info(f"Load model from {cfg.train.checkpoint_path}")

    train_hand(net, segment_net, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_train_nhot3d.yaml"),
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
