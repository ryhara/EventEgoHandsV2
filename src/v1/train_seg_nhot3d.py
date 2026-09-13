import os
import argparse
import logging
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import random
import numpy as np
import wandb
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "hot3d", "hot3d"))

from models import UNet, MaskLoss
from datasets import NHOT3DDatasetForMask
from utils import HandEstimationConfig

log = logging.getLogger(os.path.basename(__file__))


def train_segmentation(net, cfg: HandEstimationConfig, device):
    epochs = cfg.train.epochs

    net.train()

    train_dataset = NHOT3DDatasetForMask(
        event_input_dir=cfg.train.event_input_dir,
        gt_file_dir=cfg.train.gt_file_dir,
        is_get_image=cfg.train.is_get_image,
        augmentation=cfg.train.augmentation,
        is_rotate=False,
        output_size=(cfg.output_height, cfg.output_width),
        in_channels=cfg.n_channels,
        mode=cfg.dataset_mode,
        dataset_divide=cfg.dataset_divide,
        time_bin=cfg.time_bin,
        is_sequence=cfg.is_sequence,
    )
    indices = list(range(len(train_dataset)))
    train_indices, validation_indices = train_test_split(
        indices, test_size=0.2, random_state=cfg.seed
    )
    train_sub_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_sub_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
    )
    validation_dataset = torch.utils.data.Subset(train_dataset, validation_indices)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
    )

    optimizer = optim.Adam(
        net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma
    # )
    # criterion = nn.BCEWithLogitsLoss()
    criterion = MaskLoss(mode="", alpha=cfg.train.alpha)
    # criterion = nn.CrossEntropyLoss()

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
            # events = batch["events"].to(device)
            event_frames = batch["LNES"].to(device)
            gt_masks = batch["gt_mask"].to(device)

            optimizer.zero_grad()
            outputs = net(event_frames)
            losses = criterion(outputs, gt_masks)
            outputs = torch.sigmoid(outputs)


            loss = losses["loss"]
            loss.backward()
            optimizer.step()
            # scheduler.step()
            wandb.log(losses)
            # wandb.log(
            #     {
            #         "loss": loss.item(),
            #     }
            # )
            train_loss += loss.item()

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
                event_frame = batch["LNES"].to(device)
                gt_mask = batch["gt_mask"].to(device)

                outputs = net(event_frame)

                losses = criterion(outputs, gt_mask)
                loss = losses["loss"]
                val_loss += loss.item()
                # wandb.log(
                #     {
                #         "val_loss": loss.item(),
                #     }
                # )

        # validation loss
        val_loss /= len(validation_loader)
        wandb.log(
            {
                "val_loss_mean": val_loss,
            }
        )
        log.info(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")

        # save model
        if (epoch + 1) % cfg.train.save_interval == 0:
            date_str = datetime.now().strftime("%Y-%m-%d")
            save_dir = os.path.join(cfg.train.save_path, date_str)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            time_str = datetime.now().strftime("%H-%M-%S")
            save_path = os.path.join(
                save_dir, f"v1_nhot3d_seg_{epoch + 1}_{time_str}_val_loss_{val_loss:.4f}.pth"
            )
            print(f"Save model to {save_path}")
            torch.save(net.state_dict(), save_path)


def main(cfg: HandEstimationConfig) -> None:
    # seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"v1_train_seg_nhot3d_{time_str}"
    wandb_log = wandb.init(project="EventEgoHands", name=name)
    cfg_json = OmegaConf.to_container(cfg)
    wandb_log.config.update(cfg_json)
    log.info(OmegaConf.to_yaml(cfg))

    device_first = cfg.devices[0]
    device = torch.device(
        f"cuda:{device_first}" if torch.cuda.is_available() else "cpu"
    )

    # Dropout layers shift module indices in state_dict keys, so this must
    # match the value used when the checkpoint was trained.
    net = UNet(
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        dropout_prob=getattr(cfg, "segmentation_dropout_prob", 0.2),
    ).to(device)
    if os.path.exists(cfg.train.checkpoint_path):
        checkpoint = torch.load(
            cfg.train.checkpoint_path, weights_only=False, map_location=device
        )
        net.load_state_dict(checkpoint)
        log.info(f"Load model from {cfg.train.checkpoint_path}")
    train_segmentation(net, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config_train_seg_nhot3d.yaml"),
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
