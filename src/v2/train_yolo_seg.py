"""Train the YOLO hand segmentation model used by EventEgoHandsV2.

The dataset is picked with `--dataset`; everything else is a module-level
constant, so edit those before running:
    python src/v2/train_yolo_seg.py --dataset real    # EEH-R
    python src/v2/train_yolo_seg.py --dataset synth   # N-HOT3D

Training results are written to `data/save/yolo/<NAME>/weights/best.pt`, which
is what `yolo_checkpoint_path` in `src/v2/config/*.yaml` should point at.
"""

import argparse
from pathlib import Path

import ultralytics.utils.callbacks.wb as wb_cb
import wandb
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Datasets (everything that differs between synthetic and real lives here)
# ---------------------------------------------------------------------------
# Augmentation differs per dataset: real uses weaker scaling and no mosaic,
# because its frames are already recorded from a moving head-mounted camera.
DATASETS = {
    # EEH-R (real)
    "real": {
        "data_path": "/path/to/EEH-R/YOLO/train.yaml",
        "name": "log_real",
        "scale": 0.25,
        "mosaic": 0.0,
    },
    # N-HOT3D (synthetic)
    "synth": {
        "data_path": "/path/to/NHOT3D_yolo_seg_lnes_frame_v2/train.yaml",
        "name": "log_synth",
        "scale": 0.5,
        "mosaic": 0.5,
    },
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2] / "data" / "save" / "yolo"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BASE_MODEL = "yolo26n-seg.pt"  # set to a trained best.pt to finetune
BATCH_SIZE = 64
NBS = 64
EPOCHS = 100
IMAGE_SIZE = 640
WORKERS = 4
DEVICE = 0
SEED = 2

OPTIMIZER = "AdamW"
LR0 = 0.001
PATIENCE = 100
DROPOUT = 0.2
VERBOSE = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="real",
        help="which dataset preset to train on (default: real)",
    )
    p.add_argument("--data", help="override the preset's YOLO data yaml")
    p.add_argument("--name", help="override the preset's run name (output directory)")
    p.add_argument("--epochs", type=int, help=f"override the epoch count (default: {EPOCHS})")
    p.add_argument(
        "--device", type=int, help=f"override the CUDA device (default: {DEVICE})"
    )
    p.add_argument(
        "--fraction",
        type=float,
        help="train on this fraction of the dataset (default: 1.0); a smoke run "
        "uses a small value to get through an epoch quickly",
    )
    return p.parse_args()


args = parse_args()
preset = DATASETS[args.dataset]

DATA_PATH = args.data or preset["data_path"]
NAME = args.name or preset["name"]
SCALE = preset["scale"]
MOSAIC = preset["mosaic"]
if args.epochs is not None:
    EPOCHS = args.epochs
if args.device is not None:
    DEVICE = args.device
FRACTION = args.fraction if args.fraction is not None else 1.0


def custom_wandb_init(trainer):
    wandb.init(
        project="EventEgoHands-yolo",
        name=f"train_yolo_seg_{args.dataset}",
        config=vars(trainer.args),
    )


wb_cb.callbacks["on_pretrain_routine_start"] = custom_wandb_init


print(f"Dataset: {args.dataset}")
print(f"Data yaml: {DATA_PATH}")
print(f"Run name: {NAME}")
print(f"Epochs: {EPOCHS}, Fraction: {FRACTION}")

model = YOLO(BASE_MODEL)

# Training configuration
results = model.train(
    data=DATA_PATH,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    project=str(PROJECT_DIR),
    name=NAME,
    workers=WORKERS,
    pretrained=True,
    patience=PATIENCE,
    optimizer=OPTIMIZER,
    lr0=LR0,
    dropout=DROPOUT,
    verbose=VERBOSE,  # Show logs
    hsv_v=0.9,
    hsv_s=0.7,
    degrees=20,
    translate=0.25,
    scale=SCALE,
    shear=10,
    perspective=0.0005,
    # Left / right hand are separate classes, so horizontal flips are disabled
    fliplr=0.0,
    flipud=0.0,
    bgr=0.0,
    mixup=0.0,
    mosaic=MOSAIC,
    seed=SEED,
    deterministic=True,
    amp=False,
    nbs=NBS,
    fraction=FRACTION,
)

# Evaluation
print("\n=== Validation Set Evaluation ===")
NAME_VAL = f"{NAME}_val"
metrics = model.val(
    data=DATA_PATH,
    project=str(PROJECT_DIR),
    name=NAME_VAL,
    task="segment",
    split="val",
    device=DEVICE,
)
print("\n=== Test Set Evaluation ===")
NAME_TEST = f"{NAME}_test"
metrics_test = model.val(
    data=DATA_PATH,
    project=str(PROJECT_DIR),
    name=NAME_TEST,
    task="segment",
    split="test",
    device=DEVICE,
)

# Print the weights this run actually produced, so the evaluation steps can be
# pointed at them.
best_weights = getattr(model.trainer, "best", PROJECT_DIR / NAME / "weights" / "best.pt")
print(f"\nSave model to {best_weights}")
