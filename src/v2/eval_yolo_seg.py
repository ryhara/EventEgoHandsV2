"""Evaluate the YOLO hand segmentation model with COCO-style mAP metrics.

Runs `model.val()` over a sweep of confidence thresholds and, for EEH-R,
reports light / dark scene categories separately. The dataset is picked with
`--dataset`, the remaining settings are module-level constants:
    python src/v2/eval_yolo_seg.py --dataset real     # EEH-R
    python src/v2/eval_yolo_seg.py --dataset synth    # N-HOT3D

See `eval_yolo_seg_pixel.py` for pixel-level metrics (IoU / Dice / F1) that are
directly comparable with the v1 U-Net segmentation evaluation.
"""

import argparse
import csv
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Datasets (pick one with --dataset)
# ---------------------------------------------------------------------------
DATASETS = {
    # EEH-R (real): filenames are Pxx_yy_frame_*.png, so light/dark can be split
    "real": {
        "name": "real",
        "weights": "/path/to/real.pt",
        "data_path": "/path/to/EEH-R/YOLO/train.yaml",
        "use_light_dark_split": True,
    },
    # N-HOT3D (synthetic): filenames carry no sequence ids, so no light/dark split
    "synth": {
        "name": "synth",
        "weights": "/path/to/synth.pt",
        "data_path": "/path/to/NHOT3D_yolo_seg_lnes_frame_v2/train.yaml",
        "use_light_dark_split": False,
    },
}

# EEH-R dataset root, which ships mapping_id_to_scene.csv (sequence id -> light/dark)
EEHR_DATASET_ROOT = Path("/path/to/EEH-R")
MAPPING_CSV_PATH = EEHR_DATASET_ROOT / "mapping_id_to_scene.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="real",
        help="which dataset preset to evaluate on (default: real)",
    )
    p.add_argument("--weights", help="override the preset's model weights")
    p.add_argument("--data", help="override the preset's YOLO data yaml")
    p.add_argument("--name", help="override the preset's run name (output directory)")
    p.add_argument(
        "--device", type=int, help=f"override the CUDA device (default: {DEVICE})"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2] / "data" / "save" / "yolo"
DEVICE = 0
IMAGE_SIZE = 640
BATCH_SIZE = 32
SEED = 0

# Confidence threshold sweep: 0.1, 0.2, ..., 0.9
CONF_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]


def set_seed(seed: int, device: int = 0) -> None:
    """Set random seeds for reproducibility on a specific CUDA device only."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)


def safe_float(value):
    """Convert various numeric types to float safely."""
    if value is None:
        return None
    if hasattr(value, "__len__"):
        length = len(value)
        if length == 0:
            return None
        return float(value[0])
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (ValueError, RuntimeError):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ==========================================
# Light / Dark mapping helpers
# ==========================================


SEQUENCE_ID_PATTERN = re.compile(r"^(P\d+_\d+)")


def load_light_dark_mapping(csv_path: Path) -> dict:
    """Load mapping CSV: {sequence_id: 'light' or 'dark'}."""
    mapping = {}
    if not csv_path.exists():
        print(f"Warning: mapping CSV not found at {csv_path}")
        return mapping
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["id"]] = row["light_or_dark"]
    return mapping


def get_sequence_id(image_path: str) -> str:
    """Extract sequence id like 'P06_00' from filename 'P06_00_frame_xxx.png'."""
    stem = Path(image_path).stem
    m = SEQUENCE_ID_PATTERN.match(stem)
    return m.group(1) if m else ""


# ==========================================
# Light/Dark subset YAML helpers
# ==========================================


def build_subset_test_dir(
    src_img_dir: Path,
    src_label_dir: Path,
    keep_seq_ids: set,
    out_root: Path,
) -> tuple:
    """Symlink only the images/labels whose sequence id is in keep_seq_ids.

    Returns (subset_img_dir, subset_label_dir, num_images).
    """
    subset_img_dir = out_root / "images" / "test"
    subset_label_dir = out_root / "labels" / "test"
    subset_img_dir.mkdir(parents=True, exist_ok=True)
    subset_label_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        seq_id = get_sequence_id(str(img_path))
        if seq_id not in keep_seq_ids:
            continue
        link_img = subset_img_dir / img_path.name
        if not link_img.exists():
            link_img.symlink_to(img_path)
        label_src = src_label_dir / (img_path.stem + ".txt")
        if label_src.exists():
            link_label = subset_label_dir / label_src.name
            if not link_label.exists():
                link_label.symlink_to(label_src)
        n += 1

    return subset_img_dir, subset_label_dir, n


def write_subset_yaml(
    base_yaml_cfg: dict, subset_root: Path, yaml_out: Path
) -> Path:
    """Write a YOLO data yaml pointing at the subset root."""
    cfg = dict(base_yaml_cfg)
    cfg["path"] = str(subset_root)
    cfg["train"] = "images/test"
    cfg["val"] = "images/test"
    cfg["test"] = "images/test"
    with open(yaml_out, "w") as f:
        yaml.safe_dump(cfg, f)
    return yaml_out


# ==========================================
# model.val() wrapper
# ==========================================


def extract_val_metrics(metrics) -> dict:
    """Extract per-instance metrics from a YOLO val() result object."""
    out = {}
    for branch_name in ("box", "seg"):
        branch = getattr(metrics, branch_name, None)
        if branch is None:
            continue
        out[branch_name] = {
            "precision": safe_float(getattr(branch, "mp", None)),
            "recall": safe_float(getattr(branch, "mr", None)),
            "mAP50": safe_float(getattr(branch, "map50", None)),
            "mAP75": safe_float(getattr(branch, "map75", None)),
            "mAP50-95": safe_float(getattr(branch, "map", None)),
        }
    return out


def run_val(
    model: YOLO,
    data_yaml: str,
    conf: float,
    project: str,
    name: str,
) -> dict:
    """Run YOLO built-in val() with the given conf threshold."""
    metrics = model.val(
        data=data_yaml,
        task="segment",
        split="test",
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=conf,
        seed=SEED,
        project=project,
        name=name,
        exist_ok=True,
        verbose=False,
        plots=False,
        save_json=False,
    )
    return extract_val_metrics(metrics)


# ==========================================
# Main
# ==========================================


def main():
    print("=== YOLO Segmentation Test Evaluation (model.val + conf sweep) ===")

    args = parse_args()
    preset = DATASETS[args.dataset]
    name = args.name or preset["name"]
    data_path = args.data or preset["data_path"]
    use_light_dark_split = preset["use_light_dark_split"]
    if args.device is not None:
        global DEVICE
        DEVICE = args.device

    set_seed(SEED, device=DEVICE)

    model_path = Path(args.weights or preset["weights"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    print(f"Dataset: {args.dataset}")
    print(f"Data yaml: {data_path}")
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    wandb.init(
        project="EventEgoHands-yolo",
        name=f"eval_yolo_seg_{args.dataset}",
    )
    print(f"Seed: {SEED}, Device: {DEVICE}, Image Size: {IMAGE_SIZE}, Batch Size: {BATCH_SIZE}")

    # Load dataset config
    with open(data_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg["path"])
    if not dataset_root.exists():
        fallback_root = Path(data_path).parent
        print(
            f"dataset path {dataset_root} does not exist; "
            f"falling back to yaml-relative root {fallback_root}"
        )
        dataset_root = fallback_root

    test_img_dir = dataset_root / "images" / "test"
    test_label_dir = dataset_root / "labels" / "test"

    # Resolve a yaml the validator can use (must point at an existing root)
    base_yaml_cfg = dict(data_cfg)
    base_yaml_cfg["path"] = str(dataset_root)
    resolved_yaml = PROJECT_DIR / f"{name}_test" / "resolved_data.yaml"
    resolved_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_yaml, "w") as f:
        yaml.safe_dump(base_yaml_cfg, f)

    # Decide categories
    light_dark_mapping = load_light_dark_mapping(MAPPING_CSV_PATH)
    if use_light_dark_split and len(light_dark_mapping) > 0:
        categories = ["all", "light", "dark"]
    else:
        categories = ["all"]
    print(f"Categories to evaluate: {categories}")
    print(f"Mapping entries loaded: {len(light_dark_mapping)}")

    # Prepare per-category yaml files (subset symlinks for light/dark)
    cat_yaml = {"all": str(resolved_yaml)}
    tmp_root = None
    if "light" in categories or "dark" in categories:
        tmp_root = Path(tempfile.mkdtemp(prefix=f"{name}_subset_"))
        print(f"Building light/dark subset symlink trees under {tmp_root}")

        light_ids = {k for k, v in light_dark_mapping.items() if v == "light"}
        dark_ids = {k for k, v in light_dark_mapping.items() if v == "dark"}

        for cat, keep_ids in (("light", light_ids), ("dark", dark_ids)):
            sub_root = tmp_root / cat
            _, _, n_images = build_subset_test_dir(
                test_img_dir, test_label_dir, keep_ids, sub_root
            )
            print(f"  {cat}: {n_images} images")
            if n_images == 0:
                # val() cannot build a dataloader over an empty directory
                print(f"  skipping the {cat} category: no test image matched")
                categories.remove(cat)
                continue
            cat_yaml_path = sub_root / "data.yaml"
            write_subset_yaml(base_yaml_cfg, sub_root, cat_yaml_path)
            cat_yaml[cat] = str(cat_yaml_path)

    # ==========================================
    # Confidence threshold sweep using model.val()
    # ==========================================
    print(f"\n=== Confidence threshold sweep: {CONF_THRESHOLDS} ===")

    out_dir = PROJECT_DIR / f"{name}_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = {}
    for conf in CONF_THRESHOLDS:
        print(f"\n--- conf={conf} ---")
        cat_results = {}
        for cat in categories:
            run_name = f"val_{cat}_conf{int(conf * 100):02d}"
            metrics = run_val(
                model=model,
                data_yaml=cat_yaml[cat],
                conf=conf,
                project=str(out_dir),
                name=run_name,
            )
            cat_results[cat] = metrics

            seg = metrics.get("seg", {})
            box = metrics.get("box", {})
            print(
                f"[conf={conf}][{cat}] "
                f"seg: P={seg.get('precision', 0) or 0:.4f} "
                f"R={seg.get('recall', 0) or 0:.4f} "
                f"mAP50={seg.get('mAP50', 0) or 0:.4f} "
                f"mAP50-95={seg.get('mAP50-95', 0) or 0:.4f} | "
                f"box: mAP50={box.get('mAP50', 0) or 0:.4f} "
                f"mAP50-95={box.get('mAP50-95', 0) or 0:.4f}"
            )

            log_entry = {"conf_threshold": conf}
            for branch, vals in metrics.items():
                for k, v in vals.items():
                    if v is None:
                        continue
                    log_entry[f"{cat}/{branch}/{k}"] = v
            wandb.log(log_entry)

        sweep_results[f"{conf:.1f}"] = cat_results

    # ==========================================
    # Save full results to JSON
    # ==========================================
    out_path = out_dir / "conf_sweep_metrics.json"
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nFull sweep results saved to: {out_path}")

    # ==========================================
    # Summary table: per-category seg mAP50-95
    # ==========================================
    print("\n=== Summary: seg mAP50-95 across confidence thresholds ===")
    header = "conf\t" + "\t".join(categories)
    print(header)
    for conf_str, cat_results in sweep_results.items():
        row = [conf_str]
        for cat in categories:
            v = cat_results[cat].get("seg", {}).get("mAP50-95") or 0.0
            row.append(f"{v:.4f}")
        print("\t".join(row))

    # Push best per category to wandb summary
    for cat in categories:
        best_conf = None
        best_val = -1.0
        for conf_str, cat_results in sweep_results.items():
            v = cat_results[cat].get("seg", {}).get("mAP50-95") or 0.0
            if v > best_val:
                best_val = v
                best_conf = conf_str
        wandb.run.summary[f"{cat}/seg/best_mAP50-95"] = best_val
        wandb.run.summary[f"{cat}/seg/best_conf"] = best_conf

    wandb.finish()

    # Cleanup temp subset trees
    if tmp_root is not None and tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
