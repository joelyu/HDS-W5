#!/usr/bin/env python3
"""Extract features from frozen pretrained backbones.

Loads each backbone in eval mode (no training), passes all images through,
and saves the resulting feature vectors as .npz files for downstream XGBoost.

Backbones:
    1. ResNet-50     (ImageNet)    → 2048-dim
    2. EfficientNet-B0 (ImageNet)  → 1280-dim
    3. DinoBloom-S   (hematology)  → 384-dim

Outputs saved to results/:
    resnet50_features.npz
    efficientnet_b0_features.npz
    dinobloom_s_features.npz

Each .npz contains: train_X,      train_y,      train_patients,
                     validation_X, validation_y, validation_patients,
                     test_X,       test_y,       test_patients

Usage:
    python scripts/02_feature_extraction.py
    python scripts/02_feature_extraction.py --backbone resnet50      # single backbone
    python scripts/02_feature_extraction.py --backbone dinobloom_s
    python scripts/02_feature_extraction.py --batch-size 64
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

# Suppress harmless warnings from torchvision (libjpeg) and DINOv2 (xFormers)
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import FOLDER_NAME_MAP, SPLIT_ORDER

# ImageNet normalisation — used for all three backbones
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Dataset ─────────────────────────────────────────────────────────────────


class PBCDataset(Dataset):
    """Simple dataset that loads images from metadata rows."""

    def __init__(self, df: pd.DataFrame, image_root: Path, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        row = self.df.iloc[idx]
        class_folder = FOLDER_NAME_MAP[row["cell_type"]]
        original_split = row["path"].split("/")[0]
        img_path = self.image_root / original_split / class_folder / row["image_name"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, row["cell_type"], str(row["patient_id"])


# ── Backbone loading ────────────────────────────────────────────────────────


def load_resnet50() -> tuple[torch.nn.Module, int]:
    """ResNet-50 with ImageNet weights, classifier head removed."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, 2048


def load_efficientnet_b0() -> tuple[torch.nn.Module, int]:
    """EfficientNet-B0 with ImageNet weights, classifier head removed."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model, 1280


def load_dinobloom_s() -> tuple[torch.nn.Module, int]:
    """DinoBloom-S: base DINOv2 ViT-S/14 + DinoBloom hematology weights."""
    from huggingface_hub import hf_hub_download

    # Load base DINOv2 ViT-S architecture
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    # Download and apply DinoBloom-S weights
    ckpt_path = hf_hub_download(repo_id="MarrLab/DinoBloom", filename="pytorch_model_s.bin")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # DinoBloom uses 224×224 input → 16×16 patches of size 14 → 256 + 1 CLS = 257 tokens
    num_tokens = int(1 + (224 / 14) ** 2)
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_tokens, 384))
    model.load_state_dict(ckpt, strict=True)

    model.eval()
    return model, 384


BACKBONES = {
    "resnet50": load_resnet50,
    "efficientnet_b0": load_efficientnet_b0,
    "dinobloom_s": load_dinobloom_s,
}


# ── Feature extraction ──────────────────────────────────────────────────────


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_dim: int,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from a frozen backbone. Returns (features, labels, patient_ids)."""
    model.to(device)

    all_features = []
    all_labels = []
    all_patients = []
    n_batches = len(dataloader)

    with torch.no_grad():
        for i, (images, labels, patient_ids) in enumerate(dataloader):
            features = model(images.to(device))
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels)
            all_patients.extend(patient_ids)

            if (i + 1) % 20 == 0 or (i + 1) == n_batches:
                print(f"\r  {desc} batch {i + 1}/{n_batches}", end="", flush=True)

    print()
    return (
        np.concatenate(all_features),
        np.array(all_labels),
        np.array(all_patients),
    )


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features from frozen backbones.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONES.keys()),
        default=None,
        help="Extract for a single backbone (default: all three).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'mps', 'cuda', 'cpu', or 'auto' (default: auto).",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    results_dir: Path = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")

    # ── Load metadata ───────────────────────────────────────────────────
    meta_path = data_dir / "metadata_with_patient_level_splits.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.", file=sys.stderr)
        return 1

    df = pd.read_csv(meta_path)
    image_root = data_dir / "dataset"
    if not image_root.exists():
        print(f"ERROR: {image_root} not found. Run 01_data_exploration.py first to unzip.", file=sys.stderr)
        return 1

    print(f"Loaded metadata: {len(df)} images, {df['patient_id'].nunique()} patients")

    # ── Build dataloaders per split ─────────────────────────────────────
    loaders = {}
    for split in SPLIT_ORDER:
        split_df = df[df["split"] == split]
        dataset = PBCDataset(split_df, image_root, TRANSFORM)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        print(f"  {split:12s}: {len(split_df):6d} images")
    print()

    # ── Extract features per backbone ───────────────────────────────────
    backbones_to_run = [args.backbone] if args.backbone else list(BACKBONES.keys())

    for backbone_name in backbones_to_run:
        out_path = results_dir / f"{backbone_name}_features.npz"
        if out_path.exists():
            print(f"[{backbone_name}] Already exists at {out_path}, skipping.")
            print(f"  Delete the file to re-extract.")
            print()
            continue

        print(f"[{backbone_name}] Loading model...")
        model, feature_dim = BACKBONES[backbone_name]()
        print(f"  Feature dim: {feature_dim}")

        results = {}
        t0 = time.time()

        for split in SPLIT_ORDER:
            X, y, patients = extract_features(
                model, loaders[split], device, feature_dim,
                desc=f"{backbone_name}/{split}",
            )
            results[f"{split}_X"] = X
            results[f"{split}_y"] = y
            results[f"{split}_patients"] = patients
            print(f"  {split:12s}: {X.shape}")

        elapsed = time.time() - t0
        print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        np.savez(out_path, **results)
        print(f"  Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        print()

        # Free GPU memory before next backbone
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
