#!/usr/bin/env python3
"""Precompute DinoBloom-S cellness maps for handcrafted segmentation.

For each image: DinoBloom-S patch tokens -> per-image PCA -> first component ->
16x16 cellness map, oriented so high = central cell (centered-cell prior).
Saved to results/dinobloom_cell_scores.npz keyed by image_name.

Needs torch + DinoBloom. Run on laptop MPS / Mac Studio, NOT the singapore VM.

Usage:
    python scripts/02c_dinobloom_cell_scores.py            # full
    python scripts/02c_dinobloom_cell_scores.py --limit 30 # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA

from config import FOLDER_NAME_MAP, SPLIT_ORDER

warnings.filterwarnings("ignore")

IMG, PATCH, GRID = 224, 14, 16


def load_dinobloom_s() -> torch.nn.Module:
    from huggingface_hub import hf_hub_download
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    ckpt = hf_hub_download(repo_id="MarrLab/DinoBloom", filename="pytorch_model_s.bin")
    num_tokens = int(1 + (IMG / PATCH) ** 2)
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_tokens, 384))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval()
    return model


def cellness_map(model: torch.nn.Module, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    """Return an oriented 16x16 float16 cellness map (high = central cell)."""
    with torch.no_grad():
        toks = model.get_intermediate_layers(img_tensor.to(device), n=1)[0]  # (1, N, 384)
    n = toks.shape[1]
    patch = (toks[0, 1:, :] if n == GRID * GRID + 1 else toks[0, :, :]).cpu().numpy()
    pc1 = PCA(n_components=1).fit_transform(patch)[:, 0].reshape(GRID, GRID)
    center = pc1[GRID // 2 - 2:GRID // 2 + 2, GRID // 2 - 2:GRID // 2 + 2].mean()
    border = np.concatenate([pc1[0, :], pc1[-1, :], pc1[:, 0], pc1[:, -1]]).mean()
    if center < border:
        pc1 = -pc1  # orient so the central cell is the high-value foreground
    return pc1.astype(np.float16)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Precompute DinoBloom cellness maps.")
    p.add_argument("--data-dir", type=Path, default=base / "data" / "raw")
    p.add_argument("--results-dir", type=Path, default=base / "results")
    p.add_argument("--limit", type=int, default=None, help="First N images per split (smoke test).")
    p.add_argument("--device", default=None, help="mps|cpu|cuda (auto if omitted).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir, results_dir = args.data_dir.resolve(), args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    meta = data_dir / "metadata_with_patient_level_splits.csv"
    if not meta.exists():
        print(f"ERROR: {meta} not found.", file=sys.stderr)
        return 1
    df = pd.read_csv(meta)
    image_root = data_dir / "dataset"

    model = load_dinobloom_s().to(device)
    tf = T.Compose([
        T.Resize((IMG, IMG)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    out: dict[str, np.ndarray] = {}
    t0 = time.time()
    for split in SPLIT_ORDER:
        sdf = df[df["split"] == split].reset_index(drop=True)
        if args.limit is not None:
            sdf = sdf.head(args.limit)
        n = len(sdf)
        maps, names = [], []
        print(f"\n{split}: {n} images")
        for i, row in sdf.iterrows():
            original_split = row["path"].split("/")[0]
            img_path = image_root / original_split / FOLDER_NAME_MAP[row["cell_type"]] / row["image_name"]
            img = Image.open(img_path).convert("RGB")
            tensor = tf(img).unsqueeze(0)
            maps.append(cellness_map(model, tensor, device))
            names.append(row["image_name"])
            if (i + 1) % 500 == 0 or (i + 1) == n:
                print(f"\r  {i + 1}/{n} ({time.time() - t0:.0f}s)", end="", flush=True)
        print()
        out[f"{split}_scores"] = np.stack(maps).astype(np.float16)
        out[f"{split}_image_name"] = np.array(names)

    out_path = results_dir / "dinobloom_cell_scores.npz"
    np.savez(out_path, **out)
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
