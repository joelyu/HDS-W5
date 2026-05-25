#!/usr/bin/env python3
"""Precompute DinoBloom nucleus-anchored cell masks for handcrafted segmentation.

Per image: extract DinoBloom patch tokens, k-means them into ~4 semantic
clusters (the model's patch features separate nucleus / cytoplasm / red-cells /
background). The nucleus we already detect with the classical multi-Otsu method
anchors the otherwise-anonymous clusters:
  * nucleus cluster   = the cluster overlapping the detected nucleus
  * background cluster = the cluster on the image border
  * cytoplasm cluster = the remaining cluster hugging the nucleus cluster
Cell = (nucleus + cytoplasm) clusters, central connected component. Saved as a
GRIDxGRID cellness map keyed by image_name, consumed by 02b via
cell_mask_dinobloom (upsample -> Otsu -> morphology -> nucleus union -> guard).

This realises DinoBloom's "patch features detect nuclei/cytoplasm/RBC ... could
be leveraged for zero-shot segmentation" claim. The earlier single-PC1 version
took the biggest axis of variation (often a stain/illumination gradient) and
grabbed background; anchoring on the nucleus + using the full patch fingerprints
fixes that.

Needs torch + DinoBloom. Run on laptop MPS / Mac Studio, NOT the singapore VM.

Usage:
    python scripts/02c_dinobloom_cell_scores.py                 # full precompute
    python scripts/02c_dinobloom_cell_scores.py --limit 30      # smoke subset
    python scripts/02c_dinobloom_cell_scores.py --qa            # 13-image QA panels (fast)
    python scripts/02c_dinobloom_cell_scores.py --img-size 448  # finer 32x32 grid
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import binary_dilation, label as cc_label
from sklearn.cluster import KMeans

from config import FOLDER_NAME_MAP, SPLIT_ORDER
from segmentation import segment_nucleus

warnings.filterwarnings("ignore")

PATCH = 14
N_CLUSTERS = 4
NUC_COVERAGE = 0.3  # a patch counts as nucleus if >30% of its area is nucleus


def load_dinobloom_s() -> torch.nn.Module:
    from huggingface_hub import hf_hub_download
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    ckpt = hf_hub_download(repo_id="MarrLab/DinoBloom", filename="pytorch_model_s.bin")
    num_tokens = int(1 + (224 / PATCH) ** 2)  # weights are for 224 input (257 tokens)
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_tokens, 384))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval()
    return model


def get_patch_tokens(model: torch.nn.Module, batch: torch.Tensor, device: str, grid: int) -> np.ndarray:
    """batch (B,3,S,S) -> (B, grid*grid, 384) patch tokens (CLS dropped).

    At img-size != 224 DINOv2 interpolates its position embeddings automatically.
    """
    with torch.no_grad():
        toks = model.get_intermediate_layers(batch.to(device), n=1)[0]  # (B, N, 384)
    if toks.shape[1] == grid * grid + 1:
        toks = toks[:, 1:, :]  # drop CLS
    return toks.cpu().numpy()


def _nucleus_patch_grid(nucleus_mask: np.ndarray, grid: int) -> np.ndarray:
    """Downsample the full-res nucleus mask to a (grid, grid) bool patch map."""
    frac = cv2.resize(nucleus_mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return frac > NUC_COVERAGE


def cluster_cell_map(
    tokens: np.ndarray, nucleus_mask: np.ndarray, grid: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """tokens (grid*grid, 384), nucleus_mask (full res) -> (grid, grid) {0,1} cell map.

    Also returns the (grid, grid) k-means label map for QA visualisation (or
    None if there was no nucleus to anchor on, in which case the cell map is all
    zeros so 02b/cell_mask_dinobloom falls back to the convex hull).
    """
    nuc_grid = _nucleus_patch_grid(nucleus_mask, grid)
    if not nuc_grid.any():
        return np.zeros((grid, grid), np.float32), None

    # L2-normalise tokens (cosine k-means) and cluster.
    t = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8)
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed).fit_predict(t)
    lab = labels.reshape(grid, grid)

    border = np.zeros((grid, grid), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    # nucleus cluster = most overlap with the nucleus patches
    nuc_overlap = [int((lab == c)[nuc_grid].sum()) for c in range(N_CLUSTERS)]
    nucleus_cluster = int(np.argmax(nuc_overlap))
    # background cluster = most overlap with the border ring (excluding nucleus)
    bg_overlap = [
        int(((lab == c) & border).sum()) if c != nucleus_cluster else -1
        for c in range(N_CLUSTERS)
    ]
    background_cluster = int(np.argmax(bg_overlap))
    # cytoplasm cluster = remaining cluster most adjacent to the nucleus cluster
    nuc_dil = binary_dilation(lab == nucleus_cluster)
    remaining = [c for c in range(N_CLUSTERS) if c not in (nucleus_cluster, background_cluster)]
    cytoplasm_cluster = None
    if remaining:
        adj = [int(((lab == c) & nuc_dil).sum()) for c in remaining]
        cytoplasm_cluster = remaining[int(np.argmax(adj))]

    cell = lab == nucleus_cluster
    if cytoplasm_cluster is not None:
        cell = cell | (lab == cytoplasm_cluster)

    # keep the connected component containing the nucleus
    cc, n = cc_label(cell)
    if n > 1:
        overlaps = [int(((cc == i) & nuc_grid).sum()) for i in range(1, n + 1)]
        cell = cc == (int(np.argmax(overlaps)) + 1)

    return cell.astype(np.float32), lab


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Precompute DinoBloom nucleus-anchored cell masks.")
    p.add_argument("--data-dir", type=Path, default=base / "data" / "raw")
    p.add_argument("--results-dir", type=Path, default=base / "results")
    p.add_argument("--limit", type=int, default=None, help="First N images per split (smoke test).")
    p.add_argument("--batch-size", type=int, default=32, help="Images per forward pass.")
    p.add_argument("--img-size", type=int, default=224, help="DinoBloom input size; 448 -> 32x32 grid.")
    p.add_argument("--qa", action="store_true", help="Render 13-image QA panels (fast) instead of the full precompute.")
    p.add_argument("--device", default=None, help="mps|cpu|cuda (auto if omitted).")
    return p.parse_args()


def _transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _img_path(image_root: Path, row) -> Path:
    return image_root / row["path"].split("/")[0] / FOLDER_NAME_MAP[row["cell_type"]] / row["image_name"]


def run_qa(df, image_root, results_dir, model, tf, grid, device) -> int:
    """Render original | nucleus | k-means clusters | cell for one image per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = results_dir / "segmentation_check_dino"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_types = sorted(df["cell_type"].unique())
    rows = [df[df["cell_type"] == ct].iloc[0] for ct in cell_types]

    tensors, imgs = [], []
    for row in rows:
        img = Image.open(_img_path(image_root, row)).convert("RGB")
        tensors.append(tf(img))
        imgs.append(np.array(img))
    tokens = get_patch_tokens(model, torch.stack(tensors), device, grid)

    print(f"QA panels for {len(cell_types)} cell types (grid {grid}x{grid})...")
    for i, ct in enumerate(cell_types):
        nucleus_mask, _ = segment_nucleus(imgs[i])
        cell_map, lab = cluster_cell_map(tokens[i], nucleus_mask, grid)
        H, W = imgs[i].shape[:2]
        cell_full = cv2.resize(cell_map, (W, H), interpolation=cv2.INTER_NEAREST)
        lab_full = cv2.resize(
            (lab if lab is not None else np.zeros((grid, grid))).astype(np.float32),
            (W, H), interpolation=cv2.INTER_NEAREST,
        )
        nc = nucleus_mask.sum() / cell_full.sum() if cell_full.sum() else 0

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(imgs[i]); axes[0].set_title("Original", fontsize=9)
        axes[1].imshow(nucleus_mask, cmap="gray"); axes[1].set_title("Nucleus", fontsize=9)
        axes[2].imshow(lab_full, cmap="tab10", vmin=0, vmax=N_CLUSTERS)
        axes[2].set_title("Patch clusters (k=4)", fontsize=9)
        axes[3].imshow(cell_full, cmap="gray"); axes[3].set_title(f"Cell  N:C={nc:.2f}", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(ct, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ct}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {ct}")
    print(f"\nSaved QA panels to {out_dir}/ — review before the full precompute.")
    return 0


def main() -> int:
    args = parse_args()
    data_dir, results_dir = args.data_dir.resolve(), args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    grid = args.img_size // PATCH

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | img-size {args.img_size} -> {grid}x{grid} grid")

    meta = data_dir / "metadata_with_patient_level_splits.csv"
    if not meta.exists():
        print(f"ERROR: {meta} not found.", file=sys.stderr)
        return 1
    df = pd.read_csv(meta)
    image_root = data_dir / "dataset"

    model = load_dinobloom_s().to(device)
    tf = _transform(args.img_size)

    if args.qa:
        return run_qa(df, image_root, results_dir, model, tf, grid, device)

    out: dict[str, np.ndarray] = {}
    t0 = time.time()
    for split in SPLIT_ORDER:
        sdf = df[df["split"] == split].reset_index(drop=True)
        if args.limit is not None:
            sdf = sdf.head(args.limit)
        n = len(sdf)
        maps, names = [], []
        empty = 0
        print(f"\n{split}: {n} images (batch {args.batch_size})")
        for start in range(0, n, args.batch_size):
            chunk = sdf.iloc[start:start + args.batch_size]
            tensors, imgs = [], []
            for _, row in chunk.iterrows():
                img = Image.open(_img_path(image_root, row)).convert("RGB")
                tensors.append(tf(img))
                imgs.append(np.array(img))
                names.append(row["image_name"])
            tokens = get_patch_tokens(model, torch.stack(tensors), device, grid)
            for b in range(len(chunk)):
                nucleus_mask, _ = segment_nucleus(imgs[b])
                cell_map, lab = cluster_cell_map(tokens[b], nucleus_mask, grid)
                if lab is None:
                    empty += 1
                maps.append(cell_map.astype(np.float16))
            done = min(start + args.batch_size, n)
            print(f"\r  {done}/{n} ({time.time() - t0:.0f}s)", end="", flush=True)
        print()
        if empty:
            print(f"  {empty}/{n} images had no nucleus to anchor on (will fall back to convex hull in 02b)")
        out[f"{split}_scores"] = np.stack(maps).astype(np.float16)
        out[f"{split}_image_name"] = np.array(names)

    out_path = results_dir / "dinobloom_cell_scores.npz"
    np.savez(out_path, **out)
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
