#!/usr/bin/env python3
"""Precompute CellPose cell masks for handcrafted segmentation.

CellPose (Stringer et al., Nature Methods 2021) is a purpose-built instance
cell-segmenter. Per image we run the Cellpose-SAM (cpsam) generalist model, then
pick the central instance as the white blood cell (centred single-cell crops),
and save a full-resolution boolean mask. 02b's `--segmentation cellpose` path
reads these masks; cytoplasm = cell - nucleus, where the nucleus still comes
from the classical multi-Otsu method.

This replaces the DinoBloom patch-feature segmentation, which over-grabbed
background at 14px patch resolution (see the draft note). CellPose gives a real,
full-res cytoplasm boundary so the cytoplasm-dependent features (N:C, cytoplasm
GLCM) become meaningful.

Masks are saved as per-image PNGs in results/cellpose_masks/{image_name}.png
(loaded one at a time in 02b — memory-safe).

Needs `cellpose` >=4.0 (the Cellpose-SAM "cpsam" super-generalist model) + torch.
Run on laptop / Mac Studio, NOT the singapore VM. CPSAM is a SAM-based ViT, so
the full pass is slow — it's a one-time cached precompute.

Usage:
    python scripts/02d_cellpose_masks.py                  # full precompute
    python scripts/02d_cellpose_masks.py --limit 30       # smoke subset
    python scripts/02d_cellpose_masks.py --qa             # 13-image QA panels
    python scripts/02d_cellpose_masks.py --cpu            # force CPU
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import FOLDER_NAME_MAP, SPLIT_ORDER
from segmentation import segment_nucleus

warnings.filterwarnings("ignore")


def load_cellpose(gpu: bool):
    from cellpose import models
    # Cellpose-SAM (cellpose >=4.0): a single super-generalist model — no
    # model_type, no channels, no diameter estimation needed.
    return models.CellposeModel(gpu=gpu)


def segment_instances(model, img: np.ndarray) -> np.ndarray:
    """Run Cellpose-SAM on one RGB image -> (H, W) int label map (0 = background).

    CPSAM takes the RGB image directly (no channels/diameter args). result[0]
    is the label map across versions."""
    result = model.eval(img)
    return np.asarray(result[0])


def central_instance(masks: np.ndarray) -> np.ndarray:
    """Pick the white blood cell: the instance at the image centre (these are
    centred single-cell crops), falling back to the largest central / overall
    instance. Returns a boolean mask."""
    H, W = masks.shape
    centre_lbl = int(masks[H // 2, W // 2])
    if centre_lbl > 0:
        return masks == centre_lbl

    cy0, cy1 = int(H * 0.33), int(H * 0.67)
    cx0, cx1 = int(W * 0.33), int(W * 0.67)
    window = masks[cy0:cy1, cx0:cx1]
    vals = window[window > 0]
    if vals.size:
        return masks == int(np.bincount(vals).argmax())

    allv = masks[masks > 0]
    if allv.size:
        return masks == int(np.bincount(allv).argmax())
    return np.zeros_like(masks, dtype=bool)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Precompute CellPose cell masks.")
    p.add_argument("--data-dir", type=Path, default=base / "data" / "raw")
    p.add_argument("--results-dir", type=Path, default=base / "results")
    p.add_argument("--limit", type=int, default=None, help="First N images per split (smoke test).")
    p.add_argument("--qa", action="store_true", help="Render 13-image QA panels instead of the full precompute.")
    p.add_argument("--cpu", action="store_true", help="Force CPU (default tries GPU/MPS).")
    return p.parse_args()


def _img_path(image_root: Path, row) -> Path:
    return image_root / row["path"].split("/")[0] / FOLDER_NAME_MAP[row["cell_type"]] / row["image_name"]


def run_qa(df, image_root, results_dir, model) -> int:
    """Render original | nucleus | CellPose cell for one image per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = results_dir / "segmentation_check_cellpose"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_types = sorted(df["cell_type"].unique())
    print(f"CellPose QA panels for {len(cell_types)} cell types...")

    for ct in cell_types:
        row = df[df["cell_type"] == ct].iloc[0]
        img = np.array(Image.open(_img_path(image_root, row)).convert("RGB"))
        masks = segment_instances(model, img)
        cell = central_instance(masks)
        nucleus_mask, _ = segment_nucleus(img)
        nc = nucleus_mask.sum() / cell.sum() if cell.sum() else 0
        n_inst = int(masks.max())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img); axes[0].set_title("Original", fontsize=9)
        axes[1].imshow(nucleus_mask, cmap="gray"); axes[1].set_title("Nucleus", fontsize=9)
        axes[2].imshow(cell, cmap="gray"); axes[2].set_title(f"CellPose cell  N:C={nc:.2f}", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{ct}  ({n_inst} instances found)", fontsize=11)
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

    meta = data_dir / "metadata_with_patient_level_splits.csv"
    if not meta.exists():
        print(f"ERROR: {meta} not found.", file=sys.stderr)
        return 1
    df = pd.read_csv(meta)
    image_root = data_dir / "dataset"

    print(f"Loading Cellpose-SAM (cpsam, gpu={not args.cpu})...")
    model = load_cellpose(gpu=not args.cpu)

    if args.qa:
        return run_qa(df, image_root, results_dir, model)

    out_dir = results_dir / "cellpose_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for split in SPLIT_ORDER:
        sdf = df[df["split"] == split].reset_index(drop=True)
        if args.limit is not None:
            sdf = sdf.head(args.limit)
        n = len(sdf)
        empty = 0
        print(f"\n{split}: {n} images")
        for i, row in sdf.iterrows():
            img = np.array(Image.open(_img_path(image_root, row)).convert("RGB"))
            masks = segment_instances(model, img)
            cell = central_instance(masks)
            if not cell.any():
                empty += 1
            Image.fromarray((cell.astype(np.uint8) * 255)).save(out_dir / f"{row['image_name']}.png")
            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"\r  {i + 1}/{n} ({time.time() - t0:.0f}s)", end="", flush=True)
        print()
        if empty:
            print(f"  {empty}/{n} images: CellPose found no cell (02b will fall back to convex hull)")

    print(f"\nSaved masks to {out_dir}/ in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
