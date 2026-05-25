#!/usr/bin/env python3
"""Extract handcrafted features following Tavakoli et al. (2021), plus extensions.

Produces ~65 features per cell:
    51 Tavakoli features (3 nucleus shape + 48 colour ratios across 12 channels)
    +4 morphology: N:C ratio, lobe count, nucleus eccentricity, nucleus extent
    +5 nucleus chromatin GLCM (rotation-averaged Haralick)
    +5 cytoplasm granularity GLCM (rotation-averaged Haralick)

Segmentation is pluggable via --segmentation:
    convex_hull  — Tavakoli's convex-hull-of-nucleus boundary (default)
    dinobloom    — boundary from DinoBloom cellness maps (run 02c first)
    cellpose     — boundary from CellPose masks (run 02d first)

The nucleus comes from the shared multi-Otsu segmenter in segmentation.py; the
cell boundary comes from the chosen strategy; cytoplasm = cell - nucleus. The
per-image feature computation lives in features.extract_cell_features, shared
with 07 (Acevedo external validation) so both datasets are featurised identically.

Outputs results/handcrafted_features.npz (convex_hull),
results/handcrafted_dino_features.npz (dinobloom), or
results/handcrafted_cellpose_features.npz (cellpose), in the same format as
02_feature_extraction.py so 03/03b work unchanged.

Usage:
    python scripts/02b_handcrafted_features.py
    python scripts/02b_handcrafted_features.py --segmentation cellpose --force
    python scripts/02b_handcrafted_features.py --visualise --segmentation dinobloom
    python scripts/02b_handcrafted_features.py --limit 30      # smoke test
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
from segmentation import cell_mask_convex_hull, cell_mask_dinobloom, segment_nucleus
from features import extract_cell_features

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract handcrafted features from blood cell images."
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument(
        "--segmentation", choices=["convex_hull", "dinobloom", "cellpose"], default="convex_hull",
        help="Cell-boundary strategy. dinobloom needs results/dinobloom_cell_scores.npz (02c); "
             "cellpose needs results/cellpose_masks/ (02d).",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract even if the output exists.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images per split (smoke test).")
    parser.add_argument(
        "--visualise", action="store_true",
        help="Save per-class segmentation overlays to results/segmentation_check/ "
             "instead of running full feature extraction.",
    )
    return parser.parse_args()


# ── Visualisation mode ────────────────────────────────────────────────────


def run_visualisation(df, image_root, results_dir, dino_scores=None):
    """Save per-class panels: original | nucleus | convex-hull cell | DinoBloom cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = results_dir / "segmentation_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_types = sorted(df["cell_type"].unique())
    print(f"Generating segmentation panels for {len(cell_types)} cell types...")

    # Build the image_name -> 16x16 cellness lookup ONCE. dino_scores[key] on a
    # loaded .npz is lazy and re-reads the whole array on every access, so doing
    # this inside the loop reloads the full array thousands of times (OOM).
    score_by_name = {}
    if dino_scores is not None:
        for sp in SPLIT_ORDER:
            sp_names = dino_scores[f"{sp}_image_name"]
            sp_maps = dino_scores[f"{sp}_scores"]  # (N, grid, grid)
            for i, nm in enumerate(sp_names):
                score_by_name[nm] = sp_maps[i]

    for ct in cell_types:
        row = df[df["cell_type"] == ct].iloc[0]
        class_folder = FOLDER_NAME_MAP[ct]
        original_split = row["path"].split("/")[0]
        img_path = image_root / original_split / class_folder / row["image_name"]
        img = np.array(Image.open(img_path).convert("RGB"))
        img_float = img.astype(np.float64) / 255.0

        nucleus_mask, lobe_count = segment_nucleus(img_float)
        hull_cell = cell_mask_convex_hull(nucleus_mask)
        nc_hull = nucleus_mask.sum() / hull_cell.sum() if hull_cell.sum() else 0

        panels = [("Original", img), ("Nucleus", nucleus_mask),
                  (f"Convex hull  N:C={nc_hull:.2f}", hull_cell)]
        if dino_scores is not None:
            score = score_by_name.get(row["image_name"])
            if score is not None:
                dino_cell, fell = cell_mask_dinobloom(score, nucleus_mask, img_float.shape[:2])
                nc_dino = nucleus_mask.sum() / dino_cell.sum() if dino_cell.sum() else 0
                tag = f"DinoBloom{' (fallback)' if fell else ''}  N:C={nc_dino:.2f}"
                panels.append((tag, dino_cell))

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        for ax, (title, im) in zip(axes, panels):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{ct}  (lobes={lobe_count})", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ct}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {ct}")
    print(f"\nSaved panels to {out_dir}/ — review before the full run.")
    return 0


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    results_dir: Path = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / "metadata_with_patient_level_splits.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.", file=sys.stderr)
        return 1

    df = pd.read_csv(meta_path)
    image_root = data_dir / "dataset"
    print(f"Loaded metadata: {len(df)} images, {df['patient_id'].nunique()} patients")

    if args.visualise:
        dino_scores = None
        if args.segmentation == "dinobloom":
            score_path = results_dir / "dinobloom_cell_scores.npz"
            if score_path.exists():
                dino_scores = np.load(score_path)
            else:
                print("WARNING: no dinobloom_cell_scores.npz; showing convex hull only.")
        return run_visualisation(df, image_root, results_dir, dino_scores)

    seg = args.segmentation
    out_name = {
        "convex_hull": "handcrafted_features.npz",
        "dinobloom": "handcrafted_dino_features.npz",
        "cellpose": "handcrafted_cellpose_features.npz",
    }[seg]
    out_path = results_dir / out_name
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; use --force to re-extract.")
        return 0

    dino_scores = None
    cellpose_dir = None
    if seg == "dinobloom":
        score_path = results_dir / "dinobloom_cell_scores.npz"
        if not score_path.exists():
            print(f"ERROR: {score_path} not found. Run 02c_dinobloom_cell_scores.py first.", file=sys.stderr)
            return 1
        dino_scores = np.load(score_path)
    elif seg == "cellpose":
        cellpose_dir = results_dir / "cellpose_masks"
        if not cellpose_dir.exists():
            print(f"ERROR: {cellpose_dir} not found. Run 02d_cellpose_masks.py first.", file=sys.stderr)
            return 1

    results = {}
    feature_names = None
    total_fallback = 0
    t0 = time.time()

    for split in SPLIT_ORDER:
        split_df = df[df["split"] == split].reset_index(drop=True)
        if args.limit is not None:
            split_df = split_df.head(args.limit)
        n = len(split_df)
        print(f"\n{'='*60}\n  {split} ({seg}): {n} images\n{'='*60}")

        score_lookup = {}
        if dino_scores is not None:
            names = dino_scores[f"{split}_image_name"]
            maps = dino_scores[f"{split}_scores"]  # (N, grid, grid)
            score_lookup = {nm: maps[i] for i, nm in enumerate(names)}

        all_features, all_labels, all_patients = [], [], []
        failed = fell = missing = 0

        for i, row in split_df.iterrows():
            class_folder = FOLDER_NAME_MAP[row["cell_type"]]
            original_split = row["path"].split("/")[0]
            img_path = image_root / original_split / class_folder / row["image_name"]
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                score = score_lookup.get(row["image_name"]) if dino_scores is not None else None
                if dino_scores is not None and score is None:
                    missing += 1
                cp_mask = None
                if cellpose_dir is not None:
                    mp = cellpose_dir / f"{row['image_name']}.png"
                    if mp.exists():
                        cp_mask = np.array(Image.open(mp).convert("L")) > 0
                    else:
                        missing += 1
                feats, fell_back = extract_cell_features(img, dino_score=score, cellpose_mask=cp_mask)
                fell += int(fell_back)
                if feature_names is None:
                    feature_names = sorted(feats.keys())
                all_features.append(np.array([feats[k] for k in feature_names], dtype=np.float64))
                all_labels.append(row["cell_type"])
                all_patients.append(str(row["patient_id"]))
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  WARNING: Failed on {img_path}: {e}")
            if (i + 1) % 500 == 0 or (i + 1) == n:
                print(f"\r  {split}: {i + 1}/{n} ({time.time() - t0:.0f}s)", end="", flush=True)

        print()
        if failed:
            print(f"  WARNING: {failed} images failed extraction")
        if dino_scores is not None or cellpose_dir is not None:
            total_fallback += fell
            label = "DinoBloom" if dino_scores is not None else "CellPose"
            print(f"  {label} fallback to convex hull: {fell}/{n} ({100*fell/max(n,1):.1f}%)")
            if missing:
                print(f"  WARNING: {missing}/{n} images had NO {label} mask "
                      f"(used convex hull) — run 02c/02d without --limit for full coverage.")

        X = np.stack(all_features)
        nan_count = int(np.isnan(X).sum() + np.isinf(X).sum())
        if nan_count:
            print(f"  Replacing {nan_count} NaN/inf with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        results[f"{split}_X"] = X
        results[f"{split}_y"] = np.array(all_labels)
        results[f"{split}_patients"] = np.array(all_patients)
        print(f"  Shape: {X.shape}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min | dim: {len(feature_names)}")
    if dino_scores is not None or cellpose_dir is not None:
        print(f"Total fallbacks to convex hull: {total_fallback}")
    results["feature_names"] = np.array(feature_names)
    np.savez(out_path, **results)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
