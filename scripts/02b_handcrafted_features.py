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

The nucleus comes from the shared multi-Otsu segmenter in segmentation.py; the
cell boundary comes from the chosen strategy; cytoplasm = cell - nucleus.

Outputs results/handcrafted_features.npz (convex_hull) or
results/handcrafted_dino_features.npz (dinobloom), in the same format as
02_feature_extraction.py so 03/03b work unchanged.

Usage:
    python scripts/02b_handcrafted_features.py
    python scripts/02b_handcrafted_features.py --segmentation dinobloom --force
    python scripts/02b_handcrafted_features.py --visualise --segmentation dinobloom
    python scripts/02b_handcrafted_features.py --limit 30      # smoke test
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
from PIL import Image
from scipy.spatial import ConvexHull
from skimage import color

from config import FOLDER_NAME_MAP, SPLIT_ORDER
from segmentation import (
    cell_mask_convex_hull,
    cell_mask_dinobloom,
    segment_nucleus,
)
from features import glcm_descriptors, extra_morphology

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

# 12 colour channels in Tavakoli's order: RGB + HSV + LAB + YCrCb
CHANNEL_NAMES = ["R", "G", "B", "H", "S", "V", "L", "A", "Blab", "Y", "Cr", "Cb"]


def _colour_balance(img_rgb: np.ndarray) -> np.ndarray:
    """Grey-world colour balancing (Tavakoli Eq. 1).

    Each channel is scaled so its mean matches the grayscale mean.
    Input and output are float64 [0, 1] RGB images.
    """
    gray = color.rgb2gray(img_rgb)
    gray_mean = gray.mean()
    if gray_mean == 0:
        return img_rgb.copy()
    balanced = np.zeros_like(img_rgb)
    for c in range(3):
        ch = img_rgb[:, :, c]
        ch_mean = ch.mean()
        if ch_mean > 0:
            balanced[:, :, c] = np.clip(ch * gray_mean / ch_mean, 0, 1)
        else:
            balanced[:, :, c] = ch
    return balanced


def _extract_12_channels(img_u8: np.ndarray) -> list[np.ndarray]:
    """Convert a uint8 RGB image to 12 colour channels.

    Returns list of 12 float64 arrays in order:
    R, G, B, H, S, V, L, A, B*, Y, Cr, Cb
    """
    channels = []
    for c in range(3):  # RGB
        channels.append(img_u8[:, :, c].astype(np.float64))
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    for c in range(3):
        channels.append(hsv[:, :, c].astype(np.float64))
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    for c in range(3):
        channels.append(lab[:, :, c].astype(np.float64))
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)
    for c in range(3):
        channels.append(ycrcb[:, :, c].astype(np.float64))
    return channels


def extract_cell_features(
    img_rgb: np.ndarray, dino_score: np.ndarray | None = None
) -> tuple[dict[str, float], bool]:
    """Extract the ~65-feature handcrafted vector for one cell image.

    51 Tavakoli features (3 shape + 48 colour ratios) plus 14 extensions
    (N:C ratio, lobe count, nucleus eccentricity/extent, and rotation-averaged
    GLCM texture for nucleus and cytoplasm).

    If dino_score (a 16x16 cellness map) is given, the cell boundary comes from
    cell_mask_dinobloom; otherwise it is the convex hull of the nucleus.

    Returns (features, fell_back) — fell_back is True when a DinoBloom mask was
    requested but degenerated to the convex-hull fallback.
    """
    img_float = img_rgb.astype(np.float64) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb.copy()

    nucleus_mask, lobe_count = segment_nucleus(img_float)
    if dino_score is not None:
        cvx_mask, fell_back = cell_mask_dinobloom(dino_score, nucleus_mask, img_float.shape[:2])
    else:
        cvx_mask, fell_back = cell_mask_convex_hull(nucleus_mask), False

    nucleus_mask = nucleus_mask & cvx_mask
    roc_mask = cvx_mask & ~nucleus_mask

    features: dict[str, float] = {}

    # ── 3 Tavakoli shape features (nucleus) ──────────────────────────────
    nuc_area = float(nucleus_mask.sum())
    nuc_u8 = nucleus_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(nuc_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nuc_perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    cvx_area, cvx_perimeter = float(cvx_mask.sum()), nuc_perimeter
    if nucleus_mask.any():
        nuc_points = np.argwhere(nucleus_mask)
        if len(nuc_points) >= 3:
            try:
                hull = ConvexHull(nuc_points)
                cvx_area, cvx_perimeter = float(hull.volume), float(hull.area)
            except Exception:
                pass
    features["solidity"] = nuc_area / cvx_area if cvx_area > 0 else 0.0
    features["convexity"] = cvx_perimeter / nuc_perimeter if nuc_perimeter > 0 else 0.0
    features["circularity"] = nuc_perimeter ** 2 / (4 * np.pi * nuc_area) if nuc_area > 0 else 0.0

    # ── 48 Tavakoli colour ratio features ────────────────────────────────
    balanced = _colour_balance(img_float)
    balanced_u8 = (balanced * 255).clip(0, 255).astype(np.uint8)
    channels = _extract_12_channels(balanced_u8)
    has_nuc, has_cvx, has_roc = nucleus_mask.any(), cvx_mask.any(), roc_mask.any()
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ch = channels[ch_idx]
        nuc_vals = ch[nucleus_mask] if has_nuc else np.array([0.0])
        cvx_vals = ch[cvx_mask] if has_cvx else np.array([0.0])
        roc_vals = ch[roc_mask] if has_roc else np.array([0.0])
        nuc_mean, nuc_std = float(nuc_vals.mean()), float(nuc_vals.std())
        cvx_mean, cvx_std = float(cvx_vals.mean()), float(cvx_vals.std())
        roc_mean, roc_std = float(roc_vals.mean()), float(roc_vals.std())
        features[f"ncl_cvx_mean_{ch_name}"] = nuc_mean / cvx_mean if cvx_mean != 0 else 1.0
        features[f"ncl_cvx_std_{ch_name}"] = nuc_std / cvx_std if cvx_std != 0 else 1.0
        features[f"roc_cvx_mean_{ch_name}"] = roc_mean / cvx_mean if cvx_mean != 0 else 1.0
        features[f"roc_cvx_std_{ch_name}"] = roc_std / cvx_std if cvx_std != 0 else 1.0

    # ── 14 extension features ────────────────────────────────────────────
    gray = (color.rgb2gray(balanced) * 255).clip(0, 255).astype(np.uint8)
    features.update(glcm_descriptors(gray, nucleus_mask, "nuc"))
    features.update(glcm_descriptors(gray, roc_mask, "cyt"))
    features.update(extra_morphology(nucleus_mask, cvx_mask, lobe_count))

    return features, fell_back


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
        "--segmentation", choices=["convex_hull", "dinobloom"], default="convex_hull",
        help="Cell-boundary strategy. dinobloom needs results/dinobloom_cell_scores.npz (run 02c first).",
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
            split = row["split"]
            score = {nm: dino_scores[f"{split}_scores"][i]
                     for i, nm in enumerate(dino_scores[f"{split}_image_name"])}.get(row["image_name"])
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
    out_name = "handcrafted_features.npz" if seg == "convex_hull" else "handcrafted_dino_features.npz"
    out_path = results_dir / out_name
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; use --force to re-extract.")
        return 0

    dino_scores = None
    if seg == "dinobloom":
        score_path = results_dir / "dinobloom_cell_scores.npz"
        if not score_path.exists():
            print(f"ERROR: {score_path} not found. Run 02c_dinobloom_cell_scores.py first.", file=sys.stderr)
            return 1
        dino_scores = np.load(score_path)

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
            maps = dino_scores[f"{split}_scores"]
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
                feats, fell_back = extract_cell_features(img, dino_score=score)
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
        if dino_scores is not None:
            total_fallback += fell
            print(f"  DinoBloom fallback to convex hull: {fell}/{n} ({100*fell/max(n,1):.1f}%)")
            if missing:
                print(f"  WARNING: {missing}/{n} images had NO DinoBloom score "
                      f"(used convex hull) — run 02c without --limit for full coverage.")

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
    if dino_scores is not None:
        print(f"Total DinoBloom fallbacks: {total_fallback}")
    results["feature_names"] = np.array(feature_names)
    np.savez(out_path, **results)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
