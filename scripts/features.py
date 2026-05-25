"""Handcrafted feature computation for blood-cell images (torch-free).

Two layers:
  * pure per-region helpers — glcm_descriptors (Haralick texture) and
    extra_morphology (N:C ratio, lobe count, eccentricity, extent);
  * extract_cell_features — the full per-image extractor producing the
    65-feature vector (51 Tavakoli + 14 extensions).

extract_cell_features is shared by 02b (KU-Optofil) and 07 (Acevedo external
validation), so both datasets get an identical feature definition. Imports
segmentation.py for the cell-boundary strategies; no torch.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from skimage import color, measure
from skimage.feature import graycomatrix, graycoprops

from segmentation import (
    cell_mask_cellpose,
    cell_mask_convex_hull,
    cell_mask_dinobloom,
    segment_nucleus,
)

GLCM_LEVELS = 32  # gray levels for quantisation (level 0 reserved for background)
GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_GLCM_KEYS = ["contrast", "correlation", "energy", "homogeneity", "entropy"]


def glcm_descriptors(gray: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    """Rotation-averaged Haralick descriptors over the masked region.

    Args:
        gray: 2-D uint8 image.
        mask: boolean array (same shape) — True inside the region of interest.
        prefix: feature-name prefix, e.g. "nuc" or "cyt".

    Returns dict with keys {prefix}_glcm_{contrast,correlation,energy,
    homogeneity,entropy}. GLCM is computed at distance 1 over 4 angles
    (0/45/90/135 deg) and averaged over angle. Background pixels (outside
    the mask) are quantised to level 0 and excluded from the co-occurrence
    counts so they cannot pollute the texture statistics.
    """
    out = {f"{prefix}_glcm_{k}": 0.0 for k in _GLCM_KEYS}
    if mask.sum() < 2:
        return out

    # Crop to the region's bounding box to keep the GLCM small.
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    region = gray[y0:y1, x0:x1]
    rmask = mask[y0:y1, x0:x1]

    vals = region[rmask].astype(np.float64)
    vmin, vmax = vals.min(), vals.max()

    # Quantise masked pixels into levels 1..GLCM_LEVELS; background stays 0.
    q = np.zeros_like(region, dtype=np.uint8)
    if vmax > vmin:
        scaled = 1 + np.floor(
            (region.astype(np.float64) - vmin) / (vmax - vmin) * (GLCM_LEVELS - 1)
        )
        scaled = np.clip(scaled, 1, GLCM_LEVELS).astype(np.uint8)
        q[rmask] = scaled[rmask]
    else:
        q[rmask] = 1  # uniform region

    levels = GLCM_LEVELS + 1  # 0..GLCM_LEVELS inclusive
    glcm = graycomatrix(
        q, distances=[1], angles=GLCM_ANGLES, levels=levels,
        symmetric=True, normed=False,
    ).astype(np.float64)

    # Drop background (level 0) co-occurrences, then renormalise per angle.
    glcm[0, :, :, :] = 0.0
    glcm[:, 0, :, :] = 0.0
    for a in range(glcm.shape[3]):
        s = glcm[:, :, 0, a].sum()
        if s > 0:
            glcm[:, :, 0, a] /= s

    out[f"{prefix}_glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
    corr = graycoprops(glcm, "correlation")
    out[f"{prefix}_glcm_correlation"] = float(np.nan_to_num(corr).mean())
    out[f"{prefix}_glcm_energy"] = float(graycoprops(glcm, "energy").mean())
    out[f"{prefix}_glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())

    entropies = []
    for a in range(glcm.shape[3]):
        p = glcm[:, :, 0, a]
        entropies.append(float(-(p * np.log(p + 1e-10)).sum()))
    out[f"{prefix}_glcm_entropy"] = float(np.mean(entropies))
    return out


def extra_morphology(
    nucleus_mask: np.ndarray, cell_mask: np.ndarray, lobe_count: int
) -> dict[str, float]:
    """Morphology features beyond Tavakoli: N:C ratio, lobe count, eccentricity, extent.

    Args:
        nucleus_mask: boolean nucleus mask.
        cell_mask: boolean cell (whole-cell) mask.
        lobe_count: number of nucleus lobes (pre-bridge components), from
            segmentation.segment_nucleus.

    Returns dict: nc_ratio, lobe_count, nuc_eccentricity, nuc_extent.
    """
    nuc_area = float(nucleus_mask.sum())
    cell_area = float(cell_mask.sum())
    out = {
        "nc_ratio": nuc_area / cell_area if cell_area > 0 else 0.0,
        "lobe_count": float(lobe_count),
        "nuc_eccentricity": 0.0,
        "nuc_extent": 0.0,
    }
    if nucleus_mask.any():
        props = measure.regionprops(nucleus_mask.astype(np.int32))
        if props:
            p = max(props, key=lambda r: r.area)
            out["nuc_eccentricity"] = float(p.eccentricity)
            out["nuc_extent"] = float(p.extent)
    return out


# ── Full per-image extractor (51 Tavakoli + 14 extensions = 65 features) ──────

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
    img_rgb: np.ndarray,
    dino_score: np.ndarray | None = None,
    cellpose_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], bool]:
    """Extract the ~65-feature handcrafted vector for one cell image.

    51 Tavakoli features (3 shape + 48 colour ratios) plus 14 extensions
    (N:C ratio, lobe count, nucleus eccentricity/extent, and rotation-averaged
    GLCM texture for nucleus and cytoplasm).

    Cell boundary: cellpose_mask (full-res CellPose mask) takes priority, else
    dino_score (16x16 cellness map) via cell_mask_dinobloom, else the convex
    hull of the nucleus. The nucleus always comes from the classical segmenter.

    Returns (features, fell_back) — fell_back is True when a requested model
    mask degenerated to the convex-hull fallback.
    """
    img_float = img_rgb.astype(np.float64) / 255.0 if img_rgb.dtype == np.uint8 else img_rgb.copy()

    nucleus_mask, lobe_count = segment_nucleus(img_float)
    if cellpose_mask is not None:
        cvx_mask, fell_back = cell_mask_cellpose(cellpose_mask, nucleus_mask)
    elif dino_score is not None:
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
