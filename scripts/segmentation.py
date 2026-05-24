"""Shared cell-segmentation helpers for the handcrafted feature pipeline.

Kept out of config.py so the heavy cv2/skimage imports do not load into the
plotting scripts. Two cell-boundary strategies:
  * cell_mask_convex_hull — Tavakoli's convex-hull-of-nucleus boundary
  * cell_mask_dinobloom    — boundary from a DinoBloom patch-token cellness map
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from skimage import color, measure, morphology
from skimage.filters import threshold_multiotsu, threshold_otsu

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
# skimage 0.26 deprecates binary_*/min_size in morphology; we keep the exact
# calls to preserve the validated Tavakoli-baseline segmentation behaviour
# (the max_size replacement has different off-by-one semantics).
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")


def _rgb_to_cmyk(img_rgb: np.ndarray) -> tuple[np.ndarray, ...]:
    """Convert RGB [0,1] image to CMYK channels (each 0-255 uint8)."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    safe_denom = np.where(k < 1.0, 1.0 - k, 1.0)
    c = ((1.0 - r - k) / safe_denom * 255).clip(0, 255).astype(np.uint8)
    m = ((1.0 - g - k) / safe_denom * 255).clip(0, 255).astype(np.uint8)
    y = ((1.0 - b - k) / safe_denom * 255).clip(0, 255).astype(np.uint8)
    k_u8 = (k * 255).clip(0, 255).astype(np.uint8)
    return c, m, y, k_u8


def segment_nucleus(img_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Segment the central leukocyte nucleus from a single-cell image.

    Adapted from Tavakoli et al. (2021): grey-world balance -> CMYK+HLS
    discriminant -> multi-Otsu -> central blob -> lobe bridging. Identical to
    the previous 02b nucleus logic, but also returns the lobe count (number of
    disconnected nucleus pieces bridged together — high for segmented
    neutrophils, 1 for band/mononuclear cells).

    Args:
        img_rgb: float64 [0,1] RGB image.

    Returns:
        (nucleus_mask, lobe_count)
    """
    rows, cols = img_rgb.shape[:2]
    cy, cx = rows / 2, cols / 2

    img_float = img_rgb if img_rgb.dtype != np.uint8 else img_rgb.astype(np.float64) / 255.0

    gray = color.rgb2gray(img_float)
    gray_mean = gray.mean()
    if gray_mean > 0:
        balanced = np.zeros_like(img_float)
        for c_idx in range(3):
            ch = img_float[:, :, c_idx]
            ch_mean = ch.mean()
            balanced[:, :, c_idx] = (
                np.clip(ch * gray_mean / ch_mean, 0, 1) if ch_mean > 0 else ch
            )
    else:
        balanced = img_float.copy()

    _, m_ch, _, k_ch = _rgb_to_cmyk(balanced)
    img_u8 = (balanced * 255).clip(0, 255).astype(np.uint8)
    hls = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HLS)
    s_ch = hls[:, :, 2]

    m_f, s_f, k_f = m_ch.astype(np.float64), s_ch.astype(np.float64), k_ch.astype(np.float64)
    min_ms = np.minimum(m_f, s_f)
    discriminant = min_ms - np.minimum(min_ms, k_f - np.minimum(k_f, m_f))
    discriminant = np.clip(discriminant, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(discriminant, (5, 5), 0)
    try:
        thresholds = threshold_multiotsu(blurred, classes=3)
        nucleus_binary = blurred >= thresholds[-1]
    except Exception:
        _, thresh_img = cv2.threshold(m_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nucleus_binary = thresh_img > 0

    nucleus_binary = morphology.binary_opening(nucleus_binary, morphology.disk(2))
    nucleus_binary = morphology.remove_small_objects(nucleus_binary, min_size=80)
    nucleus_binary = binary_fill_holes(nucleus_binary)

    nuc_labelled = measure.label(nucleus_binary)
    nuc_props = measure.regionprops(nuc_labelled)

    if not nuc_props:
        img_hsv = color.rgb2hsv(img_float)
        s_hsv, v_hsv = img_hsv[:, :, 1], img_hsv[:, :, 2]
        nucleus_binary = (s_hsv > 0.15) & (v_hsv < 0.65)
        nucleus_binary = morphology.binary_opening(nucleus_binary, morphology.disk(1))
        nucleus_binary = morphology.remove_small_objects(nucleus_binary, min_size=50)
        nucleus_binary = binary_fill_holes(nucleus_binary)
        nuc_labelled = measure.label(nucleus_binary)
        nuc_props = measure.regionprops(nuc_labelled)

    if nuc_props:
        max_area = max(p.area for p in nuc_props)
        nuc_props = [p for p in nuc_props if p.area >= max_area * 0.1]
        best_nuc = min(
            nuc_props,
            key=lambda p: np.sqrt((p.centroid[0] - cy) ** 2 + (p.centroid[1] - cx) ** 2),
        )
        central_nuc = nuc_labelled == best_nuc.label
        bridge = morphology.binary_dilation(central_nuc, morphology.disk(15))
        lobes_region = nucleus_binary & bridge
        lobe_count = int(measure.label(lobes_region).max())
        nucleus_mask = binary_fill_holes(lobes_region)
    else:
        nucleus_mask = np.zeros((rows, cols), dtype=bool)
        lobe_count = 0

    return nucleus_mask, lobe_count


def cell_mask_convex_hull(nucleus_mask: np.ndarray) -> np.ndarray:
    """Cell mask = filled convex hull of the nucleus (Tavakoli boundary)."""
    rows, cols = nucleus_mask.shape
    if not nucleus_mask.any():
        return np.zeros((rows, cols), dtype=bool)
    nuc_points = np.argwhere(nucleus_mask)
    if len(nuc_points) < 3:
        return nucleus_mask.copy()
    try:
        hull = ConvexHull(nuc_points)
        hull_pts_cv = nuc_points[hull.vertices][:, ::-1]  # (col,row) for cv2
        cell = np.zeros((rows, cols), dtype=np.uint8)
        cv2.fillConvexPoly(cell, hull_pts_cv.astype(np.int32), 255)
        return cell > 0
    except Exception:
        return nucleus_mask.copy()


def cell_mask_dinobloom(
    score_map: np.ndarray, nucleus_mask: np.ndarray, image_shape: tuple[int, int]
) -> tuple[np.ndarray, bool]:
    """Cell mask from a DinoBloom 16x16 cellness map.

    The score_map is assumed already oriented so high = foreground (done in
    02c via the centered-cell prior). Upsample (bilinear) to full resolution,
    Otsu-threshold, clean morphologically, keep the foreground component most
    overlapping the nucleus, and union the nucleus in. Falls back to the
    convex hull when the result is degenerate.

    Returns (cell_mask, fell_back).
    """
    rows, cols = image_shape
    score_full = cv2.resize(
        score_map.astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR
    )

    fg = np.zeros((rows, cols), dtype=bool)
    if score_full.max() > score_full.min():
        try:
            fg = score_full > threshold_otsu(score_full)
        except Exception:
            fg = np.zeros((rows, cols), dtype=bool)

    if fg.any():
        fg = morphology.binary_closing(fg, morphology.disk(5))
        fg = morphology.binary_opening(fg, morphology.disk(3))
        fg = binary_fill_holes(fg)

        # Keep the foreground component with the most nucleus overlap.
        labelled = measure.label(fg)
        if labelled.max() >= 1 and nucleus_mask.any():
            best, best_overlap = None, 0
            for r in range(1, labelled.max() + 1):
                comp = labelled == r
                ov = int((comp & nucleus_mask).sum())
                if ov > best_overlap:
                    best, best_overlap = comp, ov
            if best is not None:
                fg = best

    cell = fg | nucleus_mask
    area = int(cell.sum())
    frac = area / (rows * cols)
    nuc_area = int(nucleus_mask.sum())

    degenerate = (
        not fg.any()
        or frac < 0.01
        or frac > 0.90
        or (nuc_area > 0 and area < nuc_area * 1.05)  # ~no cytoplasm gained
    )
    if degenerate:
        return cell_mask_convex_hull(nucleus_mask), True
    return cell, False
