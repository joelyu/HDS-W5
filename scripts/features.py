"""Pure feature-computation helpers — morphology + GLCM texture.

Importable and unit-testable without torch or real images. Used by
02b_handcrafted_features.py for the feature extensions beyond Tavakoli (2021).
"""
from __future__ import annotations

import numpy as np
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

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
