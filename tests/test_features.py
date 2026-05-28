import numpy as np
import config
from features import extract_cell_features, glcm_descriptors, extra_morphology

GLCM_KEYS = ["contrast", "correlation", "energy", "homogeneity", "entropy"]


def test_glcm_returns_five_prefixed_keys():
    gray = np.full((20, 20), 100, dtype=np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    out = glcm_descriptors(gray, mask, "nuc")
    assert set(out) == {f"nuc_glcm_{k}" for k in GLCM_KEYS}


def test_glcm_uniform_region_is_smooth():
    # constant region: no intensity change -> zero contrast, max homogeneity
    gray = np.full((20, 20), 100, dtype=np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    out = glcm_descriptors(gray, mask, "nuc")
    assert out["nuc_glcm_contrast"] == 0.0
    assert out["nuc_glcm_homogeneity"] == 1.0
    assert out["nuc_glcm_entropy"] < 1e-6


def test_glcm_checkerboard_is_rough():
    # alternating values -> high contrast, low homogeneity
    gray = np.indices((20, 20)).sum(axis=0) % 2
    gray = (gray * 255).astype(np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    out = glcm_descriptors(gray, mask, "cyt")
    assert out["cyt_glcm_contrast"] > out["cyt_glcm_homogeneity"]


def test_glcm_masking_ignores_background():
    # garbage outside the mask must not change the descriptors. A circular mask
    # leaves background pixels inside the bounding-box crop, exercising the
    # level-0 (background) exclusion path.
    gray = np.full((20, 20), 100, dtype=np.uint8)
    rr, cc = np.ogrid[:20, :20]
    mask = (rr - 10) ** 2 + (cc - 10) ** 2 <= 6 ** 2
    clean = glcm_descriptors(gray.copy(), mask, "nuc")
    noisy = gray.copy()
    noisy[~mask] = np.random.randint(0, 256, size=int((~mask).sum())).astype(np.uint8)
    dirty = glcm_descriptors(noisy, mask, "nuc")
    for k in clean:
        assert abs(clean[k] - dirty[k]) < 1e-9


def test_glcm_empty_mask_returns_zeros():
    gray = np.full((20, 20), 100, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    out = glcm_descriptors(gray, mask, "nuc")
    assert all(v == 0.0 for v in out.values())


def test_nc_ratio_is_nucleus_over_cell():
    cell = np.zeros((40, 40), dtype=bool)
    cell[10:30, 10:30] = True          # 400 px
    nucleus = np.zeros((40, 40), dtype=bool)
    nucleus[15:25, 15:25] = True        # 100 px
    out = extra_morphology(nucleus, cell, lobe_count=1)
    assert abs(out["nc_ratio"] - 0.25) < 1e-9
    assert out["lobe_count"] == 1.0


def test_eccentricity_circle_low_ellipse_high():
    rr, cc = np.ogrid[:60, :60]
    circle = (rr - 30) ** 2 + (cc - 30) ** 2 <= 10 ** 2
    ellipse = ((rr - 30) / 18.0) ** 2 + ((cc - 30) / 6.0) ** 2 <= 1.0
    cell = np.ones((60, 60), dtype=bool)
    ecc_circle = extra_morphology(circle, cell, 1)["nuc_eccentricity"]
    ecc_ellipse = extra_morphology(ellipse, cell, 1)["nuc_eccentricity"]
    assert ecc_circle < 0.4
    assert ecc_ellipse > 0.8


def test_extent_in_unit_interval():
    nucleus = np.zeros((40, 40), dtype=bool)
    nucleus[10:30, 10:20] = True
    out = extra_morphology(nucleus, np.ones((40, 40), bool), 2)
    assert 0.0 < out["nuc_extent"] <= 1.0
    assert out["lobe_count"] == 2.0


# ── Full per-image extractor (moved from 02b into features.py) ────────────────

# The 65 features = 3 shape + 48 colour ratios (4 x 12 channels) + 5 nucleus
# GLCM + 5 cytoplasm GLCM + 4 morphology.
_EXPECTED_N_FEATURES = 65


def test_extract_cell_features_returns_65_named_features():
    # Synthetic centred purple nucleus on a pale background (same shape the
    # segmentation tests use), exercised through the convex-hull path.
    img = np.full((80, 80, 3), 220, dtype=np.uint8)
    img[34:46, 34:46] = (90, 40, 120)
    feats, fell_back = extract_cell_features(img)

    assert len(feats) == _EXPECTED_N_FEATURES
    assert fell_back is False  # convex-hull path never reports a fallback
    for key in ("solidity", "convexity", "circularity", "nc_ratio", "lobe_count",
                "nuc_eccentricity", "nuc_extent", "nuc_glcm_contrast",
                "cyt_glcm_entropy", "ncl_cvx_mean_R", "roc_cvx_std_Cb"):
        assert key in feats
    assert all(isinstance(v, float) for v in feats.values())
    assert not any(np.isnan(v) or np.isinf(v) for v in feats.values())


def test_tavakoli_51_is_subset_of_extracted_features():
    # 03/03b select the Tavakoli-51 baseline *by name* from the 65-feature dict.
    # The 12-channel list + ratio prefixes are duplicated between config and
    # features, so a drift in either would silently break the ablation arm
    # (KeyError, or worse a wrong subset). Assert every Tavakoli name is a real
    # extractor key, and a strict subset of the full 65.
    img = np.full((80, 80, 3), 220, dtype=np.uint8)
    img[34:46, 34:46] = (90, 40, 120)
    feats, _ = extract_cell_features(img)
    keys = set(feats)
    missing = set(config.TAVAKOLI_51) - keys
    assert not missing, f"Tavakoli-51 names absent from extractor output: {sorted(missing)}"
    assert set(config.TAVAKOLI_51) < keys  # 51 is a strict subset of the 65
