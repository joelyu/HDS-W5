import numpy as np
from segmentation import (
    cell_mask_cellpose,
    cell_mask_convex_hull,
    cell_mask_dinobloom,
    segment_nucleus,
)


def test_convex_hull_is_superset_of_nucleus():
    nucleus = np.zeros((50, 50), dtype=bool)
    nucleus[10:20, 10:20] = True
    nucleus[30:40, 30:40] = True  # two blobs -> hull spans both
    cell = cell_mask_convex_hull(nucleus)
    assert cell[nucleus].all()
    assert cell.sum() > nucleus.sum()


def test_convex_hull_empty_nucleus_returns_empty():
    nucleus = np.zeros((50, 50), dtype=bool)
    cell = cell_mask_convex_hull(nucleus)
    assert cell.sum() == 0


def test_segment_nucleus_returns_mask_and_lobecount():
    # synthetic centred purple blob on a pale background
    img = np.full((80, 80, 3), 220, dtype=np.uint8)
    img[34:46, 34:46] = (90, 40, 120)  # dark purple nucleus near centre
    mask, lobe_count = segment_nucleus(img.astype(np.float64) / 255.0)
    assert mask.dtype == bool
    assert mask.shape == (80, 80)
    assert isinstance(lobe_count, int)
    assert mask.sum() > 0


def _nucleus_blob(shape=(224, 224)):
    m = np.zeros(shape, dtype=bool)
    cy, cx = shape[0] // 2, shape[1] // 2
    m[cy - 15:cy + 15, cx - 15:cx + 15] = True
    return m


def test_dino_mask_matches_image_shape_and_contains_nucleus():
    score = np.zeros((16, 16), dtype=np.float32)
    score[5:11, 5:11] = 1.0  # high cellness in the centre
    nucleus = _nucleus_blob()
    cell, fell_back = cell_mask_dinobloom(score, nucleus, (224, 224))
    assert cell.shape == (224, 224)
    assert cell[nucleus].all()        # nucleus always inside the cell
    assert not fell_back
    assert cell.sum() > nucleus.sum()  # cytoplasm is non-empty


def test_dino_mask_degenerate_falls_back_to_hull():
    score = np.zeros((16, 16), dtype=np.float32)  # flat -> no foreground
    nucleus = _nucleus_blob()
    cell, fell_back = cell_mask_dinobloom(score, nucleus, (224, 224))
    assert fell_back
    assert cell[nucleus].all()


def test_dino_mask_full_frame_foreground_falls_back():
    score = np.ones((16, 16), dtype=np.float32)  # everything "foreground"
    nucleus = _nucleus_blob()
    cell, fell_back = cell_mask_dinobloom(score, nucleus, (224, 224))
    assert fell_back


def test_dino_mask_background_grab_falls_back():
    # attention concentrated in the bottom half = off-centre and touching the
    # bottom border -> the guard should reject it and fall back to convex hull.
    score = np.zeros((16, 16), dtype=np.float32)
    score[9:, :] = 1.0
    nucleus = _nucleus_blob()
    cell, fell_back = cell_mask_dinobloom(score, nucleus, (224, 224))
    assert fell_back
    assert cell[nucleus].all()


def test_cellpose_mask_contains_nucleus_and_gains_cytoplasm():
    nucleus = _nucleus_blob()
    cp = np.zeros((224, 224), dtype=bool)
    cp[80:144, 80:144] = True  # larger than the nucleus, overlapping it
    cell, fell_back = cell_mask_cellpose(cp, nucleus)
    assert not fell_back
    assert cell[nucleus].all()
    assert cell.sum() > nucleus.sum()


def test_cellpose_mask_none_falls_back():
    nucleus = _nucleus_blob()
    cell, fell_back = cell_mask_cellpose(None, nucleus)
    assert fell_back
    assert cell[nucleus].all()


def test_cellpose_mask_equal_to_nucleus_falls_back():
    nucleus = _nucleus_blob()
    cell, fell_back = cell_mask_cellpose(nucleus.copy(), nucleus)  # no cytoplasm gained
    assert fell_back


def test_cellpose_mask_resized_when_shape_differs():
    nucleus = _nucleus_blob((224, 224))
    cp = np.zeros((112, 112), dtype=bool)  # half-res CellPose mask
    cp[40:72, 40:72] = True
    cell, fell_back = cell_mask_cellpose(cp, nucleus)
    assert cell.shape == (224, 224)
    assert cell[nucleus].all()
