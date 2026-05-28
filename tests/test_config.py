import numpy as np
import config


def test_handcrafted_cellpose_registered():
    assert "handcrafted_cellpose" in config.BACKBONES
    assert config.BACKBONE_DISPLAY["handcrafted_cellpose"] == "Handcrafted (CellPose seg)"


def test_handcrafted_dino_not_registered():
    # DinoBloom-as-segmenter was abandoned (documented negative result); the
    # backbone entry is dropped so bare 03/03b loops don't sys.exit on a
    # missing handcrafted_dino_features.npz.
    assert "handcrafted_dino" not in config.BACKBONES
    assert "handcrafted_dino" not in config.BACKBONE_DISPLAY


def test_tavakoli_51_has_51_names():
    assert len(config.TAVAKOLI_51) == 51
    assert len(set(config.TAVAKOLI_51)) == 51
    for shape in ("solidity", "convexity", "circularity"):
        assert shape in config.TAVAKOLI_51


def test_parse_key_plain_backbone():
    assert config.parse_key("dinobloom_s_multilevel") == {
        "base": "dinobloom_s_multilevel", "feature_set": "all", "class_mode": "13class",
    }


def test_parse_key_tavakoli_on_cellpose():
    out = config.parse_key("handcrafted_cellpose_tavakoli")
    assert out == {"base": "handcrafted_cellpose", "feature_set": "tavakoli", "class_mode": "13class"}


def test_parse_key_tavakoli_and_5class():
    assert config.parse_key("handcrafted_tavakoli_5class") == {
        "base": "handcrafted", "feature_set": "tavakoli", "class_mode": "5class",
    }


def test_parse_key_5class_only():
    assert config.parse_key("dinobloom_s_5class") == {
        "base": "dinobloom_s", "feature_set": "all", "class_mode": "5class",
    }


def test_segmentation_of():
    assert config.segmentation_of("handcrafted") == "convex-hull"
    assert config.segmentation_of("handcrafted_cellpose") == "CellPose"
    assert config.segmentation_of("dinobloom_s") == "—"


def test_label_encoder_is_alphabetical_and_stable():
    # The integer<->class mapping the WHOLE pipeline rides on: 02 stores string
    # labels, 03/03b/07 decode through this encoder, and 05 pairs predictions in
    # this integer space. If this ordering ever drifts, every saved result and
    # every confusion-matrix axis label silently desynchronises.
    le = config.get_label_encoder()
    assert list(le.classes_) == config.CLASS_ORDER_ALPHA
    assert len(le.classes_) == config.NUM_CLASSES
    # Concrete anchors (alphabetical 0..12), not just self-consistency.
    assert le.transform(["band_neutrophil", "basophil", "blast"]).tolist() == [0, 1, 2]
    assert le.transform(["segmented_neutrophil"])[0] == 12


def test_load_features_encodes_labels_alphabetically(tmp_path):
    np.savez(
        tmp_path / "resnet50_features.npz",
        train_X=np.zeros((3, 4)), train_y=np.array(["blast", "basophil", "band_neutrophil"]),
        validation_X=np.zeros((1, 4)), validation_y=np.array(["monocyte"]),
        test_X=np.zeros((1, 4)), test_y=np.array(["segmented_neutrophil"]),
    )
    data = config.load_features(tmp_path, "resnet50")
    # Raw strings preserved...
    assert list(data["train_y_str"]) == ["blast", "basophil", "band_neutrophil"]
    # ...and encoded into the alphabetical integer space (blast=2, basophil=1,
    # band_neutrophil=0; segmented_neutrophil=12).
    assert data["train_y"].tolist() == [2, 1, 0]
    assert data["test_y"].tolist() == [12]


def test_reduce_to_5class_merges_drops_and_encodes():
    # band+segmented -> neutrophil; lymphocyte+reactive -> lymphocyte; the six
    # non-mappable classes are dropped. Encoded to the alphabetical 5-class space.
    labels = np.array([
        "band_neutrophil", "segmented_neutrophil", "reactive_lymphocyte",
        "lymphocyte", "monocyte", "eosinophil", "basophil",
        "blast", "erythroblast", "giant_platelet",   # all dropped
    ])
    X = np.arange(len(labels)).reshape(-1, 1).astype(float)  # row tag = index
    data = {f"{s}_X": X.copy() for s in ("train", "val", "test")}
    data.update({f"{s}_y_str": labels.copy() for s in ("train", "val", "test")})

    out = config.reduce_to_5class(data)

    assert list(out["train_y_str"]) == [
        "neutrophil", "neutrophil", "lymphocyte", "lymphocyte",
        "monocyte", "eosinophil", "basophil",
    ]
    # Dropped rows (indices 7,8,9) removed from X too, surviving order preserved.
    assert out["train_X"].ravel().tolist() == [0, 1, 2, 3, 4, 5, 6]
    le = out["label_encoder"]
    assert list(le.classes_) == ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
    # basophil=0 eosinophil=1 lymphocyte=2 monocyte=3 neutrophil=4
    assert out["train_y"].tolist() == [4, 4, 2, 2, 3, 1, 0]


def test_load_features_passes_feature_names(tmp_path):
    fn = np.array(["solidity", "nc_ratio", "nuc_glcm_contrast"])
    np.savez(
        tmp_path / "handcrafted_features.npz",
        train_X=np.zeros((2, 3)), train_y=np.array(["blast", "monocyte"]),
        validation_X=np.zeros((1, 3)), validation_y=np.array(["blast"]),
        test_X=np.zeros((1, 3)), test_y=np.array(["monocyte"]),
        feature_names=fn,
    )
    data = config.load_features(tmp_path, "handcrafted")
    assert data["feature_names"] is not None
    assert list(data["feature_names"]) == ["solidity", "nc_ratio", "nuc_glcm_contrast"]
