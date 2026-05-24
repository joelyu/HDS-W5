import numpy as np
import config


def test_handcrafted_dino_registered():
    assert "handcrafted_dino" in config.BACKBONES
    assert config.BACKBONE_DISPLAY["handcrafted_dino"] == "Handcrafted (DinoBloom seg)"


def test_tavakoli_51_has_51_names():
    assert len(config.TAVAKOLI_51) == 51
    assert len(set(config.TAVAKOLI_51)) == 51
    for shape in ("solidity", "convexity", "circularity"):
        assert shape in config.TAVAKOLI_51


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
