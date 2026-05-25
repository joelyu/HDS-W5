"""Shared constants and helpers for all pipeline scripts.

Single source of truth for folder mappings, class labels, split ordering,
colour palette, and common data-loading utilities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Mapping from metadata class names to zip folder names
FOLDER_NAME_MAP = {
    "band_neutrophil": "Band Neutrophil",
    "basophil": "Basophil",
    "blast": "Blast",
    "eosinophil": "Eosinophil",
    "erythroblast": "Erythroblast",
    "giant_platelet": "Giant Platelet",
    "lymphocyte": "Lymphocyte",
    "metamyelocyte": "Metamyelocyte",
    "monocyte": "Monocyte",
    "myelocyte": "Myelocyte",
    "platelet_cluster": "Platelet Cluster",
    "reactive_lymphocyte": "Reactive Lymphocyte",
    "segmented_neutrophil": "Segmented Neutrophil",
}

# Display-friendly labels (shorter, for plot axes and confusion matrices)
CLASS_LABELS = {
    "segmented_neutrophil": "Seg. Neut.",
    "band_neutrophil": "Band Neut.",
    "eosinophil": "Eosinophil",
    "basophil": "Basophil",
    "lymphocyte": "Lymphocyte",
    "reactive_lymphocyte": "React. Lymph.",
    "monocyte": "Monocyte",
    "metamyelocyte": "Metamyelocyte",
    "myelocyte": "Myelocyte",
    "blast": "Blast",
    "erythroblast": "Erythroblast",
    "giant_platelet": "Giant Plt.",
    "platelet_cluster": "Plt. Cluster",
}

CLASS_ORDER = list(CLASS_LABELS.keys())

SPLIT_ORDER = ["train", "validation", "test"]

NUM_CLASSES = 13

BACKBONES = ["resnet50", "efficientnet_b0", "dinobloom_s", "vit_s16", "dinobloom_s_multilevel", "handcrafted", "handcrafted_cellpose"]

SEEDS = [42, 123, 456, 789, 1024]

# The 51 Tavakoli (2021) feature names, generated to match the keys produced
# by 02b's extract_cell_features. Used to subset the extended feature matrix
# back to the Tavakoli baseline (positional slicing fails — names are sorted).
_TAVAKOLI_CHANNELS = ["R", "G", "B", "H", "S", "V", "L", "A", "Blab", "Y", "Cr", "Cb"]
TAVAKOLI_51 = ["solidity", "convexity", "circularity"] + [
    f"{ratio}_{ch}"
    for ch in _TAVAKOLI_CHANNELS
    for ratio in ("ncl_cvx_mean", "ncl_cvx_std", "roc_cvx_mean", "roc_cvx_std")
]

# Okabe-Ito-adjacent colorblind-safe palette — red/blue theme for blood cells.
COLOURS = {
    "primary": "#B2182B",
    "secondary": "#2166AC",
    "tertiary": "#4D4D4D",
    "train": "#2166AC",
    "validation": "#B2182B",
    "test": "#4D4D4D",
    "highlight": "#D6604D",
    "muted": "#92C5DE",
    "tpe": "#2166AC",
    "gp": "#B2182B",
}

SPLIT_COLOURS = [COLOURS["train"], COLOURS["validation"], COLOURS["test"]]

BACKBONE_DISPLAY = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "dinobloom_s": "DinoBloom-S",
    "vit_s16": "ViT-S/16",
    "dinobloom_s_multilevel": "DinoBloom-S (multi-level)",
    "handcrafted": "Handcrafted",
    "handcrafted_cellpose": "Handcrafted (CellPose seg)",
}

# Alphabetical class order — matches LabelEncoder / sklearn output ordering
CLASS_ORDER_ALPHA = sorted(CLASS_LABELS.keys())

# Colourblind-safe palette for 13 classes (Okabe-Ito extended)
CLASS_COLOURS = [
    "#E69F00",  # segmented_neutrophil — orange
    "#56B4E9",  # band_neutrophil — sky blue
    "#009E73",  # eosinophil — green
    "#F0E442",  # basophil — yellow
    "#0072B2",  # lymphocyte — blue
    "#D55E00",  # reactive_lymphocyte — vermillion
    "#CC79A7",  # monocyte — pink
    "#999999",  # metamyelocyte — grey
    "#882255",  # myelocyte — wine
    "#B2182B",  # blast — red (clinical emphasis)
    "#332288",  # erythroblast — indigo
    "#44AA99",  # giant_platelet — teal
    "#DDCC77",  # platelet_cluster — sand
]
CLASS_COLOUR_MAP = dict(zip(CLASS_ORDER, CLASS_COLOURS))


# ── 5-class WBC differential ──────────────────────────────────────────────

# Merge 13-class KU-Optofil → standard 5-class WBC differential.
# Classes not in this mapping are dropped (no clean mapping to standard 5).
FIVE_CLASS_MAP = {
    "segmented_neutrophil": "neutrophil",
    "band_neutrophil": "neutrophil",
    "eosinophil": "eosinophil",
    "basophil": "basophil",
    "lymphocyte": "lymphocyte",
    "reactive_lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
}

FIVE_CLASS_ORDER = sorted(set(FIVE_CLASS_MAP.values()))

FIVE_CLASS_LABELS = {
    "basophil": "Basophil",
    "eosinophil": "Eosinophil",
    "lymphocyte": "Lymphocyte",
    "monocyte": "Monocyte",
    "neutrophil": "Neutrophil",
}


def reduce_to_5class(data: dict) -> dict:
    """Reduce a 13-class loaded feature dict to 5-class WBC differential.

    Merges band+segmented→neutrophil, lymphocyte+reactive→lymphocyte.
    Drops blast, erythroblast, metamyelocyte, myelocyte, giant_platelet,
    platelet_cluster (no clean mapping to standard 5).
    """
    le5 = LabelEncoder()
    le5.fit(FIVE_CLASS_ORDER)

    out = {}
    for split in ("train", "val", "test"):
        X = data[f"{split}_X"]
        y_str = data[f"{split}_y_str"]

        # Keep only samples whose 13-class label maps to a 5-class label
        keep = np.array([lbl in FIVE_CLASS_MAP for lbl in y_str])
        X = X[keep]
        y_str_kept = y_str[keep]

        # Remap to 5-class names
        y_mapped = np.array([FIVE_CLASS_MAP[lbl] for lbl in y_str_kept])

        out[f"{split}_X"] = X
        out[f"{split}_y"] = le5.transform(y_mapped)
        out[f"{split}_y_str"] = y_mapped

    out["label_encoder"] = le5
    return out


# ── Acevedo external validation ────────────────────────────────────────────

# Acevedo PBC (2020) folder names → KU-Optofil class names.
# Only classes with a clear 1:1 mapping are included.
# Dropped: "ig" (immature granulocyte — no single KU-Optofil equivalent),
#          "platelet" (KU-Optofil has giant_platelet + platelet_cluster, not single platelets)
ACEVEDO_TO_KUOPTOFIL = {
    "neutrophil": "segmented_neutrophil",
    "eosinophil": "eosinophil",
    "basophil": "basophil",
    "lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
    "erythroblast": "erythroblast",
}

# The 6 shared classes (in KU-Optofil naming), alphabetically sorted for LabelEncoder
SHARED_CLASSES = sorted(ACEVEDO_TO_KUOPTOFIL.values())


# ── Shared helpers ─────────────────────────────────────────────────────────


def get_label_encoder() -> LabelEncoder:
    """Return a LabelEncoder fitted to alphabetically sorted class names."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER_ALPHA)
    return le


def load_features(results_dir: Path, backbone: str) -> dict:
    """Load train/val/test features + labels from .npz file."""
    path = results_dir / f"{backbone}_features.npz"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 02_feature_extraction.py first.", file=sys.stderr)
        sys.exit(1)

    data = np.load(path)
    le = get_label_encoder()

    return {
        "train_X": data["train_X"],
        "train_y": le.transform(data["train_y"]),
        "train_y_str": data["train_y"],
        "val_X": data["validation_X"],
        "val_y": le.transform(data["validation_y"]),
        "val_y_str": data["validation_y"],
        "test_X": data["test_X"],
        "test_y": le.transform(data["test_y"]),
        "test_y_str": data["test_y"],
        "feature_names": data["feature_names"] if "feature_names" in data.files else None,
        "label_encoder": le,
    }


def style_axis(ax: plt.Axes) -> None:
    """Apply consistent axis styling — remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
