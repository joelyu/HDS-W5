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

BACKBONES = ["resnet50", "efficientnet_b0", "dinobloom_s"]

SEEDS = [42, 123, 456, 789, 1024]

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
        "test_X": data["test_X"],
        "test_y": le.transform(data["test_y"]),
        "test_y_str": data["test_y"],
        "label_encoder": le,
    }


def style_axis(ax: plt.Axes) -> None:
    """Apply consistent axis styling — remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
