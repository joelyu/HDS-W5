#!/usr/bin/env python3
"""External validation on the Acevedo PBC dataset (2020).

Tests cross-institution, cross-instrument generalisability by applying classifiers
trained on KU-Optofil to Acevedo images. Crucially this covers BOTH feature
philosophies — frozen deep backbones AND handcrafted features — because the study
is a feature-extraction *comparison*: validating only the foundation model would
miss the point. For each feature set we run both downstream classifiers (XGBoost
and linear probe), so the cross-paradigm story extends to the external set too.

Pipeline (per backbone):
    1. Load Acevedo images, map to shared classes (6 of 8 Acevedo classes)
    2. Build the Acevedo feature matrix:
         deep backbones  -> frozen backbone inference (same transforms as 02)
         handcrafted     -> features.extract_cell_features (same as 02b), with the
                            CellPose variant reading masks from 02d --acevedo-dir
    3. Retrain the classifier on KU-Optofil train+val (shared classes only) using
       best HPs from 03 (XGBoost) / defaults (linear probe). Never sees Acevedo.
    4. Predict on Acevedo, multiseed-evaluate.

Data leakage checks:
    - DinoBloom explicitly held out Acevedo as external test set (MICCAI 2024)
    - ResNet-50 / EfficientNet-B0 / ViT-S/16 are ImageNet-pretrained — no overlap
    - Handcrafted features are deterministic — no training data at all
    - Classifiers trained ONLY on KU-Optofil; no HP tuning on Acevedo

Acevedo PBC dataset:
    - 17,092 images, 8 classes, CellaVision DM96, Hospital Clinic of Barcelona
    - Acevedo A et al. Data in Brief 2020. DOI: 10.1016/j.dib.2020.105474
    - Structure: PBC_dataset_normal_DIB/{class_name}/*.jpg

Shared classes (6): basophil, eosinophil, erythroblast, lymphocyte, monocyte,
    segmented_neutrophil (= Acevedo "neutrophil"). Dropped: ig, platelet.

Outputs saved to results/:
    external_validation.json          — {backbone: {xgboost: ..., linear: ...}}
    external_confusion_{backbone}.png — confusion matrix on Acevedo
    external_comparison.csv           — summary table (backbone x classifier)
    external_per_class.csv            — per-class P/R/F1 on Acevedo

Usage:
    python scripts/07_external_validation.py
    python scripts/07_external_validation.py --acevedo-dir data/acevedo/PBC_dataset_normal_DIB
    python scripts/07_external_validation.py --backbone handcrafted
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import xgboost as xgb
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    ACEVEDO_TO_KUOPTOFIL,
    BACKBONES,
    BACKBONE_DISPLAY,
    CLASS_LABELS,
    SEEDS,
    SHARED_CLASSES,
    load_features,
    style_axis,
)

# Which backbones this script can externally validate.
DEEP_BACKBONES = {"resnet50", "efficientnet_b0", "dinobloom_s", "vit_s16"}
HANDCRAFTED_BACKBONES = {"handcrafted", "handcrafted_cellpose"}

# Same transform as 02_feature_extraction.py — ImageNet normalisation
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Acevedo data loading ──────────────────────────────────────────────────


def _image_files(folder: Path) -> list[Path]:
    """jpg/png in a folder, excluding macOS dotfiles (.DS_Store, ._* AppleDouble)
    which `glob("*.jpg")` otherwise matches and PIL can't open."""
    return [p for p in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            if not p.name.startswith(".")]


def load_acevedo_images(acevedo_dir: Path) -> tuple[list[Path], list[str]]:
    """Scan Acevedo folders, return image paths and labels in KU-Optofil naming.

    Only includes classes present in ACEVEDO_TO_KUOPTOFIL (6 shared classes).
    """
    image_paths, labels, skipped = [], [], {}
    for folder in sorted(acevedo_dir.iterdir()):
        if not folder.is_dir():
            continue
        acevedo_class = folder.name.lower()
        if acevedo_class not in ACEVEDO_TO_KUOPTOFIL:
            n = len(_image_files(folder))
            if n > 0:
                skipped[acevedo_class] = n
            continue
        kuoptofil_class = ACEVEDO_TO_KUOPTOFIL[acevedo_class]
        for img_path in _image_files(folder):
            image_paths.append(img_path)
            labels.append(kuoptofil_class)
    if skipped:
        print(f"  Skipped classes (no shared mapping): {skipped}")
    return image_paths, labels


# ── Feature extraction: deep backbones ──────────────────────────────────────


def load_backbone(backbone_name: str) -> tuple[torch.nn.Module, int]:
    """Load a frozen deep backbone — mirrors 02_feature_extraction.py exactly."""
    import torchvision.models as models

    if backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        model.eval()
        return model, 2048

    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model.eval()
        return model, 1280

    if backbone_name == "dinobloom_s":
        from huggingface_hub import hf_hub_download
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        ckpt_path = hf_hub_download(repo_id="MarrLab/DinoBloom", filename="pytorch_model_s.bin")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        num_tokens = int(1 + (224 / 14) ** 2)
        model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_tokens, 384))
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        return model, 384

    if backbone_name == "vit_s16":
        import timm
        model = timm.create_model("vit_small_patch16_224", pretrained=True)
        model.head = torch.nn.Identity()
        model.eval()
        return model, 384

    print(f"ERROR: Unknown deep backbone '{backbone_name}'.", file=sys.stderr)
    sys.exit(1)


def extract_acevedo_features(
    model: torch.nn.Module, image_paths: list[Path], device: torch.device, batch_size: int = 32
) -> np.ndarray:
    """Extract features from Acevedo images using a frozen deep backbone."""
    model.to(device)
    all_features = []
    n_images = len(image_paths)
    with torch.no_grad():
        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            batch = torch.stack([
                TRANSFORM(Image.open(p).convert("RGB")) for p in image_paths[start:end]
            ]).to(device)
            all_features.append(model(batch).cpu().numpy())
            if (start // batch_size + 1) % 20 == 0 or end == n_images:
                print(f"\r    batch {start // batch_size + 1}/{(n_images + batch_size - 1) // batch_size}",
                      end="", flush=True)
    print()
    return np.concatenate(all_features)


# ── Feature extraction: handcrafted ─────────────────────────────────────────


def extract_acevedo_handcrafted(
    image_paths: list[Path], segmentation: str, mask_dir: Path, feature_names: list[str],
) -> tuple[np.ndarray, int, int]:
    """Extract the 65 handcrafted features on Acevedo images, ordered by
    feature_names (the KU-Optofil column order, so the trained model lines up).

    segmentation: 'cellpose' reads a precomputed mask per image from mask_dir
    (02d --acevedo-dir); 'convex_hull' needs no masks. Returns (X, n_fellback,
    n_missing_mask)."""
    from features import extract_cell_features  # heavy (cv2 + segmentation); lazy

    X, fell, missing = [], 0, 0
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        img = np.array(Image.open(p).convert("RGB"))
        cp_mask = None
        if segmentation == "cellpose":
            mp = mask_dir / f"{p.name}.png"
            if mp.exists():
                cp_mask = np.array(Image.open(mp).convert("L")) > 0
            else:
                missing += 1
        feats, fell_back = extract_cell_features(img, cellpose_mask=cp_mask)
        fell += int(fell_back)
        X.append(np.array([feats[k] for k in feature_names], dtype=np.float64))
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"\r    {i + 1}/{n}", end="", flush=True)
    print()
    X = np.nan_to_num(np.stack(X), nan=0.0, posinf=0.0, neginf=0.0)
    return X, fell, missing


# ── Retraining on KU-Optofil shared classes (never sees Acevedo) ────────────


def _shared_trainval(kuoptofil_features: dict, shared_le: LabelEncoder):
    """KU-Optofil train+val features/labels restricted to the 6 shared classes.

    Uses the stored raw string labels (train_y_str/val_y_str) directly, then
    re-encodes with the 6-class shared_le — the same encoder used for the
    Acevedo labels, so train and test integer labels share one space."""
    all_X = np.concatenate([kuoptofil_features["train_X"], kuoptofil_features["val_X"]])
    all_y_str = np.concatenate([kuoptofil_features["train_y_str"], kuoptofil_features["val_y_str"]])
    mask = np.array([y in set(SHARED_CLASSES) for y in all_y_str])
    return all_X[mask], shared_le.transform(all_y_str[mask])


def retrain_xgb_shared(best_params: dict, kuoptofil_features: dict, shared_le: LabelEncoder, seed: int = 42):
    """XGBoost on KU-Optofil shared classes with best HPs from 03."""
    X, y = _shared_trainval(kuoptofil_features, shared_le)
    params = {**best_params, "random_state": seed, "n_jobs": -1, "verbosity": 0}
    params.pop("early_stopping_rounds", None)
    params["num_class"] = len(SHARED_CLASSES)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, sample_weight=compute_sample_weight("balanced", y), verbose=False)
    return model


class _ScaledPredictor:
    """StandardScaler + LogisticRegression, exposing .predict like the XGB model."""

    def __init__(self, model: LogisticRegression, scaler: StandardScaler):
        self.model, self.scaler = model, scaler

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


def retrain_linear_shared(kuoptofil_features: dict, shared_le: LabelEncoder, seed: int = 42):
    """Linear probe (StandardScaler + balanced LogReg) on KU-Optofil shared classes.

    Mirrors 03b. Deterministic (lbfgs), so a single fit suffices."""
    X, y = _shared_trainval(kuoptofil_features, shared_le)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed, n_jobs=-1,
    )
    model.fit(Xs, y)
    return _ScaledPredictor(model, scaler)


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate_on_acevedo(model, X_acevedo, y_acevedo, shared_le: LabelEncoder) -> dict:
    """Evaluate any .predict model on Acevedo features. Returns metrics dict."""
    preds = model.predict(X_acevedo)
    return {
        "macro_f1": float(f1_score(y_acevedo, preds, average="macro")),
        "weighted_f1": float(f1_score(y_acevedo, preds, average="weighted")),
        "accuracy": float(accuracy_score(y_acevedo, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_acevedo, preds)),
        "classification_report": classification_report(
            y_acevedo, preds, target_names=list(shared_le.classes_), output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_acevedo, preds).tolist(),
    }


def evaluate_multiseed(make_model_fn, X_acevedo, y_acevedo_int, shared_le, seeds: list[int]) -> dict:
    """Fit `make_model_fn(seed)` and evaluate on Acevedo across seeds; aggregate."""
    all_metrics = []
    for seed in seeds:
        metrics = evaluate_on_acevedo(make_model_fn(seed), X_acevedo, y_acevedo_int, shared_le)
        metrics["seed"] = seed
        all_metrics.append(metrics)

    def _mean_std(key):
        vals = [m[key] for m in all_metrics]
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return float(np.mean(vals)), std

    macro_f1s = [m["macro_f1"] for m in all_metrics]
    median_idx = int(np.argsort(macro_f1s)[len(macro_f1s) // 2])
    out = {"per_seed": all_metrics,
           "classification_report": all_metrics[median_idx]["classification_report"],
           "confusion_matrix": all_metrics[median_idx]["confusion_matrix"]}
    for key in ("macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"):
        out[f"{key}_mean"], out[f"{key}_std"] = _mean_std(key)
    return out


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_confusion_matrix(cm, class_names, backbone, results_dir: Path) -> None:
    """Normalised confusion matrix for Acevedo validation (primary classifier)."""
    cm_arr = np.array(cm, dtype=float)
    cm_norm = np.nan_to_num(cm_arr / cm_arr.sum(axis=1, keepdims=True))
    short_names = [CLASS_LABELS.get(n, n) for n in class_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    for i in range(len(short_names)):
        for j in range(len(short_names)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}\n({int(cm_arr[i, j])})", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"External Validation — {BACKBONE_DISPLAY.get(backbone, backbone)} on Acevedo PBC")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Recall")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(results_dir / f"external_confusion_{backbone}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved external_confusion_{backbone}.png")


# ── Main ────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="External validation on Acevedo PBC.")
    parser.add_argument(
        "--acevedo-dir", type=Path,
        default=base / "data" / "acevedo" / "PBC_dataset_normal_DIB",
        help="Path to Acevedo dataset root (folder containing class subfolders).",
    )
    parser.add_argument("--results-dir", type=Path, default=base / "results")
    parser.add_argument("--backbone", choices=BACKBONES, default=None,
                        help="Run a single backbone (default: all supported).")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def acevedo_X_for_backbone(backbone, image_paths, results_dir, device, batch_size):
    """Build the Acevedo feature matrix + the KU-Optofil features for a backbone.

    Returns (X_acevedo, kuoptofil_features) or (None, None) if the backbone can't
    be externally validated (e.g. missing CellPose masks, unsupported backbone)."""
    if not (results_dir / f"{backbone}_features.npz").exists():
        print(f"  Skipping {backbone} — {backbone}_features.npz missing (run 02/02b first).")
        return None, None

    if backbone in HANDCRAFTED_BACKBONES:
        kuoptofil_features = load_features(results_dir, backbone)
        fnames = kuoptofil_features.get("feature_names")
        if fnames is None:
            print(f"  Skipping {backbone} — no feature_names in its .npz.")
            return None, None
        seg = "cellpose" if backbone == "handcrafted_cellpose" else "convex_hull"
        mask_dir = results_dir / "cellpose_masks_acevedo"
        if seg == "cellpose" and not mask_dir.exists():
            print(f"  Skipping {backbone} — {mask_dir} missing "
                  f"(run: 02d_cellpose_masks.py --acevedo-dir ... first).")
            return None, None
        print(f"  Extracting handcrafted features on Acevedo ({seg})...")
        X_acevedo, fell, missing = extract_acevedo_handcrafted(
            image_paths, seg, mask_dir, list(fnames)
        )
        if missing:
            print(f"  WARNING: {missing}/{len(image_paths)} Acevedo images had no CellPose "
                  f"mask (used convex hull).")
        if fell:
            print(f"  {fell}/{len(image_paths)} fell back to convex hull.")
        return X_acevedo, kuoptofil_features

    if backbone in DEEP_BACKBONES:
        print("  Extracting features...")
        model_bb, _ = load_backbone(backbone)
        X_acevedo = extract_acevedo_features(model_bb, image_paths, device, batch_size)
        del model_bb
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        return X_acevedo, load_features(results_dir, backbone)

    print(f"  Skipping {backbone} — not supported in external validation.")
    return None, None


def main() -> int:
    args = parse_args()
    acevedo_dir: Path = args.acevedo_dir.resolve()
    results_dir: Path = args.results_dir.resolve()

    if not acevedo_dir.exists():
        print(f"ERROR: Acevedo dataset not found at {acevedo_dir}", file=sys.stderr)
        print("Run: python scripts/00b_download_acevedo.py   (or download manually from "
              "https://data.mendeley.com/datasets/snkd93bnjr/1)", file=sys.stderr)
        return 1

    print("=== Loading Acevedo PBC dataset ===")
    image_paths, labels_str = load_acevedo_images(acevedo_dir)
    print(f"  Loaded {len(image_paths)} images across {len(set(labels_str))} shared classes")
    if not image_paths:
        print("ERROR: No images found in shared classes.", file=sys.stderr)
        return 1
    for cls, count in pd.Series(labels_str).value_counts().items():
        print(f"    {CLASS_LABELS.get(cls, cls):15s}: {count:5d}")

    shared_le = LabelEncoder()
    shared_le.fit(SHARED_CLASSES)
    y_acevedo_int = shared_le.transform(labels_str)
    print(f"  Shared classes: {list(shared_le.classes_)}\n")

    xgb_results_path = results_dir / "xgboost_results.json"
    xgb_results = {}
    if xgb_results_path.exists():
        with open(xgb_results_path) as f:
            xgb_results = json.load(f)
    else:
        print("  WARNING: no xgboost_results.json — XGBoost arm will be skipped (linear only).")

    device = get_device()
    print(f"Device: {device}\n")

    backbones_to_run = [args.backbone] if args.backbone else BACKBONES
    all_results = {}

    for backbone in backbones_to_run:
        print(f"{'='*60}\n  {BACKBONE_DISPLAY.get(backbone, backbone)}\n{'='*60}")
        t0 = time.time()
        X_acevedo, kuoptofil_features = acevedo_X_for_backbone(
            backbone, image_paths, results_dir, device, args.batch_size
        )
        if X_acevedo is None:
            continue
        print(f"  Acevedo features: {X_acevedo.shape} ({time.time()-t0:.0f}s)")

        backbone_results = {}

        # XGBoost (needs best params from 03)
        if backbone in xgb_results:
            best_params = xgb_results[backbone]["best_params"]
            backbone_results["xgboost"] = evaluate_multiseed(
                lambda s, bp=best_params, kf=kuoptofil_features: retrain_xgb_shared(bp, kf, shared_le, s),
                X_acevedo, y_acevedo_int, shared_le, SEEDS,
            )
        else:
            print(f"  (no XGBoost best_params for {backbone} — skipping its XGBoost arm)")

        # Linear probe (deterministic — a single fit)
        backbone_results["linear"] = evaluate_multiseed(
            lambda s, kf=kuoptofil_features: retrain_linear_shared(kf, shared_le, s),
            X_acevedo, y_acevedo_int, shared_le, [42],
        )

        for clf, r in backbone_results.items():
            print(f"  [{clf:7s}] macro F1 {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}  "
                  f"acc {r['accuracy_mean']:.4f}")

        primary = backbone_results.get("xgboost", backbone_results["linear"])
        plot_confusion_matrix(primary["confusion_matrix"], list(shared_le.classes_), backbone, results_dir)
        all_results[backbone] = backbone_results
        print()

    # ── Save + tables ──────────────────────────────────────────────────
    out_path = results_dir / "external_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_path}")

    rows, per_class_rows = [], []
    for backbone in BACKBONES:
        if backbone not in all_results:
            continue
        for clf, r in all_results[backbone].items():
            rows.append({
                "Backbone": BACKBONE_DISPLAY.get(backbone, backbone),
                "Classifier": clf,
                "Macro F1": f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}",
                "Weighted F1": f"{r['weighted_f1_mean']:.4f} ± {r['weighted_f1_std']:.4f}",
                "Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                "Balanced Acc.": f"{r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f}",
            })
            report = r["classification_report"]
            for cls in SHARED_CLASSES:
                if cls in report:
                    per_class_rows.append({
                        "Backbone": BACKBONE_DISPLAY.get(backbone, backbone),
                        "Classifier": clf,
                        "Class": CLASS_LABELS.get(cls, cls),
                        "Precision": report[cls]["precision"],
                        "Recall": report[cls]["recall"],
                        "F1": report[cls]["f1-score"],
                        "Support": report[cls]["support"],
                    })

    if rows:
        comp_df = pd.DataFrame(rows)
        comp_df.to_csv(results_dir / "external_comparison.csv", index=False)
        print(f"Saved external_comparison.csv\n")
        print(comp_df.to_string(index=False))
    if per_class_rows:
        pd.DataFrame(per_class_rows).to_csv(results_dir / "external_per_class.csv", index=False)
        print(f"\nSaved external_per_class.csv")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
