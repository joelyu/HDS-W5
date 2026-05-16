#!/usr/bin/env python3
"""External validation on the Acevedo PBC dataset (2020).

Tests cross-institution, cross-instrument generalisability by applying
XGBoost classifiers (trained on KU-Optofil) to Acevedo images.

Pipeline:
    1. Load Acevedo images, map to shared classes (6 of 8 Acevedo classes)
    2. Extract features using same frozen backbones + transforms as 02
    3. Retrain XGBoost on KU-Optofil train+val (shared classes only) using
       best hyperparameters from 03
    4. Predict on Acevedo, compute metrics

Data leakage checks:
    - DinoBloom explicitly held out Acevedo as external test set (MICCAI 2024)
    - ResNet-50 / EfficientNet-B0 are ImageNet-pretrained — no overlap
    - XGBoost trained ONLY on KU-Optofil (never sees Acevedo during training)
    - No hyperparameter tuning on Acevedo — best params from 03 used directly

Acevedo PBC dataset:
    - 17,092 images, 8 classes, CellaVision DM96, Hospital Clinic of Barcelona
    - Acevedo A et al. Data in Brief 2020. DOI: 10.1016/j.dib.2020.105474
    - Structure: PBC_dataset_normal_DIB/{class_name}/*.jpg

Shared classes (6): basophil, eosinophil, erythroblast, lymphocyte, monocyte,
    segmented_neutrophil (= Acevedo "neutrophil")
Dropped: ig (no 1:1 match), platelet (KU-Optofil has giant_platelet/platelet_cluster)

Outputs saved to results/:
    external_validation.json          — all metrics per backbone
    external_confusion_{backbone}.png — confusion matrix on Acevedo
    external_comparison.csv           — summary table across backbones
    external_per_class.csv            — per-class P/R/F1 on Acevedo

Usage:
    python scripts/07_external_validation.py
    python scripts/07_external_validation.py --acevedo-dir data/acevedo
    python scripts/07_external_validation.py --backbone dinobloom_s
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    ACEVEDO_TO_KUOPTOFIL,
    BACKBONES,
    BACKBONE_DISPLAY,
    CLASS_LABELS,
    CLASS_ORDER_ALPHA,
    COLOURS,
    SEEDS,
    SHARED_CLASSES,
    get_label_encoder,
    load_features,
    style_axis,
)


# Same transform as 02_feature_extraction.py — ImageNet normalisation
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Acevedo data loading ──────────────────────────────────────────────────


def load_acevedo_images(
    acevedo_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Scan Acevedo folder structure, return image paths and mapped labels.

    Only includes classes present in ACEVEDO_TO_KUOPTOFIL (6 shared classes).
    Returns labels in KU-Optofil naming convention.
    """
    image_paths = []
    labels = []
    skipped = {}

    for folder in sorted(acevedo_dir.iterdir()):
        if not folder.is_dir():
            continue
        acevedo_class = folder.name.lower()

        if acevedo_class not in ACEVEDO_TO_KUOPTOFIL:
            # Track how many images we're dropping
            n = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
            if n > 0:
                skipped[acevedo_class] = n
            continue

        kuoptofil_class = ACEVEDO_TO_KUOPTOFIL[acevedo_class]
        imgs = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        for img_path in imgs:
            image_paths.append(img_path)
            labels.append(kuoptofil_class)

    if skipped:
        print(f"  Skipped classes (no shared mapping): {skipped}")

    return image_paths, labels


# ── Feature extraction ─────────────────────────────────────────────────────


def load_backbone(backbone_name: str) -> tuple[torch.nn.Module, int]:
    """Load a frozen backbone — mirrors 02_feature_extraction.py exactly."""
    import torchvision.models as models

    if backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        model.eval()
        return model, 2048

    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model.eval()
        return model, 1280

    elif backbone_name == "dinobloom_s":
        from huggingface_hub import hf_hub_download
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        ckpt_path = hf_hub_download(repo_id="MarrLab/DinoBloom", filename="pytorch_model_s.bin")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        num_tokens = int(1 + (224 / 14) ** 2)
        model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_tokens, 384))
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        return model, 384

    else:
        print(f"ERROR: Unknown backbone '{backbone_name}'.", file=sys.stderr)
        sys.exit(1)


def extract_acevedo_features(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract features from Acevedo images using a frozen backbone."""
    model.to(device)
    all_features = []
    n_images = len(image_paths)

    with torch.no_grad():
        for start in range(0, n_images, batch_size):
            end = min(start + batch_size, n_images)
            batch_tensors = []
            for img_path in image_paths[start:end]:
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(TRANSFORM(img))

            batch = torch.stack(batch_tensors).to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())

            if (start // batch_size + 1) % 20 == 0 or end == n_images:
                print(f"\r    batch {start // batch_size + 1}/{(n_images + batch_size - 1) // batch_size}",
                      end="", flush=True)

    print()
    return np.concatenate(all_features)


# ── XGBoost retraining (shared classes only) ───────────────────────────────


def retrain_on_shared_classes(
    best_params: dict,
    kuoptofil_features: dict,
    shared_le: LabelEncoder,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Retrain XGBoost on KU-Optofil train+val, restricted to shared classes.

    This ensures the model only predicts shared classes, making metrics
    directly comparable. The model has never seen Acevedo data.
    """
    # Recover string labels from integer-encoded labels via the 13-class encoder
    full_le = kuoptofil_features["label_encoder"]

    train_X = kuoptofil_features["train_X"]
    train_y_str = full_le.inverse_transform(kuoptofil_features["train_y"])

    val_X = kuoptofil_features["val_X"]
    val_y_str = full_le.inverse_transform(kuoptofil_features["val_y"])

    # Combine train + val
    all_X = np.concatenate([train_X, val_X])
    all_y_str = np.concatenate([train_y_str, val_y_str])

    # Filter to shared classes only
    shared_set = set(SHARED_CLASSES)
    mask = np.array([y in shared_set for y in all_y_str])
    X_filtered = all_X[mask]
    y_filtered = shared_le.transform(all_y_str[mask])

    print(f"    KU-Optofil train+val: {len(all_X)} → {mask.sum()} (shared classes only)")

    # Train with best params, adjusted for fewer classes
    params = {**best_params, "random_state": seed, "n_jobs": -1, "verbosity": 0}
    params.pop("early_stopping_rounds", None)
    params["num_class"] = len(SHARED_CLASSES)

    model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight("balanced", y_filtered)
    model.fit(X_filtered, y_filtered, sample_weight=sample_weights, verbose=False)
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate_on_acevedo(
    model: xgb.XGBClassifier,
    X_acevedo: np.ndarray,
    y_acevedo: np.ndarray,
    shared_le: LabelEncoder,
) -> dict:
    """Evaluate trained model on Acevedo features. Returns metrics dict."""
    preds = model.predict(X_acevedo)

    class_names = list(shared_le.classes_)
    report = classification_report(
        y_acevedo, preds, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_acevedo, preds).tolist()

    return {
        "macro_f1": float(f1_score(y_acevedo, preds, average="macro")),
        "weighted_f1": float(f1_score(y_acevedo, preds, average="weighted")),
        "accuracy": float(accuracy_score(y_acevedo, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_acevedo, preds)),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def evaluate_multiseed(
    best_params: dict,
    kuoptofil_features: dict,
    X_acevedo: np.ndarray,
    y_acevedo_int: np.ndarray,
    shared_le: LabelEncoder,
    seeds: list[int],
) -> dict:
    """Multi-seed evaluation on Acevedo for stability."""
    all_metrics = []

    for seed in seeds:
        model = retrain_on_shared_classes(best_params, kuoptofil_features, shared_le, seed)
        metrics = evaluate_on_acevedo(model, X_acevedo, y_acevedo_int, shared_le)
        metrics["seed"] = seed
        all_metrics.append(metrics)

    # Aggregate
    macro_f1s = [m["macro_f1"] for m in all_metrics]
    weighted_f1s = [m["weighted_f1"] for m in all_metrics]
    accuracies = [m["accuracy"] for m in all_metrics]
    balanced_accs = [m["balanced_accuracy"] for m in all_metrics]

    # Use median seed for detailed report + confusion matrix
    median_idx = int(np.argsort(macro_f1s)[len(macro_f1s) // 2])

    return {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "balanced_accuracy_mean": float(np.mean(balanced_accs)),
        "balanced_accuracy_std": float(np.std(balanced_accs)),
        "per_seed": all_metrics,
        "classification_report": all_metrics[median_idx]["classification_report"],
        "confusion_matrix": all_metrics[median_idx]["confusion_matrix"],
    }


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    backbone: str,
    results_dir: Path,
) -> None:
    """Plot normalised confusion matrix for Acevedo validation."""
    cm_arr = np.array(cm, dtype=float)
    cm_norm = cm_arr / cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

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
            count = int(cm_arr[i, j])
            colour = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({count})", ha="center", va="center",
                    fontsize=7, color=colour)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"External Validation — {BACKBONE_DISPLAY.get(backbone, backbone)} "
                 f"→ XGBoost on Acevedo PBC")
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
    parser = argparse.ArgumentParser(description="External validation on Acevedo PBC.")
    parser.add_argument(
        "--acevedo-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "acevedo" / "PBC_dataset_normal_DIB",
        help="Path to Acevedo dataset root (folder containing class subfolders).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=None,
        help="Run for a single backbone (default: all three).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    acevedo_dir: Path = args.acevedo_dir.resolve()
    results_dir: Path = args.results_dir.resolve()

    # ── Verify Acevedo data exists ─────────────────────────────────────
    if not acevedo_dir.exists():
        print(f"ERROR: Acevedo dataset not found at {acevedo_dir}", file=sys.stderr)
        print("Download from: https://data.mendeley.com/datasets/snkd93bnjr/1", file=sys.stderr)
        print(f"Expected structure: {acevedo_dir}/{{neutrophil,eosinophil,...}}/*.jpg", file=sys.stderr)
        return 1

    # ── Load Acevedo images ────────────────────────────────────────────
    print("=== Loading Acevedo PBC dataset ===")
    image_paths, labels_str = load_acevedo_images(acevedo_dir)
    print(f"  Loaded {len(image_paths)} images across {len(set(labels_str))} shared classes")

    if len(image_paths) == 0:
        print("ERROR: No images found in shared classes.", file=sys.stderr)
        return 1

    # Class distribution
    label_counts = pd.Series(labels_str).value_counts()
    for cls, count in label_counts.items():
        print(f"    {CLASS_LABELS.get(cls, cls):15s}: {count:5d}")
    print()

    # ── Encode Acevedo labels ──────────────────────────────────────────
    # Use a LabelEncoder fitted to SHARED_CLASSES only (alphabetical)
    shared_le = LabelEncoder()
    shared_le.fit(SHARED_CLASSES)
    y_acevedo_int = shared_le.transform(labels_str)

    print(f"  Shared LabelEncoder classes: {list(shared_le.classes_)}")
    print()

    # ── Load XGBoost results for best params ───────────────────────────
    xgb_results_path = results_dir / "xgboost_results.json"
    if not xgb_results_path.exists():
        print(f"ERROR: {xgb_results_path} not found. Run 03_xgboost_training.py first.",
              file=sys.stderr)
        return 1
    with open(xgb_results_path) as f:
        xgb_results = json.load(f)

    # ── Process each backbone ──────────────────────────────────────────
    device = get_device()
    print(f"Device: {device}")
    print()

    backbones_to_run = [args.backbone] if args.backbone else BACKBONES
    all_results = {}

    for backbone in backbones_to_run:
        print(f"{'='*60}")
        print(f"  {BACKBONE_DISPLAY.get(backbone, backbone)}")
        print(f"{'='*60}")

        if backbone not in xgb_results:
            print(f"  Skipping — no XGBoost results found for {backbone}.")
            continue

        # ── Extract features from Acevedo images ──────────────────────
        print(f"  Extracting features...")
        t0 = time.time()
        model, feature_dim = load_backbone(backbone)
        X_acevedo = extract_acevedo_features(model, image_paths, device, args.batch_size)
        elapsed = time.time() - t0
        print(f"  Features: {X_acevedo.shape}, {elapsed:.1f}s")

        # Free memory
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Load KU-Optofil features for retraining ───────────────────
        print(f"  Loading KU-Optofil features for retraining...")
        kuoptofil_features = load_features(results_dir, backbone)
        best_params = xgb_results[backbone]["best_params"]

        # ── Multi-seed evaluation ─────────────────────────────────────
        print(f"  Evaluating ({len(SEEDS)} seeds)...")
        eval_results = evaluate_multiseed(
            best_params, kuoptofil_features, X_acevedo, y_acevedo_int,
            shared_le, SEEDS,
        )

        print(f"\n  Acevedo macro F1:    {eval_results['macro_f1_mean']:.4f} ± {eval_results['macro_f1_std']:.4f}")
        print(f"  Acevedo weighted F1: {eval_results['weighted_f1_mean']:.4f} ± {eval_results['weighted_f1_std']:.4f}")
        print(f"  Acevedo accuracy:    {eval_results['accuracy_mean']:.4f} ± {eval_results['accuracy_std']:.4f}")
        print(f"  Acevedo balanced acc:{eval_results['balanced_accuracy_mean']:.4f} ± {eval_results['balanced_accuracy_std']:.4f}")

        # ── Confusion matrix ──────────────────────────────────────────
        plot_confusion_matrix(
            eval_results["confusion_matrix"],
            list(shared_le.classes_),
            backbone,
            results_dir,
        )

        all_results[backbone] = eval_results
        print()

    # ── Save results ───────────────────────────────────────────────────
    out_path = results_dir / "external_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_path}")

    # ── Comparison table ───────────────────────────────────────────────
    if all_results:
        rows = []
        for backbone in BACKBONES:
            if backbone not in all_results:
                continue
            r = all_results[backbone]
            rows.append({
                "Backbone": BACKBONE_DISPLAY.get(backbone, backbone),
                "Macro F1": f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}",
                "Weighted F1": f"{r['weighted_f1_mean']:.4f} ± {r['weighted_f1_std']:.4f}",
                "Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                "Balanced Acc.": f"{r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f}",
            })
        comp_df = pd.DataFrame(rows)
        comp_df.to_csv(results_dir / "external_comparison.csv", index=False)
        print(f"Saved external_comparison.csv")
        print()
        print(comp_df.to_string(index=False))

        # Per-class F1 comparison
        per_class_rows = []
        for backbone in BACKBONES:
            if backbone not in all_results:
                continue
            report = all_results[backbone]["classification_report"]
            for cls in SHARED_CLASSES:
                if cls in report:
                    per_class_rows.append({
                        "Backbone": BACKBONE_DISPLAY.get(backbone, backbone),
                        "Class": CLASS_LABELS.get(cls, cls),
                        "Precision": report[cls]["precision"],
                        "Recall": report[cls]["recall"],
                        "F1": report[cls]["f1-score"],
                        "Support": report[cls]["support"],
                    })
        if per_class_rows:
            pc_df = pd.DataFrame(per_class_rows)
            pc_df.to_csv(results_dir / "external_per_class.csv", index=False)
            print(f"Saved external_per_class.csv")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
