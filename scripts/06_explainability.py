#!/usr/bin/env python3
"""Explainability visualisations for frozen backbone → XGBoost pipelines.

Generate-everything exploration tool: produces every explainability view that is
available and meaningful for the models we have. Pick the report figures later.

Produces:
    1. UMAP: 2D embedding of test set features, coloured by class — one per
       feature extractor plus a combined comparison panel (extractor-level, no
       model: shows whether the representation is class-separable).
    2. SHAP: feature contributions for the INTERPRETABLE (handcrafted) XGBoost
       classifiers only — mean-|SHAP| bars, per-class bars, and directional
       beeswarms with real feature names. Skipped for deep backbones, whose
       features are abstract dimensions (use UMAP/attention there instead).
    3. Attention maps: PCA of DinoBloom-S ViT patch tokens showing the frozen
       backbone's spatial representation.

GradCAM is intentionally absent — it needs a fine-tuned CNN (gradients to conv
layers), which the frozen→XGBoost pipeline does not provide. It rejoins with
fine-tuning.

Outputs saved to results/:
    umap_{backbone}.png                       — per-extractor UMAP
    umap_comparison.png                       — side-by-side UMAP
    shap_bar_{backbone}.png                   — mean-|SHAP| ranking (handcrafted only)
    shap_{backbone}.png                       — per-class mean-|SHAP| bars (handcrafted only)
    shap_beeswarm_{backbone}_{class}.png      — directional beeswarm, named feats (handcrafted only)
    attention_maps_dinobloom.png              — DinoBloom-S patch-PCA

Usage:
    python scripts/06_explainability.py
    python scripts/06_explainability.py --skip-shap          # UMAP + attention only
    python scripts/06_explainability.py --skip-attention      # UMAP + SHAP only
    python scripts/06_explainability.py --backbone handcrafted  # single backbone
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    BACKBONES, CLASS_COLOUR_MAP, CLASS_LABELS, CLASS_ORDER, COLOURS,
    FOLDER_NAME_MAP, SEEDS, load_features,
)


# ── Data loading ───────────────────────────────────────────────────────────


def load_xgboost_results(results_dir: Path) -> dict:
    """Load xgboost_results.json for best params."""
    path = results_dir / "xgboost_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 03_xgboost_training.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def train_best_xgboost(
    best_params: dict,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Train XGBoost with best params on train+val for explainability."""
    params = {**best_params, "random_state": seed, "n_jobs": -1, "verbosity": 0}
    params.pop("early_stopping_rounds", None)
    model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight("balanced", y_trainval)
    model.fit(X_trainval, y_trainval, sample_weight=sample_weights, verbose=False)
    return model


# ── UMAP ───────────────────────────────────────────────────────────────────


def plot_umap_single(
    features: np.ndarray,
    labels_str: np.ndarray,
    backbone: str,
    results_dir: Path,
) -> np.ndarray:
    """Compute and plot UMAP for a single backbone. Returns 2D embedding."""
    import umap

    print(f"  Computing UMAP for {backbone} ({features.shape})...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 7))

    for cls in CLASS_ORDER:
        mask = labels_str == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=CLASS_COLOUR_MAP[cls], label=CLASS_LABELS[cls],
            s=4, alpha=0.5, edgecolors="none",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP — {backbone} features (test set)")
    ax.legend(fontsize=7, markerscale=3, loc="best", framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(results_dir / f"umap_{backbone}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved umap_{backbone}.png")

    return embedding


def plot_umap_comparison(
    embeddings: dict[str, np.ndarray],
    labels_str: np.ndarray,
    results_dir: Path,
) -> None:
    """Side-by-side UMAP comparison across backbones."""
    n = len(embeddings)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (backbone, embedding) in zip(axes, embeddings.items()):
        for cls in CLASS_ORDER:
            mask = labels_str == cls
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=CLASS_COLOUR_MAP[cls], label=CLASS_LABELS[cls],
                s=3, alpha=0.4, edgecolors="none",
            )
        ax.set_title(backbone, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Single legend for last axis
    axes[-1].legend(fontsize=6, markerscale=3, loc="best", framealpha=0.8)

    fig.suptitle("UMAP comparison — test set features by backbone", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(results_dir / "umap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved umap_comparison.png")


# ── SHAP ───────────────────────────────────────────────────────────────────


def plot_shap(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    le: LabelEncoder,
    backbone: str,
    results_dir: Path,
    max_display: int = 20,
    shap_sample: int = 1000,
    feature_names: list[str] | None = None,
) -> None:
    """SHAP analysis for an XGBoost classifier.

    feature_names: real names (handcrafted backbones) for axis labels; when
    None (deep backbones) labels fall back to abstract "dim i".
    """
    import shap

    def _label(i: int) -> str:
        return feature_names[i] if feature_names is not None else f"dim {i}"

    unit = "features" if feature_names is not None else "embedding dimensions"

    print(f"  Computing SHAP values for {backbone}...")

    # Subsample for speed if test set is large
    if len(X_test) > shap_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), shap_sample, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Normalise format: SHAP < 0.42 returns list of arrays (one per class),
    # SHAP >= 0.42 returns 3D ndarray (n_samples, n_features, n_classes).
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Convert (n_samples, n_features, n_classes) → list of (n_samples, n_features)
        shap_values = [shap_values[:, :, c] for c in range(shap_values.shape[2])]

    # Bar plot — mean |SHAP| across all classes
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    # Save raw mean |SHAP| per feature to CSV for report rendering
    import csv
    shap_rows = sorted(
        [(_label(i), float(mean_abs_shap[i])) for i in range(len(mean_abs_shap))],
        key=lambda r: r[1], reverse=True,
    )
    with open(results_dir / f"shap_importance_{backbone}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "mean_abs_shap"])
        w.writerows(shap_rows)
    print(f"  Saved shap_importance_{backbone}.csv")

    # Top features
    top_idx = np.argsort(mean_abs_shap)[-max_display:]
    ax.barh(range(len(top_idx)), mean_abs_shap[top_idx], color=COLOURS["secondary"])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([_label(i) for i in top_idx], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {max_display} {unit} — {backbone}")
    fig.tight_layout()
    fig.savefig(results_dir / f"shap_bar_{backbone}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved shap_bar_{backbone}.png")

    # Per-class SHAP — which dimensions matter most for each class
    if isinstance(shap_values, list) and len(shap_values) == len(le.classes_):
        n_classes = len(le.classes_)
        n_top = min(10, X_explain.shape[1])

        fig, axes = plt.subplots(1, min(n_classes, 4), figsize=(5 * min(n_classes, 4), 5))
        # Show a subset of clinically interesting classes
        interesting = ["blast", "myelocyte", "lymphocyte", "segmented_neutrophil"]
        interesting_idx = [list(le.classes_).index(c) for c in interesting if c in le.classes_]

        if len(interesting_idx) == 1:
            axes = [axes]

        for ax, cls_idx in zip(axes, interesting_idx[:4]):
            sv = np.abs(shap_values[cls_idx]).mean(axis=0)
            top = np.argsort(sv)[-n_top:]
            ax.barh(range(len(top)), sv[top], color=COLOURS["secondary"])
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([_label(i) for i in top], fontsize=7)
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title(CLASS_LABELS.get(le.classes_[cls_idx], le.classes_[cls_idx]),
                         fontsize=10)

        fig.suptitle(f"Per-class SHAP — {backbone}", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(results_dir / f"shap_{backbone}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved shap_{backbone}.png")

    # Beeswarm (directional) — one per clinically-key class. Unlike the mean-|SHAP|
    # bars (magnitude only), the beeswarm colours each sample by its feature VALUE,
    # so it reads as "high N:C ratio pushes toward blast". Only meaningful with real
    # feature names, so it is gated on feature_names (handcrafted backbones).
    if feature_names is not None and isinstance(shap_values, list) and len(shap_values) == len(le.classes_):
        classes = list(le.classes_)
        for cls in ["blast", "myelocyte", "lymphocyte", "segmented_neutrophil"]:
            if cls not in classes:
                continue
            cls_idx = classes.index(cls)
            shap.summary_plot(
                shap_values[cls_idx], X_explain, feature_names=feature_names,
                max_display=15, show=False, plot_size=(8, 6),
            )
            fig = plt.gcf()
            # Title on the main (first-created) axis, not the colorbar shap adds.
            fig.axes[0].set_title(
                f"SHAP beeswarm — {backbone}: {CLASS_LABELS.get(cls, cls)}", fontsize=11
            )
            fig.savefig(results_dir / f"shap_beeswarm_{backbone}_{cls}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved shap_beeswarm_{backbone}_{cls}.png")


# ── DinoBloom attention maps ───────────────────────────────────────────────


def plot_attention_maps(
    data_dir: Path,
    results_dir: Path,
    n_samples: int = 8,
) -> None:
    """Visualise DinoBloom-S ViT attention via PCA of patch tokens.

    For each sample image, extracts all patch tokens from the last ViT layer
    and projects them to 3 principal components (mapped to RGB) to show what
    spatial regions the model represents differently.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.decomposition import PCA
    import pandas as pd

    print("  Loading DinoBloom-S for attention maps...")

    # Load model
    from huggingface_hub import hf_hub_download

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    ckpt_path = hf_hub_download(repo_id="MarrLab/DinoBloom", filename="pytorch_model_s.bin")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    num_tokens = int(1 + (224 / 14) ** 2)
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_tokens, 384))
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load metadata and pick diverse samples
    meta_path = data_dir / "metadata_with_patient_level_splits.csv"
    df = pd.read_csv(meta_path)
    image_root = data_dir / "dataset"

    # Pick one image from each of the first n_samples classes
    samples = []
    for cls in CLASS_ORDER[:n_samples]:
        cls_df = df[df["cell_type"] == cls]
        if len(cls_df) > 0:
            row = cls_df.sample(n=1, random_state=42).iloc[0]
            class_folder = FOLDER_NAME_MAP[row["cell_type"]]
            original_split = row["path"].split("/")[0]
            img_path = image_root / original_split / class_folder / row["image_name"]
            if img_path.exists():
                samples.append((cls, img_path))

    if not samples:
        print("  WARNING: No sample images found for attention maps.")
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    patch_size = 14
    grid_size = 224 // patch_size  # 16

    for i, (cls, img_path) in enumerate(samples):
        # Load and transform
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Extract patch tokens (skip CLS token)
        with torch.no_grad():
            tokens = model.get_intermediate_layers(img_tensor, n=1)[0]
            # tokens shape: (1, N, 384) where N is 257 (CLS+patches) or 256 (patches only)
            n_tokens = tokens.shape[1]
            expected_patches = grid_size * grid_size  # 256
            if n_tokens == expected_patches + 1:
                # CLS token present — remove it
                patch_tokens = tokens[0, 1:, :].cpu().numpy()
            else:
                patch_tokens = tokens[0, :, :].cpu().numpy()

        # PCA to 3 components → RGB
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(patch_tokens)  # (256, 3)

        # Normalise each component to [0, 1]
        for c in range(3):
            mn, mx = pca_result[:, c].min(), pca_result[:, c].max()
            if mx > mn:
                pca_result[:, c] = (pca_result[:, c] - mn) / (mx - mn)

        pca_image = pca_result.reshape(grid_size, grid_size, 3)

        # CLS attention (approximate via token norms)
        token_norms = np.linalg.norm(patch_tokens, axis=1)
        norm_map = token_norms.reshape(grid_size, grid_size)
        norm_map = (norm_map - norm_map.min()) / (norm_map.max() - norm_map.min() + 1e-8)

        # Plot: original | PCA of patches | token norm heatmap
        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(CLASS_LABELS[cls], fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pca_image)
        axes[i, 1].set_title("Patch PCA (RGB)", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(norm_map, cmap="inferno")
        axes[i, 2].set_title("Token norms", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle("DinoBloom-S attention — PCA of patch tokens", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(results_dir / "attention_maps_dinobloom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attention_maps_dinobloom.png")

    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explainability visualisations.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
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
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis.")
    parser.add_argument("--skip-attention", action="store_true", help="Skip attention maps.")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP.")
    parser.add_argument(
        "--shap-sample", type=int, default=1000,
        help="Number of test samples for SHAP (default: 1000).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    results_dir: Path = args.results_dir.resolve()

    backbones_to_run = [args.backbone] if args.backbone else BACKBONES

    # ── UMAP ───────────────────────────────────────────────────────────
    if not args.skip_umap:
        print("=== UMAP ===")
        embeddings = {}
        labels_str = None

        for backbone in backbones_to_run:
            feat = load_features(results_dir, backbone)
            labels_str = feat["test_y_str"]
            embedding = plot_umap_single(
                feat["test_X"], feat["test_y_str"], backbone, results_dir
            )
            embeddings[backbone] = embedding

        if len(embeddings) > 1 and labels_str is not None:
            plot_umap_comparison(embeddings, labels_str, results_dir)
        print()

    # ── SHAP ───────────────────────────────────────────────────────────
    if not args.skip_shap:
        print("=== SHAP ===")
        xgb_results = load_xgboost_results(results_dir)

        for backbone in backbones_to_run:
            if backbone not in xgb_results:
                print(f"  Skipping {backbone} — no XGBoost results found.")
                continue

            feat = load_features(results_dir, backbone)
            fnames = feat.get("feature_names")

            # SHAP only where features are named (handcrafted variants). On deep
            # backbones SHAP attributes importance to abstract dimensions ("dim 47"),
            # which has no clinical meaning — UMAP/attention are the right tools there.
            if fnames is None:
                print(f"  Skipping SHAP for {backbone} — deep features (abstract dimensions, not interpretable).")
                continue

            best_params = xgb_results[backbone]["best_params"]

            # Train model on train+val
            X_trainval = np.concatenate([feat["train_X"], feat["val_X"]])
            y_trainval = np.concatenate([feat["train_y"], feat["val_y"]])

            print(f"  Training XGBoost for {backbone}...")
            model = train_best_xgboost(best_params, X_trainval, y_trainval)

            plot_shap(
                model, feat["test_X"], feat["label_encoder"],
                backbone, results_dir, shap_sample=args.shap_sample,
                feature_names=list(fnames),
            )

        print()

    # ── Attention maps ─────────────────────────────────────────────────
    if not args.skip_attention:
        print("=== Attention maps ===")
        plot_attention_maps(data_dir, results_dir)
        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
