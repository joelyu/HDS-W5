#!/usr/bin/env python3
"""EDA for KU-Optofil PBC dataset.

Outputs saved to results/:
    class_distribution.png          — bar chart of images per class
    split_sizes.png                 — images + patients per split with percentages
    class_by_split.png              — class distribution within each split (with %)
    patient_distribution.png        — box + strip plot with Q1/Q3/min/max
    sample_grid.png                 — example images for each class
    eda_summary.txt                 — console summary (split sizes, integrity checks, dimensions)

Usage:
    python scripts/01_data_exploration.py
    python scripts/01_data_exploration.py --data-dir /path/to/data/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import CLASS_LABELS, CLASS_ORDER, COLOURS, FOLDER_NAME_MAP, SPLIT_COLOURS, SPLIT_ORDER, style_axis


# ── Helpers ─────────────────────────────────────────────────────────────────


def check_images_exist(data_dir: Path) -> Path:
    """Verify images are extracted. Returns image root path."""
    image_root = data_dir / "dataset"
    if image_root.exists() and any(image_root.iterdir()):
        return image_root

    print(
        f"ERROR: {image_root} not found. Run 00_download_data.py first "
        f"(it downloads and extracts automatically).",
        file=sys.stderr,
    )
    sys.exit(1)


def resolve_image_path(row: pd.Series, image_root: Path) -> Path:
    """Build the actual filesystem path for an image from its metadata row."""
    class_folder = FOLDER_NAME_MAP[row["cell_type"]]
    original_split = row["path"].split("/")[0]
    return image_root / original_split / class_folder / row["image_name"]


def load_sample_image(path: Path) -> np.ndarray | None:
    """Load a single image as numpy array, or None if missing."""
    if not path.exists():
        return None
    return np.array(Image.open(path))


# ── Plots ───────────────────────────────────────────────────────────────────


def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df["cell_type"].value_counts().reindex(CLASS_ORDER)
    labels = [CLASS_LABELS[c] for c in counts.index]
    total = len(df)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(counts)), counts.values, color=COLOURS["secondary"],
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution — KU-Optofil PBC (31,484 images, 13 classes)")
    style_axis(ax)

    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{int(val)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_split_sizes(df: pd.DataFrame, out: Path) -> None:
    total_images = len(df)
    total_patients = df["patient_id"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Images per split
    split_img_counts = df["split"].value_counts().reindex(SPLIT_ORDER)
    bars = axes[0].bar(SPLIT_ORDER, split_img_counts.values, color=SPLIT_COLOURS,
                       edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, split_img_counts.values):
        pct = val / total_images * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{int(val)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Images per split")
    style_axis(axes[0])

    # Patients per split
    split_pat_counts = df.groupby("split")["patient_id"].nunique().reindex(SPLIT_ORDER)
    bars = axes[1].bar(SPLIT_ORDER, split_pat_counts.values, color=SPLIT_COLOURS,
                       edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, split_pat_counts.values):
        pct = val / total_patients * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{int(val)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Number of patients")
    axes[1].set_title("Patients per split")
    style_axis(axes[1])

    fig.suptitle("Patient-level split sizes", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_class_by_split(df: pd.DataFrame, out: Path) -> None:
    ct = pd.crosstab(df["cell_type"], df["split"]).reindex(index=CLASS_ORDER, columns=SPLIT_ORDER)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    labels = [CLASS_LABELS[c] for c in CLASS_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Absolute counts
    x = np.arange(len(CLASS_ORDER))
    width = 0.25
    for i, (split, colour) in enumerate(zip(SPLIT_ORDER, SPLIT_COLOURS)):
        bars = axes[0].bar(x + i * width, ct[split].values, width, label=split,
                           color=colour, edgecolor="white", linewidth=0.3)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Images per class per split (absolute)")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Split")
    style_axis(axes[0])

    # Proportions (stacked) with percentage labels
    bottom = np.zeros(len(CLASS_ORDER))
    for split, colour in zip(SPLIT_ORDER, SPLIT_COLOURS):
        vals = ct_norm[split].values
        axes[1].bar(x, vals, bottom=bottom, label=split, color=colour,
                    edgecolor="white", linewidth=0.3)
        # Label percentages in the middle of each segment (only if > 8%)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.08:
                axes[1].text(j, b + v / 2, f"{v:.0%}", ha="center", va="center",
                             fontsize=6, color="white", fontweight="bold")
        bottom += vals
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Split proportions per class")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(title="Split")
    style_axis(axes[1])

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_patient_distribution(df: pd.DataFrame, out: Path) -> None:
    imgs_per_patient = df.groupby("patient_id").size()

    q1, median, q3 = imgs_per_patient.quantile([0.25, 0.5, 0.75])
    mn, mx = imgs_per_patient.min(), imgs_per_patient.max()

    fig, ax = plt.subplots(figsize=(9, 4))

    # Box plot on top
    bp = ax.boxplot(imgs_per_patient, vert=False, widths=0.6,
                    patch_artist=True, positions=[1],
                    boxprops=dict(facecolor=COLOURS["muted"], edgecolor=COLOURS["secondary"]),
                    medianprops=dict(color=COLOURS["primary"], linewidth=2),
                    whiskerprops=dict(color=COLOURS["secondary"]),
                    capprops=dict(color=COLOURS["secondary"]),
                    flierprops=dict(marker="o", markerfacecolor=COLOURS["highlight"],
                                    markeredgecolor="none", markersize=4, alpha=0.6))

    # Strip plot (jittered dots)
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(imgs_per_patient))
    ax.scatter(imgs_per_patient.values, 1 + jitter, alpha=0.3, s=8,
               color=COLOURS["secondary"], zorder=2)

    # Stats annotation
    stats_text = (f"Min: {mn:.0f}  |  Q1: {q1:.0f}  |  Median: {median:.0f}  "
                  f"|  Q3: {q3:.0f}  |  Max: {mx:.0f}")
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, ha="center", fontsize=9,
            color=COLOURS["tertiary"])

    ax.set_xlabel("Images per patient")
    ax.set_yticks([])
    ax.set_title(f"Patient distribution ({df['patient_id'].nunique()} patients)")
    style_axis(ax)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_sample_grid(df: pd.DataFrame, image_root: Path, out: Path, n_per_class: int = 5) -> None:
    n_classes = len(CLASS_ORDER)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2.8, n_classes * 2.5))

    for i, cls in enumerate(CLASS_ORDER):
        cls_df = df[df["cell_type"] == cls]
        subset = cls_df.sample(n=min(n_per_class, len(cls_df)), random_state=42)
        for j in range(n_per_class):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                label = CLASS_LABELS[cls]
                count = len(cls_df)
                ax.set_ylabel(f"{label}\n(n={count})", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=10)
            if j < len(subset):
                row = subset.iloc[j]
                img_path = resolve_image_path(row, image_root)
                img = load_sample_image(img_path)
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes)
            else:
                ax.axis("off")

    fig.suptitle("Sample images per class", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def check_image_dimensions(df: pd.DataFrame, image_root: Path, sample_n: int = 500) -> str:
    """Check image dimensions on a random sample. Returns summary string."""
    sample = df.sample(n=min(sample_n, len(df)), random_state=42)
    dims = set()
    for _, row in sample.iterrows():
        img_path = resolve_image_path(row, image_root)
        if img_path.exists():
            with Image.open(img_path) as img:
                dims.add(img.size)

    if len(dims) == 1:
        w, h = dims.pop()
        return f"All images: {w}×{h} px (checked {sample_n} samples)"
    else:
        return f"Mixed dimensions found ({len(dims)} unique): {sorted(dims)[:5]}"


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for KU-Optofil PBC dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Directory containing dataset.zip and metadata CSVs (default: <repo>/data/raw/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Output directory for plots (default: <repo>/results/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    results_dir: Path = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ───────────────────────────────────────────────────
    meta_path = data_dir / "metadata_with_patient_level_splits.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.", file=sys.stderr)
        return 1

    df = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(df)} rows, {df.columns.tolist()}")
    print(f"Classes: {sorted(df['cell_type'].unique())}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print()

    # ── Verify images ──────────────────────────────────────────────────
    image_root = check_images_exist(data_dir)
    print()

    # ── Patient-level split integrity check ─────────────────────────────
    print("=== Patient-level split integrity ===")
    patients_per_split = df.groupby("patient_id")["split"].nunique()
    leaked = patients_per_split[patients_per_split > 1]
    if len(leaked) == 0:
        print("  PASS: No patient appears in multiple splits.")
    else:
        print(f"  FAIL: {len(leaked)} patient(s) appear in multiple splits!")
        print(f"  Patient IDs: {leaked.index.tolist()[:10]}")

    total_images = len(df)
    total_patients = df["patient_id"].nunique()
    for split in SPLIT_ORDER:
        split_df = df[df["split"] == split]
        n_patients = split_df["patient_id"].nunique()
        n_images = len(split_df)
        print(f"  {split:12s}: {n_images:6d} images ({n_images/total_images*100:5.1f}%), "
              f"{n_patients:4d} patients ({n_patients/total_patients*100:5.1f}%)")
    print()

    # ── Class distribution stats ────────────────────────────────────────
    print("=== Class distribution ===")
    class_counts = df["cell_type"].value_counts().reindex(CLASS_ORDER)
    for cls, count in class_counts.items():
        print(f"  {cls:25s}: {count:6d}  ({count / total_images * 100:5.1f}%)")
    print()

    # ── Image dimensions ────────────────────────────────────────────────
    print("=== Image dimensions ===")
    dim_summary = check_image_dimensions(df, image_root)
    print(f"  {dim_summary}")
    print()

    # ── Patient distribution stats ──────────────────────────────────────
    imgs_per_patient = df.groupby("patient_id").size()
    q1, median, q3 = imgs_per_patient.quantile([0.25, 0.5, 0.75])
    print("=== Patient distribution (images per patient) ===")
    print(f"  Min: {imgs_per_patient.min()}, Q1: {q1:.0f}, Median: {median:.0f}, "
          f"Q3: {q3:.0f}, Max: {imgs_per_patient.max()}")
    print()

    # ── Save summary ────────────────────────────────────────────────────
    summary_path = results_dir / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write("KU-Optofil PBC — EDA Summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Total images:   {total_images}\n")
        f.write(f"Total patients: {total_patients}\n")
        f.write(f"Classes:        {len(df['cell_type'].unique())}\n")
        f.write(f"Dimensions:     {dim_summary}\n\n")

        f.write("Patient-level split integrity: ")
        f.write("PASS\n" if len(leaked) == 0 else f"FAIL ({len(leaked)} leaked)\n")

        f.write("\nSplit sizes:\n")
        for split in SPLIT_ORDER:
            split_df = df[df["split"] == split]
            n_img = len(split_df)
            n_pat = split_df["patient_id"].nunique()
            f.write(f"  {split:12s}: {n_img:6d} images ({n_img/total_images*100:5.1f}%), "
                    f"{n_pat:4d} patients ({n_pat/total_patients*100:5.1f}%)\n")

        f.write("\nClass distribution:\n")
        for cls, count in class_counts.items():
            f.write(f"  {cls:25s}: {count:6d}  ({count / total_images * 100:5.1f}%)\n")

        f.write(f"\nPatient distribution (images per patient):\n")
        f.write(f"  Min: {imgs_per_patient.min()}, Q1: {q1:.0f}, Median: {median:.0f}, "
                f"Q3: {q3:.0f}, Max: {imgs_per_patient.max()}\n")

    print(f"  Saved {summary_path}")
    print()

    # ── Plots ───────────────────────────────────────────────────────────
    print("=== Generating plots ===")
    plot_class_distribution(df, results_dir / "class_distribution.png")
    plot_split_sizes(df, results_dir / "split_sizes.png")
    plot_class_by_split(df, results_dir / "class_by_split.png")
    plot_patient_distribution(df, results_dir / "patient_distribution.png")
    plot_sample_grid(df, image_root, results_dir / "sample_grid.png")

    print()
    print("Done. All outputs in", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
