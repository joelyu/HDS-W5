#!/usr/bin/env python3
"""Cross-model evaluation and comparison tables.

Reads xgboost_results.json (from 03) and optionally fine-tuning results (from 04),
then produces:
    - Headline comparison table (CSV + LaTeX) with mean ± std across seeds
    - Per-class F1 comparison across backbones (heatmap + CSV)
    - Confusion matrix comparison (side-by-side normalised)
    - Clinical focus: blast and myelocyte misclassification analysis
    - TPE vs GP sampler comparison table

Outputs saved to results/:
    comparison_table.csv              — headline metrics (macro F1, weighted F1, accuracy, balanced acc)
    comparison_table.tex              — LaTeX version for report
    per_class_f1_heatmap.png          — per-class F1 across backbones
    per_class_f1.csv                  — per-class F1 values
    confusion_matrices_compared.png   — side-by-side normalised confusion matrices
    sampler_comparison.csv            — TPE vs GP best val F1 per backbone
    clinical_focus.txt                — blast/myelocyte misclassification summary

Usage:
    python scripts/05_evaluation.py
    python scripts/05_evaluation.py --results-dir results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BACKBONES, BACKBONE_DISPLAY, CLASS_LABELS, CLASS_ORDER, CLASS_ORDER_ALPHA,
    COLOURS, style_axis,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def load_results(results_dir: Path) -> dict:
    """Load xgboost_results.json. Returns parsed dict."""
    path = results_dir / "xgboost_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 03_xgboost_training.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def fmt_metric(mean: float, std: float) -> str:
    """Format metric as 'mean ± std' with 4 decimal places."""
    return f"{mean:.4f} ± {std:.4f}"


def fmt_metric_pct(mean: float, std: float) -> str:
    """Format metric as percentage 'mean ± std'."""
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


# ── Headline comparison table ──────────────────────────────────────────────


def make_comparison_table(results: dict, results_dir: Path) -> pd.DataFrame:
    """Build and save the headline comparison table."""
    rows = []
    for backbone in BACKBONES:
        if backbone not in results:
            continue
        tr = results[backbone]["test_results"]
        rows.append({
            "Backbone": BACKBONE_DISPLAY.get(backbone, backbone),
            "Macro F1": fmt_metric(tr["macro_f1_mean"], tr["macro_f1_std"]),
            "Weighted F1": fmt_metric(tr["weighted_f1_mean"], tr["weighted_f1_std"]),
            "Accuracy": fmt_metric(tr["accuracy_mean"], tr["accuracy_std"]),
            "Balanced Acc.": fmt_metric(tr["balanced_accuracy_mean"], tr["balanced_accuracy_std"]),
            "Best Sampler": results[backbone]["best_sampler"],
            "Val Macro F1": f"{results[backbone]['best_val_macro_f1']:.4f}",
        })

    df = pd.DataFrame(rows)

    # CSV
    csv_path = results_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # LaTeX
    tex_path = results_dir / "comparison_table.tex"
    # Build LaTeX manually for better control
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Test set performance of frozen backbone $\\rightarrow$ XGBoost pipelines "
                "(mean $\\pm$ std over 5 seeds).}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Backbone & Macro F1 & Weighted F1 & Accuracy & Balanced Acc. \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            # Replace ± with $\pm$ for LaTeX
            cols = [row["Backbone"]]
            for col in ["Macro F1", "Weighted F1", "Accuracy", "Balanced Acc."]:
                val = row[col].replace("±", "$\\pm$")
                cols.append(val)
            f.write(" & ".join(cols) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  Saved {tex_path}")

    return df


# ── Per-class F1 heatmap ──────────────────────────────────────────────────


def make_per_class_heatmap(results: dict, results_dir: Path) -> None:
    """Per-class F1 scores across backbones as a heatmap."""
    backbones_present = [b for b in BACKBONES if b in results]
    if not backbones_present:
        return

    # Extract per-class F1 from classification_report (median seed)
    class_names = CLASS_ORDER
    data = {}
    for backbone in backbones_present:
        report = results[backbone]["test_results"]["classification_report"]
        f1s = []
        for cls in class_names:
            if cls in report:
                f1s.append(report[cls]["f1-score"])
            else:
                f1s.append(0.0)
        data[BACKBONE_DISPLAY.get(backbone, backbone)] = f1s

    df = pd.DataFrame(data, index=[CLASS_LABELS[c] for c in class_names])

    # Save CSV
    csv_path = results_dir / "per_class_f1.csv"
    df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(df.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)

    # Annotate cells
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            colour = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=colour)

    ax.set_title("Per-class F1 score by backbone (median seed)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 Score")
    fig.tight_layout()
    fig.savefig(results_dir / "per_class_f1_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {results_dir / 'per_class_f1_heatmap.png'}")


# ── Confusion matrix comparison ──────────────────────────────────────────


def plot_confusion_matrices_compared(results: dict, results_dir: Path) -> None:
    """Side-by-side normalised confusion matrices for all backbones."""
    backbones_present = [b for b in BACKBONES if b in results]
    n = len(backbones_present)
    if n == 0:
        return

    # Confusion matrix from script 03 is ordered by le.classes_ (alphabetical)
    short_names = [CLASS_LABELS[c] for c in CLASS_ORDER_ALPHA]

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, backbone in zip(axes, backbones_present):
        cm = np.array(results[backbone]["test_results"]["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)

        for i in range(len(short_names)):
            for j in range(len(short_names)):
                val = cm_norm[i, j]
                if val > 0.005:  # Only label non-trivial cells
                    colour = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=5, color=colour)

        ax.set_xlabel("Predicted")
        if ax == axes[0]:
            ax.set_ylabel("True")
        ax.set_title(BACKBONE_DISPLAY.get(backbone, backbone), fontsize=11)

    fig.suptitle("Normalised confusion matrices — frozen backbone → XGBoost (median seed)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrices_compared.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {results_dir / 'confusion_matrices_compared.png'}")


# ── Sampler comparison ─────────────────────────────────────────────────────


def make_sampler_comparison(results: dict, results_dir: Path) -> None:
    """Compare TPE vs GP sampler performance."""
    rows = []
    for backbone in BACKBONES:
        if backbone not in results:
            continue
        r = results[backbone]
        rows.append({
            "Backbone": BACKBONE_DISPLAY.get(backbone, backbone),
            "TPE Best Val F1": f"{r['tpe_best_val_f1']:.4f}" if r["tpe_best_val_f1"] else "—",
            "TPE Trials": r["tpe_trials"],
            "GP Best Val F1": f"{r['gp_best_val_f1']:.4f}" if r.get("gp_best_val_f1") else "—",
            "GP Trials": r.get("gp_trials", "—"),
            "Winner": r["best_sampler"],
            "Tuning Time (s)": r["tuning_time_s"],
        })

    df = pd.DataFrame(rows)
    csv_path = results_dir / "sampler_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")


# ── Clinical focus ─────────────────────────────────────────────────────────


def clinical_focus(results: dict, results_dir: Path) -> None:
    """Analyse blast and myelocyte classification — clinically important classes."""
    lines = []
    lines.append("Clinical Focus — Blast and Myelocyte Classification")
    lines.append("=" * 55)
    lines.append("")

    clinical_classes = ["blast", "myelocyte", "metamyelocyte"]
    # Confusion matrix is ordered alphabetically by le.classes_

    for backbone in BACKBONES:
        if backbone not in results:
            continue
        lines.append(f"--- {BACKBONE_DISPLAY.get(backbone, backbone)} ---")
        report = results[backbone]["test_results"]["classification_report"]
        cm = np.array(results[backbone]["test_results"]["confusion_matrix"], dtype=float)

        for cls in clinical_classes:
            if cls not in report:
                lines.append(f"  {CLASS_LABELS[cls]}: not in report")
                continue
            r = report[cls]
            lines.append(f"  {CLASS_LABELS[cls]}:")
            lines.append(f"    Precision: {r['precision']:.4f}")
            lines.append(f"    Recall:    {r['recall']:.4f}")
            lines.append(f"    F1:        {r['f1-score']:.4f}")
            lines.append(f"    Support:   {r['support']}")

            # Top misclassifications from confusion matrix row
            cls_idx = CLASS_ORDER_ALPHA.index(cls)
            row = cm[cls_idx]
            total = row.sum()
            if total > 0:
                misclass = [(CLASS_LABELS[CLASS_ORDER_ALPHA[j]], int(row[j]), row[j] / total * 100)
                            for j in range(len(row)) if j != cls_idx and row[j] > 0]
                misclass.sort(key=lambda x: -x[1])
                if misclass:
                    lines.append("    Top misclassified as:")
                    for name, count, pct in misclass[:3]:
                        lines.append(f"      → {name}: {count} ({pct:.1f}%)")

        lines.append("")

    # Blast recall comparison (key clinical metric)
    lines.append("--- Blast Recall Comparison ---")
    for backbone in BACKBONES:
        if backbone not in results:
            continue
        report = results[backbone]["test_results"]["classification_report"]
        if "blast" in report:
            lines.append(f"  {BACKBONE_DISPLAY.get(backbone, backbone)}: "
                         f"recall={report['blast']['recall']:.4f}, "
                         f"F1={report['blast']['f1-score']:.4f}")
    lines.append("")

    out_path = results_dir / "clinical_focus.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-model evaluation and comparison.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir: Path = args.results_dir.resolve()

    print("Loading results...")
    results = load_results(results_dir)
    print(f"  Found {len(results)} backbone(s): {list(results.keys())}")
    print()

    # ── Headline comparison table ──────────────────────────────────────
    print("=== Comparison table ===")
    comp_df = make_comparison_table(results, results_dir)
    print()
    print(comp_df.to_string(index=False))
    print()

    # ── Per-class F1 heatmap ───────────────────────────────────────────
    print("=== Per-class F1 heatmap ===")
    make_per_class_heatmap(results, results_dir)
    print()

    # ── Confusion matrix comparison ────────────────────────────────────
    print("=== Confusion matrices ===")
    plot_confusion_matrices_compared(results, results_dir)
    print()

    # ── Sampler comparison ─────────────────────────────────────────────
    print("=== Sampler comparison ===")
    make_sampler_comparison(results, results_dir)
    print()

    # ── Clinical focus ─────────────────────────────────────────────────
    print("=== Clinical focus ===")
    clinical_focus(results, results_dir)
    print()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
