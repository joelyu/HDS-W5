#!/usr/bin/env python3
"""Cross-model evaluation: exploration tables + statistical comparison.

Loads results from all three paradigms (XGBoost, linear probe, fine-tune) and:
  * builds browsable tables (one xlsx workbook + live DataFrames),
  * renders per-entity plots (one confusion matrix per model, per-paradigm
    per-class F1 heatmaps),
  * runs the substantive deliverable — paired statistical tests (McNemar +
    bootstrap-delta macro-F1, Holm-corrected) on the comparisons that carry the
    report's argument.

05 is an exploration/decision tool: its descriptive tables and plots are scratch
for choosing what to show — report figures/tables are regenerated in the qmd.
The model_comparisons output, by contrast, is the evidence behind every
"model A beats model B" claim, so its correctness matters.

The result key encodes base backbone + feature-set (_tavakoli) + class-mode
(_5class); the paradigm is which file the key lives in. config.parse_key
decodes the former, load_all_results the latter.

Outputs (results/):
    comparison.xlsx              — master / cross_paradigm / handcrafted_ablation / model_comparisons
    confusion/{paradigm}_{key}.png
    per_class_f1_{paradigm}.png
    clinical_focus.txt
    model_comparisons.txt

Usage:
    python scripts/05_evaluation.py
    python -i scripts/05_evaluation.py     # leaves a module-level `tables` dict of DataFrames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    BACKBONE_DISPLAY,
    CLASS_LABELS,
    CLASS_ORDER,
    CLASS_ORDER_ALPHA,
    FIVE_CLASS_LABELS,
    FIVE_CLASS_ORDER,
    parse_key,
    segmentation_of,
)
from stats import bootstrap_delta, holm_correction, mcnemar_test  # noqa: E402

# Paradigm -> result file. Order here is the display order in tables.
PARADIGM_FILES = {
    "XGBoost": "xgboost_results.json",
    "Linear probe": "linear_probe_results.json",
    "Fine-tune": "finetune_results.json",
}
_PARADIGM_SLUG = {"XGBoost": "xgboost", "Linear probe": "linear_probe", "Fine-tune": "finetune"}

METRICS = ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"]

# Clinically focal classes: blast (leukaemia) + band/myelocyte/metamyelocyte
# (the left-shift markers — infection / CML), which are also the rarest, hardest
# classes. classification_report keys are the raw class names.
CLINICAL_CLASSES = ["band_neutrophil", "blast", "myelocyte", "metamyelocyte"]

# The comparisons that carry the report's argument. All XGBoost / 13-class /
# full features unless the key/paradigm says otherwise. Missing sides are
# skipped gracefully while the run is partial.
COMPARISON_PAIRS = [
    {"label": "Foundation vs handcrafted (main axis)", "a": ("XGBoost", "dinobloom_s"), "b": ("XGBoost", "handcrafted_cellpose")},
    {"label": "Foundation vs handcrafted (baseline seg)", "a": ("XGBoost", "dinobloom_s"), "b": ("XGBoost", "handcrafted")},
    {"label": "Domain vs generic pretraining", "a": ("XGBoost", "dinobloom_s"), "b": ("XGBoost", "efficientnet_b0")},
    {"label": "Domain vs generic (CNN)", "a": ("XGBoost", "dinobloom_s"), "b": ("XGBoost", "resnet50")},
    {"label": "Architecture vs pretraining", "a": ("XGBoost", "dinobloom_s"), "b": ("XGBoost", "vit_s16")},
    {"label": "Segmentation effect (handcrafted)", "a": ("XGBoost", "handcrafted_cellpose"), "b": ("XGBoost", "handcrafted")},
    {"label": "Feature extensions (convex-hull)", "a": ("XGBoost", "handcrafted"), "b": ("XGBoost", "handcrafted_tavakoli")},
    {"label": "Feature extensions (CellPose)", "a": ("XGBoost", "handcrafted_cellpose"), "b": ("XGBoost", "handcrafted_cellpose_tavakoli")},
    {"label": "Nonlinearity (DinoBloom)", "a": ("XGBoost", "dinobloom_s"), "b": ("Linear probe", "dinobloom_s")},
]


# ── Loading ──────────────────────────────────────────────────────────────────


def load_all_results(results_dir: Path) -> dict[str, dict]:
    """Load each paradigm's result JSON, skipping any that are absent."""
    loaded: dict[str, dict] = {}
    for paradigm, fname in PARADIGM_FILES.items():
        path = results_dir / fname
        if path.exists():
            with open(path) as f:
                loaded[paradigm] = json.load(f)
            print(f"  {paradigm}: {len(loaded[paradigm])} entries ({fname})")
        else:
            print(f"  {paradigm}: not found ({fname}) — skipping")
    return loaded


# ── Tables (return DataFrames) ───────────────────────────────────────────────


def build_master_df(loaded: dict[str, dict]) -> pd.DataFrame:
    """One row per (paradigm, key) with all metrics — the explore-everything sheet."""
    rows = []
    for paradigm, results in loaded.items():
        for key, entry in results.items():
            tr = entry.get("test_results", {})
            ax = parse_key(key)
            row = {
                "paradigm": paradigm,
                "key": key,
                "base": ax["base"],
                "display": BACKBONE_DISPLAY.get(ax["base"], ax["base"]),
                "segmentation": segmentation_of(ax["base"]),
                "feature_set": ax["feature_set"],
                "class_mode": ax["class_mode"],
            }
            for m in METRICS:
                row[f"{m}_mean"] = tr.get(f"{m}_mean")
                row[f"{m}_std"] = tr.get(f"{m}_std")
            row["n_seeds"] = tr.get("n_seeds")
            row["tuning_time_s"] = entry.get("tuning_time_s")
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["class_mode", "paradigm", "macro_f1_mean"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    return df


def build_cross_paradigm_df(master: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 of each backbone under Linear probe / XGBoost / Fine-tune (13-class, full features)."""
    if master.empty:
        return pd.DataFrame()
    sub = master[(master["class_mode"] == "13class") & (master["feature_set"] == "all")]
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index="display", columns="paradigm", values="macro_f1_mean", aggfunc="first")
    cols = [p for p in PARADIGM_FILES if p in pivot.columns]
    return pivot[cols].reset_index()


def build_handcrafted_ablation_df(master: pd.DataFrame) -> pd.DataFrame:
    """2x2: segmentation (convex-hull/CellPose) x feature-set (Tavakoli-51/+ext), XGBoost macro-F1."""
    if master.empty:
        return pd.DataFrame()
    sub = master[
        (master["paradigm"] == "XGBoost")
        & (master["class_mode"] == "13class")
        & (master["base"].isin(["handcrafted", "handcrafted_cellpose"]))
    ]
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index="segmentation", columns="feature_set", values="macro_f1_mean", aggfunc="first")
    cols = [c for c in ["tavakoli", "all"] if c in pivot.columns]
    pivot = pivot[cols].rename(columns={"tavakoli": "Tavakoli-51", "all": "+extensions"})
    return pivot.reset_index()


# ── Statistical comparison (the deliverable) ─────────────────────────────────


def _get_predictions(loaded, paradigm, key):
    """Return (median_predictions, test_y_true) as int arrays, or None if absent."""
    entry = loaded.get(paradigm, {}).get(key)
    if entry is None:
        return None
    tr = entry.get("test_results", {})
    preds, y = tr.get("median_predictions"), tr.get("test_y_true")
    if preds is None or y is None:
        return None
    return np.asarray(preds), np.asarray(y)


def compare_models(loaded, pairs=COMPARISON_PAIRS, n_boot=2000, seed=42) -> pd.DataFrame:
    """McNemar + bootstrap-delta macro-F1 over the report's comparison pairs.

    Asserts paired alignment (same y_true vector) before each test; Holm-corrects
    the McNemar p-values across the OK comparisons.
    """
    rows, raw_p = [], []
    for pair in pairs:
        label_a = f'{pair["a"][1]} [{pair["a"][0]}]'
        label_b = f'{pair["b"][1]} [{pair["b"][0]}]'
        a = _get_predictions(loaded, *pair["a"])
        b = _get_predictions(loaded, *pair["b"])
        if a is None or b is None:
            rows.append({"comparison": pair["label"], "A": label_a, "B": label_b, "status": "missing"})
            continue
        (preds_a, y_a), (preds_b, y_b) = a, b
        if len(y_a) != len(y_b) or not np.array_equal(y_a, y_b):
            raise ValueError(
                f"Misaligned test labels for '{pair['label']}' ({label_a} vs {label_b}): "
                "predictions are not paired over the same images. Check feature .npz test ordering."
            )
        mc = mcnemar_test(preds_a, preds_b, y_a)
        bs = bootstrap_delta(preds_a, preds_b, y_a, metric="macro_f1", n_boot=n_boot, seed=seed)
        rows.append({
            "comparison": pair["label"], "A": label_a, "B": label_b, "status": "ok",
            "delta_macro_f1": bs["delta"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
            "mcnemar_chi2": mc["statistic"], "p_raw": mc["pvalue"], "b": mc["b"], "c": mc["c"],
        })
        raw_p.append(mc["pvalue"])

    adj = iter(holm_correction(raw_p))
    for row in rows:
        if row["status"] == "ok":
            row["p_holm"] = next(adj)
            row["significant_0.05"] = bool(row["p_holm"] < 0.05)
        else:
            row["p_holm"] = None
            row["significant_0.05"] = None

    df = pd.DataFrame(rows)
    cols = ["comparison", "A", "B", "status", "delta_macro_f1", "ci_low", "ci_high",
            "mcnemar_chi2", "p_raw", "p_holm", "significant_0.05", "b", "c"]
    return df.reindex(columns=[c for c in cols if c in df.columns])


# ── Plots (split for readability) ────────────────────────────────────────────


def _plot_one_cm(cm: np.ndarray, names: list[str], title: str, out_path: Path) -> None:
    cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            v = cm_norm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Recall")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cm_label_names(n: int) -> list[str]:
    """Class display names for an n x n confusion matrix (sklearn alphabetical order)."""
    if n == len(CLASS_ORDER_ALPHA):
        return [CLASS_LABELS[c] for c in CLASS_ORDER_ALPHA]
    if n == len(FIVE_CLASS_ORDER):
        return [FIVE_CLASS_LABELS.get(c, c) for c in FIVE_CLASS_ORDER]
    return [str(i) for i in range(n)]


def plot_confusion_split(loaded, results_dir: Path) -> None:
    """One full-size confusion matrix PNG per (paradigm, key)."""
    out_dir = results_dir / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for paradigm, results in loaded.items():
        for key, entry in results.items():
            cm = entry.get("test_results", {}).get("confusion_matrix")
            if cm is None:
                continue
            cm = np.array(cm, dtype=float)
            names = _cm_label_names(cm.shape[0])
            title = f"{BACKBONE_DISPLAY.get(parse_key(key)['base'], key)} — {paradigm}"
            _plot_one_cm(cm, names, title, out_dir / f"{_PARADIGM_SLUG[paradigm]}_{key}.png")
            count += 1
    print(f"  Saved {count} confusion matrices to {out_dir}/")


def plot_per_class_heatmaps(loaded, results_dir: Path) -> None:
    """One per-class F1 heatmap per paradigm (13-class keys; methods as columns)."""
    for paradigm, results in loaded.items():
        keys = [k for k in results if parse_key(k)["class_mode"] == "13class"]
        if not keys:
            continue
        data = {}
        for key in keys:
            report = results[key].get("test_results", {}).get("classification_report", {})
            data[BACKBONE_DISPLAY.get(parse_key(key)["base"], key)] = [
                report.get(cls, {}).get("f1-score", 0.0) for cls in CLASS_ORDER
            ]
        df = pd.DataFrame(data, index=[CLASS_LABELS[c] for c in CLASS_ORDER])
        fig, ax = plt.subplots(figsize=(1.4 * len(df.columns) + 2, 7))
        im = ax.imshow(df.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=8)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                v = df.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else "black")
        ax.set_title(f"Per-class F1 — {paradigm} (median seed)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="F1")
        fig.tight_layout()
        out = results_dir / f"per_class_f1_{_PARADIGM_SLUG[paradigm]}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Text outputs ─────────────────────────────────────────────────────────────


def write_clinical_focus(loaded, results_dir: Path) -> None:
    """Per-class precision/recall/F1 + top misclassifications for the clinical classes, all paradigms."""
    lines = ["Clinical Focus — left-shift + leukaemia classes", "=" * 55, ""]
    for paradigm, results in loaded.items():
        keys = [k for k in results if parse_key(k)["class_mode"] == "13class"]
        if not keys:
            continue
        lines.append(f"##### {paradigm} #####\n")
        for key in keys:
            tr = results[key].get("test_results", {})
            report = tr.get("classification_report", {})
            cm = np.array(tr.get("confusion_matrix", []), dtype=float)
            lines.append(f"--- {BACKBONE_DISPLAY.get(parse_key(key)['base'], key)} ---")
            for cls in CLINICAL_CLASSES:
                if cls not in report:
                    lines.append(f"  {CLASS_LABELS.get(cls, cls)}: not in report")
                    continue
                r = report[cls]
                lines.append(
                    f"  {CLASS_LABELS.get(cls, cls)}: P={r['precision']:.3f} "
                    f"R={r['recall']:.3f} F1={r['f1-score']:.3f} n={int(r['support'])}"
                )
                if cm.size and cls in CLASS_ORDER_ALPHA:
                    idx = CLASS_ORDER_ALPHA.index(cls)
                    row = cm[idx]
                    total = row.sum()
                    if total > 0:
                        miss = sorted(
                            [(CLASS_LABELS[CLASS_ORDER_ALPHA[j]], int(row[j]), row[j] / total * 100)
                             for j in range(len(row)) if j != idx and row[j] > 0],
                            key=lambda x: -x[1],
                        )[:3]
                        for nm, ct, pct in miss:
                            lines.append(f"      -> {nm}: {ct} ({pct:.1f}%)")
            lines.append("")
        lines.append("")
    out = results_dir / "clinical_focus.txt"
    out.write_text("\n".join(lines))
    print(f"  Saved {out}")


def write_comparison_summary(comparisons: pd.DataFrame, results_dir: Path) -> None:
    """Human-readable McNemar + bootstrap summary."""
    lines = ["Statistical model comparison (McNemar + bootstrap macro-F1, Holm-corrected)",
             "=" * 74, ""]
    if comparisons.empty:
        lines.append("No comparisons available.")
    for _, r in comparisons.iterrows():
        lines.append(r["comparison"])
        lines.append(f"  {r['A']}  vs  {r['B']}")
        if r.get("status") != "ok":
            lines.append("  (skipped — predictions not available for one side)\n")
            continue
        sig = "SIGNIFICANT" if r["significant_0.05"] else "n.s."
        lines.append(
            f"  ΔmacroF1 = {r['delta_macro_f1']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        )
        lines.append(
            f"  McNemar χ²={r['mcnemar_chi2']:.2f}  p_raw={r['p_raw']:.2e}  "
            f"p_holm={r['p_holm']:.2e}  [{sig}]  (b={int(r['b'])}, c={int(r['c'])})\n"
        )
    out = results_dir / "model_comparisons.txt"
    out.write_text("\n".join(lines))
    print(f"  Saved {out}")


def write_workbook(results_dir: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write all DataFrames into one xlsx workbook (openpyxl)."""
    out = results_dir / "comparison.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for name, df in sheets.items():
            (df if not df.empty else pd.DataFrame({"(empty)": []})).to_excel(
                xl, sheet_name=name, index=False
            )
    print(f"  Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-model evaluation + statistical comparison.")
    p.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for the CI.")
    return p.parse_args()


def run(results_dir: Path, n_boot: int = 2000) -> dict:
    """Load everything, build tables, render plots, run stats. Returns the table dict."""
    results_dir = results_dir.resolve()
    print("Loading results...")
    loaded = load_all_results(results_dir)
    if not loaded:
        print("No result files found — nothing to do.")
        return {}

    master = build_master_df(loaded)
    cross = build_cross_paradigm_df(master)
    ablation = build_handcrafted_ablation_df(master)
    comparisons = compare_models(loaded, n_boot=n_boot)

    print("\n=== Tables ===")
    write_workbook(results_dir, {
        "master": master, "cross_paradigm": cross,
        "handcrafted_ablation": ablation, "model_comparisons": comparisons,
    })
    print("\n=== Plots ===")
    plot_confusion_split(loaded, results_dir)
    plot_per_class_heatmaps(loaded, results_dir)
    print("\n=== Text ===")
    write_clinical_focus(loaded, results_dir)
    write_comparison_summary(comparisons, results_dir)

    n_ok = int((comparisons.get("status") == "ok").sum()) if not comparisons.empty else 0
    print(f"\nDone. {len(master)} model entries; {n_ok}/{len(comparisons)} comparisons evaluated.")
    return {
        "master": master, "cross_paradigm": cross,
        "handcrafted_ablation": ablation, "model_comparisons": comparisons,
        "loaded": loaded,
    }


if __name__ == "__main__":
    args = parse_args()
    tables = run(args.results_dir, n_boot=args.n_boot)  # available in `python -i`
