#!/usr/bin/env python3
"""Quick linear probe baseline on frozen backbone features.

Trains LogisticRegression (= single linear layer + softmax + cross-entropy)
on the same extracted features used by XGBoost, for direct comparison.

This is mathematically identical to the linear probing protocol used by
DinoBloom (Koch et al., MICCAI 2024) for evaluating foundation models.

Usage:
    python scripts/03b_linear_probe.py
    python scripts/03b_linear_probe.py --backbone dinobloom_s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import BACKBONES, SEEDS, TAVAKOLI_51, load_features, reduce_to_5class


def evaluate_with_seeds(
    X_trainval,
    y_trainval,
    X_test,
    y_test,
    le,
    seeds: list[int],
) -> dict:
    """Train linear probe on train+val, evaluate on test across multiple seeds."""
    all_metrics = []
    all_preds = []

    for seed in seeds:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(X_trainval, y_trainval)
        preds = model.predict(X_test)

        metrics = {
            "seed": seed,
            "macro_f1": float(f1_score(y_test, preds, average="macro")),
            "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
            "accuracy": float(accuracy_score(y_test, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        }
        all_metrics.append(metrics)
        all_preds.append(preds)

    macro_f1s = [m["macro_f1"] for m in all_metrics]
    weighted_f1s = [m["weighted_f1"] for m in all_metrics]
    accuracies = [m["accuracy"] for m in all_metrics]
    balanced_accs = [m["balanced_accuracy"] for m in all_metrics]

    # 95% confidence interval: t * (std / sqrt(n))
    n = len(seeds)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))

    def _ci(values):
        return float(t_crit * np.std(values, ddof=1) / np.sqrt(n))

    summary = {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
        "macro_f1_ci95": _ci(macro_f1s),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s, ddof=1)),
        "weighted_f1_ci95": _ci(weighted_f1s),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)),
        "accuracy_ci95": _ci(accuracies),
        "balanced_accuracy_mean": float(np.mean(balanced_accs)),
        "balanced_accuracy_std": float(np.std(balanced_accs, ddof=1)),
        "balanced_accuracy_ci95": _ci(balanced_accs),
        "n_seeds": n,
        "t_critical": t_crit,
        "per_seed": all_metrics,
    }

    # Per-class report from median seed
    median_idx = int(np.argsort(macro_f1s)[len(macro_f1s) // 2])
    median_preds = all_preds[median_idx]
    summary["classification_report"] = classification_report(
        y_test, median_preds, target_names=le.classes_, output_dict=True
    )
    summary["confusion_matrix"] = confusion_matrix(y_test, median_preds).tolist()

    # Per-image predictions of the median-seed model + the true labels, for
    # paired statistical comparison (McNemar / bootstrap) in 05. Label-encoded
    # ints; same order across models (shared test-split iteration order), which
    # 05 verifies before pairing.
    summary["median_predictions"] = [int(x) for x in median_preds]
    summary["test_y_true"] = [int(x) for x in y_test]

    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Linear probe baseline on frozen features.")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument("--backbone", choices=BACKBONES, default=None)
    parser.add_argument("--five-class", action="store_true", help="Merge to 5-class WBC differential before training.")
    parser.add_argument(
        "--feature-set", choices=["all", "tavakoli"], default="all",
        help="For handcrafted backbones: 'tavakoli' keeps only the 51 baseline features.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    backbones_to_run = [args.backbone] if args.backbone else BACKBONES
    suffix = "_5class" if args.five_class else ""
    all_results = {}

    for backbone in backbones_to_run:
        data = load_features(results_dir, backbone)

        feat_suffix = ""
        if args.feature_set == "tavakoli":
            fn = data["feature_names"]
            if fn is None:
                print(f"ERROR: --feature-set tavakoli needs feature_names; {backbone} has none.", file=sys.stderr)
                return 1
            keep = [i for i, name in enumerate(fn) if name in set(TAVAKOLI_51)]
            if len(keep) != 51:
                print(f"WARNING: matched {len(keep)}/51 Tavakoli features.")
            for key in ("train_X", "val_X", "test_X"):
                data[key] = data[key][:, keep]
            feat_suffix = "_tavakoli"

        if args.five_class:
            data = reduce_to_5class(data)

        result_key = f"{backbone}{feat_suffix}{suffix}"
        print(f"\n{'='*60}")
        print(f"  {result_key} — Linear Probe")
        print(f"{'='*60}")
        if feat_suffix:
            print("  Feature subset: Tavakoli-51")

        X_train, y_train = data["train_X"], data["train_y"]
        X_val, y_val = data["val_X"], data["val_y"]
        X_test, y_test = data["test_X"], data["test_y"]
        le = data["label_encoder"]

        mode = "5-class" if args.five_class else "13-class"
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape} ({mode})")

        # Combine train+val (no HP tuning needed — just regularisation default)
        X_trainval = np.concatenate([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        # Scale features — logistic regression is sensitive to feature scale
        scaler = StandardScaler()
        X_trainval = scaler.fit_transform(X_trainval)
        X_test = scaler.transform(X_test)

        print(f"  Evaluating ({len(SEEDS)} seeds)...")
        eval_results = evaluate_with_seeds(
            X_trainval, y_trainval, X_test, y_test, le, SEEDS
        )

        print(f"  Macro F1:      {eval_results['macro_f1_mean']:.4f} ± {eval_results['macro_f1_std']:.4f}")
        print(f"  Weighted F1:   {eval_results['weighted_f1_mean']:.4f} ± {eval_results['weighted_f1_std']:.4f}")
        print(f"  Accuracy:      {eval_results['accuracy_mean']:.4f} ± {eval_results['accuracy_std']:.4f}")
        print(f"  Balanced Acc:  {eval_results['balanced_accuracy_mean']:.4f} ± {eval_results['balanced_accuracy_std']:.4f}")

        all_results[result_key] = {"test_results": eval_results}

    # Save results (merge with existing if running per-backbone)
    out_path = results_dir / "linear_probe_results.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  {'Backbone':<20} {'Macro F1':>12} {'Weighted F1':>14} {'Accuracy':>12}")
    print(f"{'='*70}")
    for bb, res in all_results.items():
        tr = res["test_results"]
        print(
            f"  {bb:<20} "
            f"{tr['macro_f1_mean']:.4f}±{tr['macro_f1_std']:.4f} "
            f"{tr['weighted_f1_mean']:.4f}±{tr['weighted_f1_std']:.4f} "
            f"{tr['accuracy_mean']:.4f}±{tr['accuracy_std']:.4f}"
        )
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
