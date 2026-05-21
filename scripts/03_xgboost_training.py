#!/usr/bin/env python3
"""Train XGBoost classifiers on frozen backbone features with Optuna HP tuning.

For each backbone's features (from 02_feature_extraction.py), runs Optuna TPE
to find optimal XGBoost hyperparameters. Then evaluates the best configuration
across multiple seeds for stability.

GP sampler is available via --include-gp but disabled by default: GP surrogate
fitting (O(n³) per trial) dominates walltime when evaluations are cheap, making
it slower in wall-clock time despite needing fewer trials.

Outputs saved to results/:
    xgboost_results.json              — all metrics, best params, per-seed results
    optuna_history_{backbone}.png     — convergence plots
    confusion_matrix_{backbone}.png

Usage:
    python scripts/03_xgboost_training.py
    python scripts/03_xgboost_training.py --backbone dinobloom_s   # single backbone
    python scripts/03_xgboost_training.py --n-trials 50            # fewer trials
    python scripts/03_xgboost_training.py --include-gp             # also run GP sampler
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from config import BACKBONES, CLASS_LABELS, COLOURS, SEEDS, load_features

# Suppress Optuna's trial-level logs (we print our own progress)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Data loading ────────────────────────────────────────────────────────────


# ── Optuna objective ────────────────────────────────────────────────────────


def make_objective(X_train, y_train, X_val, y_val, n_classes: int):
    """Return an Optuna objective function for XGBoost HP tuning."""
    sample_weights = compute_sample_weight("balanced", y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "objective": "multi:softprob",
            "num_class": n_classes,
            "tree_method": "hist",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=False,
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    return objective


# ── Multi-seed evaluation ───────────────────────────────────────────────────


def evaluate_with_seeds(
    best_params: dict,
    X_trainval,
    y_trainval,
    X_test,
    y_test,
    n_classes: int,
    le: LabelEncoder,
    seeds: list[int],
) -> dict:
    """Train best HP config on train+val, evaluate on test across multiple seeds."""
    all_metrics = []
    all_preds = []
    sample_weights = compute_sample_weight("balanced", y_trainval)

    for seed in seeds:
        params = {**best_params, "random_state": seed, "n_jobs": -1, "verbosity": 0}
        # Remove early_stopping_rounds for final training (no eval_set)
        params.pop("early_stopping_rounds", None)
        model = xgb.XGBClassifier(**params)
        model.fit(X_trainval, y_trainval, sample_weight=sample_weights, verbose=False)
        preds = model.predict(X_test)

        metrics = {
            "seed": seed,
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "weighted_f1": f1_score(y_test, preds, average="weighted"),
            "accuracy": accuracy_score(y_test, preds),
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        }
        all_metrics.append(metrics)
        all_preds.append(preds)

    # Aggregate across seeds
    macro_f1s = [m["macro_f1"] for m in all_metrics]
    weighted_f1s = [m["weighted_f1"] for m in all_metrics]
    accuracies = [m["accuracy"] for m in all_metrics]
    balanced_accs = [m["balanced_accuracy"] for m in all_metrics]

    summary = {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "balanced_accuracy_mean": float(np.mean(balanced_accs)),
        "balanced_accuracy_std": float(np.std(balanced_accs)),
        "per_seed": all_metrics,
    }

    # Per-class report from median seed
    median_idx = int(np.argsort(macro_f1s)[len(macro_f1s) // 2])
    median_preds = all_preds[median_idx]
    summary["classification_report"] = classification_report(
        y_test, median_preds, target_names=le.classes_, output_dict=True
    )
    summary["confusion_matrix"] = confusion_matrix(y_test, median_preds).tolist()

    return summary


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_optuna_history(
    studies: dict[str, optuna.Study],
    backbone: str,
    results_dir: Path,
):
    """Plot optimisation convergence for TPE and/or GP."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for sampler_name, study in studies.items():
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]
        best_so_far = np.maximum.accumulate(values)
        colour = COLOURS["tpe"] if sampler_name == "TPE" else COLOURS["gp"]
        ax.plot(range(1, len(values) + 1), best_so_far, label=sampler_name, color=colour, linewidth=2)
        ax.scatter(range(1, len(values) + 1), values, alpha=0.25, s=12, color=colour)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Macro F1 (validation)")
    ax.set_title(f"Optuna Convergence — {backbone}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"optuna_history_{backbone}.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    backbone: str,
    results_dir: Path,
):
    """Plot normalised confusion matrix."""
    cm_arr = np.array(cm, dtype=float)
    cm_norm = cm_arr / cm_arr.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    short_names = [CLASS_LABELS.get(n, n) for n in class_names]
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    # Annotate cells
    for i in range(len(short_names)):
        for j in range(len(short_names)):
            val = cm_norm[i, j]
            count = int(cm_arr[i, j])
            colour = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({count})", ha="center", va="center",
                    fontsize=6, color=colour)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {backbone} → XGBoost")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Recall")
    fig.tight_layout()
    fig.savefig(results_dir / f"confusion_matrix_{backbone}.png", dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on frozen backbone features.")
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
    parser.add_argument("--n-trials", type=int, default=100, help="Max Optuna trials per sampler.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (trials).")
    parser.add_argument("--include-gp", action="store_true", help="Also run GP sampler (slow — O(n³) surrogate fitting).")
    return parser.parse_args()


class EarlyStoppingCallback:
    """Stop Optuna study if no improvement for `patience` consecutive trials."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_value: float | None = None
        self.trials_without_improvement = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            return
        if self.best_value is None or trial.value > self.best_value:
            self.best_value = trial.value
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1
        if self.trials_without_improvement >= self.patience:
            study.stop()


def run_study(
    sampler_name: str,
    sampler: optuna.samplers.BaseSampler,
    objective_fn,
    n_trials: int,
    patience: int,
) -> optuna.Study:
    """Run a single Optuna study and print progress."""
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def progress_callback(study, trial):
        if trial.value is not None:
            best = study.best_value
            print(
                f"\r  [{sampler_name}] trial {trial.number + 1:3d} | "
                f"val={trial.value:.4f} | best={best:.4f}",
                end="",
                flush=True,
            )

    early_stop = EarlyStoppingCallback(patience)

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        callbacks=[progress_callback, early_stop],
        catch=(ValueError,),
    )
    print()
    return study


def main() -> int:
    args = parse_args()
    results_dir: Path = args.results_dir.resolve()

    backbones_to_run = [args.backbone] if args.backbone else BACKBONES
    all_results = {}

    for backbone in backbones_to_run:
        print(f"\n{'='*60}")
        print(f"  {backbone}")
        print(f"{'='*60}")

        # Load features
        data = load_features(results_dir, backbone)
        X_train, y_train = data["train_X"], data["train_y"]
        X_val, y_val = data["val_X"], data["val_y"]
        X_test, y_test = data["test_X"], data["test_y"]
        le = data["label_encoder"]
        n_classes = len(le.classes_)

        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"  Classes: {n_classes}")
        print()

        objective_fn = make_objective(X_train, y_train, X_val, y_val, n_classes)

        # ── Optuna studies ──────────────────────────────────────────────
        studies = {}
        t0 = time.time()

        # TPE sampler
        print("  Running TPE sampler...")
        tpe_study = run_study(
            "TPE",
            optuna.samplers.TPESampler(seed=42),
            objective_fn,
            args.n_trials,
            args.patience,
        )
        studies["TPE"] = tpe_study
        print(f"  TPE best: {tpe_study.best_value:.4f} in {len(tpe_study.trials)} trials")

        # GP sampler (optional — slow due to O(n³) surrogate fitting)
        if args.include_gp:
            print("  Running GP sampler...")
            gp_study = run_study(
                "GP",
                optuna.samplers.GPSampler(seed=42),
                objective_fn,
                args.n_trials,
                args.patience,
            )
            studies["GP"] = gp_study
            print(f"  GP best:  {gp_study.best_value:.4f} in {len(gp_study.trials)} trials")

        elapsed = time.time() - t0
        print(f"  Tuning time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Pick best study
        best_study_name = max(studies, key=lambda k: studies[k].best_value)
        best_study = studies[best_study_name]
        best_params = best_study.best_params
        # Add fixed params back
        best_params["objective"] = "multi:softprob"
        best_params["num_class"] = n_classes
        best_params["tree_method"] = "hist"

        print(f"\n  Best sampler: {best_study_name} (macro F1={best_study.best_value:.4f})")
        print(f"  Best params: {json.dumps(best_params, indent=4)}")

        # ── Multi-seed evaluation on TEST set ───────────────────────────
        # Retrain on train+val combined (HP selection is done)
        X_trainval = np.concatenate([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        print(f"\n  Evaluating on test set ({len(SEEDS)} seeds, trained on train+val)...")
        eval_results = evaluate_with_seeds(
            best_params, X_trainval, y_trainval, X_test, y_test, n_classes, le, SEEDS
        )

        print(f"  Test macro F1:    {eval_results['macro_f1_mean']:.4f} ± {eval_results['macro_f1_std']:.4f}")
        print(f"  Test weighted F1: {eval_results['weighted_f1_mean']:.4f} ± {eval_results['weighted_f1_std']:.4f}")
        print(f"  Test accuracy:    {eval_results['accuracy_mean']:.4f} ± {eval_results['accuracy_std']:.4f}")
        print(f"  Test balanced acc:{eval_results['balanced_accuracy_mean']:.4f} ± {eval_results['balanced_accuracy_std']:.4f}")

        # ── Plots ───────────────────────────────────────────────────────
        plot_optuna_history(studies, backbone, results_dir)
        print(f"  Saved optuna_history_{backbone}.png")

        plot_confusion_matrix(
            eval_results["confusion_matrix"],
            list(le.classes_),
            backbone,
            results_dir,
        )
        print(f"  Saved confusion_matrix_{backbone}.png")

        # ── Store results ───────────────────────────────────────────────
        gp_study = studies.get("GP")
        all_results[backbone] = {
            "best_sampler": best_study_name,
            "best_val_macro_f1": float(best_study.best_value),
            "best_params": best_params,
            "tpe_trials": len(tpe_study.trials),
            "tpe_best_val_f1": float(tpe_study.best_value),
            "gp_trials": len(gp_study.trials) if gp_study else None,
            "gp_best_val_f1": float(gp_study.best_value) if gp_study else None,
            "tuning_time_s": round(elapsed, 1),
            "test_results": eval_results,
        }

    # ── Save all results (merge with existing if running per-backbone) ──
    out_path = results_dir / "xgboost_results.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ── Summary table ───────────────────────────────────────────────────
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

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
