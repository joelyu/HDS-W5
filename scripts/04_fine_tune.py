#!/usr/bin/env python3
"""End-to-end fine-tuning of pretrained backbones for PBC classification.

Fine-tunes ResNet-50, EfficientNet-B0, and DinoBloom-S with Optuna HP search,
then evaluates across multiple seeds. Complements the frozen-feature XGBoost
pipeline (script 03) to show the performance ceiling of end-to-end training.

Outputs saved to results/:
    finetune_results.json             — all metrics, best params, per-seed results
    finetune_{backbone}_best.pt       — best model checkpoint per backbone

Usage:
    python scripts/04_fine_tune.py
    python scripts/04_fine_tune.py --backbone dinobloom_s --n-trials 10
    python scripts/04_fine_tune.py --n-trials 1 --n-epochs 2 --device mps  # quick test
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

import numpy as np
import optuna
from scipy import stats
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

from config import (
    BACKBONES,
    CLASS_ORDER_ALPHA,
    FOLDER_NAME_MAP,
    NUM_CLASSES,
    SEEDS,
    SPLIT_ORDER,
    get_label_encoder,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)



# ── Transforms ─────────────────────────────────────────────────────────────


TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Dataset ────────────────────────────────────────────────────────────────


class PBCDataset(Dataset):
    """Dataset that loads images and returns (image, label_idx)."""

    def __init__(self, df: pd.DataFrame, image_root: Path, transform: T.Compose, le):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.le = le

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        class_folder = FOLDER_NAME_MAP[row["cell_type"]]
        original_split = row["path"].split("/")[0]
        img_path = self.image_root / original_split / class_folder / row["image_name"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(self.le.transform([row["cell_type"]])[0])
        return img, label


# ── Model building ─────────────────────────────────────────────────────────


def build_model(backbone_name: str, dropout: float = 0.3) -> nn.Module:
    """Load pretrained backbone and attach a classification head."""
    if backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, NUM_CLASSES))
        return model

    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, NUM_CLASSES))
        return model

    if backbone_name == "dinobloom_s":
        from huggingface_hub import hf_hub_download

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        ckpt_path = hf_hub_download(repo_id="MarrLab/DinoBloom", filename="pytorch_model_s.bin")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        num_tokens = int(1 + (224 / 14) ** 2)
        model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, 384))
        model.load_state_dict(ckpt, strict=True)
        # DINOv2's forward already applies self.head — replace the default Identity head
        model.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(384, NUM_CLASSES))
        return model

    raise ValueError(f"Unknown backbone: {backbone_name}")


def configure_freezing(model: nn.Module, backbone_name: str, unfreeze: str) -> None:
    """Freeze/unfreeze layers based on strategy."""
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    if unfreeze == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    if unfreeze == "last_block":
        if backbone_name == "resnet50":
            for p in model.layer4.parameters():
                p.requires_grad = True
        elif backbone_name == "efficientnet_b0":
            for p in model.features[-1].parameters():
                p.requires_grad = True
        elif backbone_name == "dinobloom_s":
            for p in model.blocks[-1].parameters():
                p.requires_grad = True

    # Always unfreeze the head
    if backbone_name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif backbone_name == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif backbone_name == "dinobloom_s":
        for p in model.head.parameters():
            p.requires_grad = True


# ── Training loop ──────────────────────────────────────────────────────────


class SubsetWrapper(Dataset):
    """Wrap a Subset to expose parent dataset attributes needed by training."""

    def __init__(self, subset, parent):
        self.subset = subset
        self.df = parent.df.iloc[subset.indices].reset_index(drop=True)
        self.le = parent.le
        self.image_root = parent.image_root
        self.transform = parent.transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


def compute_class_weights(dataset: PBCDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    labels = dataset.df["cell_type"].values
    le = dataset.le
    encoded = le.transform(labels)
    counts = np.bincount(encoded, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (macro_f1, all_preds, all_labels)."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return f1_score(all_labels, all_preds, average="macro"), all_preds, all_labels


def train_model(
    backbone_name: str,
    train_dataset: PBCDataset,
    val_dataset: PBCDataset,
    hp: dict,
    n_epochs: int,
    patience: int,
    device: torch.device,
    batch_size: int,
    report_fn=None,
) -> tuple[dict, float, list[dict]]:
    """Train a model with given HPs. Returns (best_model_state, best_val_f1, epoch_history)."""
    model = build_model(backbone_name, dropout=hp["dropout"])
    configure_freezing(model, backbone_name, hp["unfreeze_layers"])
    model.to(device)

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    nw = 0 if device.type == "mps" else 2
    pm = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pm
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_f1 = 0.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_f1, _, _ = evaluate(model, val_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": float(val_f1)})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if report_fn:
            report_fn(epoch, val_f1)

        if no_improve >= patience:
            break

    return best_state, best_f1, history


# ── Optuna ─────────────────────────────────────────────────────────────────


def run_optuna(
    backbone_name: str,
    train_dataset: PBCDataset,
    val_dataset: PBCDataset,
    n_trials: int,
    n_epochs: int,
    patience: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, float]:
    """Run Optuna HP search. Returns (best_params, best_val_f1)."""

    def objective(trial: optuna.Trial) -> float:
        hp = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "unfreeze_layers": trial.suggest_categorical(
                "unfreeze_layers", ["head_only", "last_block", "all"]
            ),
        }

        def report_fn(epoch, val_f1):
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, best_f1, _ = train_model(
            backbone_name, train_dataset, val_dataset, hp,
            n_epochs, patience, device, batch_size, report_fn,
        )
        return best_f1

    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=n_epochs, reduction_factor=3)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best trial: {study.best_value:.4f} (trial {study.best_trial.number})")
    print(f"  Best params: {study.best_params}")

    return study.best_params, study.best_value


# ── Multi-seed evaluation ──────────────────────────────────────────────────


def evaluate_multiseed(
    backbone_name: str,
    trainval_train_dataset: PBCDataset,
    trainval_eval_dataset: PBCDataset,
    test_dataset: PBCDataset,
    hp: dict,
    n_epochs: int,
    patience: int,
    device: torch.device,
    batch_size: int,
    seeds: list[int],
    results_dir: Path,
) -> dict:
    """Train with best HPs across seeds, evaluate on test. Returns results dict."""
    all_metrics = []
    all_preds_list = []
    all_histories = []
    best_f1_overall = 0.0
    best_state_overall = None

    le = trainval_train_dataset.le
    nw = 0 if device.type == "mps" else 2
    pm = device.type == "cuda"
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm
    )
    # Pre-compute test labels from dataframe (avoids iterating dataset)
    test_labels = le.transform(test_dataset.df["cell_type"].values)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Split trainval into train/val for early stopping
        n = len(trainval_train_dataset)
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val
        generator = torch.Generator().manual_seed(seed)
        train_sub, val_sub = torch.utils.data.random_split(
            trainval_train_dataset, [n_train, n_val], generator=generator
        )
        # Val subset uses eval transform (no augmentation) via trainval_eval_dataset
        val_sub_eval = torch.utils.data.Subset(
            trainval_eval_dataset, val_sub.indices
        )

        train_wrapped = SubsetWrapper(train_sub, trainval_train_dataset)
        val_wrapped = SubsetWrapper(val_sub_eval, trainval_eval_dataset)

        state, val_f1, seed_history = train_model(
            backbone_name, train_wrapped, val_wrapped, hp,
            n_epochs, patience, device, batch_size,
        )

        # Load best state and evaluate on test
        model = build_model(backbone_name, dropout=hp["dropout"])
        configure_freezing(model, backbone_name, hp["unfreeze_layers"])
        model.load_state_dict(state)
        model.to(device)

        test_f1, preds, labels = evaluate(model, test_loader, device)

        metrics = {
            "seed": seed,
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "weighted_f1": float(f1_score(labels, preds, average="weighted")),
            "accuracy": float(accuracy_score(labels, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        }
        all_metrics.append(metrics)
        all_preds_list.append(preds)
        all_histories.append(seed_history)
        print(f"    seed {seed}: macro_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1_overall:
            best_f1_overall = metrics["macro_f1"]
            best_state_overall = state

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save best checkpoint
    ckpt_path = results_dir / f"finetune_{backbone_name}_best.pt"
    torch.save(best_state_overall, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    # Aggregate
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

    # Classification report from median seed
    median_idx = int(np.argsort(macro_f1s)[len(macro_f1s) // 2])
    median_preds = all_preds_list[median_idx]

    summary["classification_report"] = classification_report(
        test_labels, median_preds, target_names=le.classes_, output_dict=True
    )
    summary["confusion_matrix"] = confusion_matrix(test_labels, median_preds).tolist()
    # Training history from median seed (for bias-variance tradeoff plots)
    summary["training_history"] = all_histories[median_idx]

    # Per-image predictions of the median-seed model + true labels, for paired
    # statistical comparison (McNemar / bootstrap) in 05. Label-encoded ints.
    summary["median_predictions"] = [int(x) for x in median_preds]
    summary["test_y_true"] = [int(x) for x in test_labels]

    return summary


# ── Main ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune backbones for PBC classification.")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument(
        "--backbone", choices=BACKBONES, default=None,
        help="Fine-tune a single backbone (default: all three).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    results_dir: Path = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load metadata
    meta_path = data_dir / "metadata_with_patient_level_splits.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.", file=sys.stderr)
        return 1

    df = pd.read_csv(meta_path)
    image_root = data_dir / "dataset"
    if not image_root.exists():
        print(f"ERROR: {image_root} not found.", file=sys.stderr)
        return 1

    le = get_label_encoder()
    print(f"Loaded metadata: {len(df)} images, {df['patient_id'].nunique()} patients")

    # Build datasets
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    train_dataset = PBCDataset(train_df, image_root, TRAIN_TRANSFORM, le)
    val_dataset = PBCDataset(val_df, image_root, EVAL_TRANSFORM, le)
    test_dataset = PBCDataset(test_df, image_root, EVAL_TRANSFORM, le)
    trainval_train_dataset = PBCDataset(trainval_df, image_root, TRAIN_TRANSFORM, le)
    trainval_eval_dataset = PBCDataset(trainval_df, image_root, EVAL_TRANSFORM, le)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()

    backbones_to_run = [args.backbone] if args.backbone else BACKBONES
    all_results = {}

    for backbone in backbones_to_run:
        print(f"\n{'='*60}")
        print(f"  Fine-tuning: {backbone}")
        print(f"{'='*60}")
        t0 = time.time()

        # Optuna HP search (train on train, validate on val)
        print("  Running Optuna HP search...")
        best_params, best_val_f1 = run_optuna(
            backbone, train_dataset, val_dataset,
            args.n_trials, args.n_epochs, args.patience,
            device, args.batch_size,
        )

        tuning_time = time.time() - t0
        print(f"  Tuning time: {tuning_time:.0f}s ({tuning_time / 60:.1f} min)")

        # Multi-seed evaluation on test (retrain on train+val)
        print(f"\n  Evaluating on test set ({len(SEEDS)} seeds, trained on train+val)...")
        eval_results = evaluate_multiseed(
            backbone, trainval_train_dataset, trainval_eval_dataset,
            test_dataset, best_params,
            args.n_epochs, args.patience, device, args.batch_size,
            SEEDS, results_dir,
        )

        print(f"  Test macro F1:    {eval_results['macro_f1_mean']:.4f} +/- {eval_results['macro_f1_std']:.4f}")
        print(f"  Test weighted F1: {eval_results['weighted_f1_mean']:.4f} +/- {eval_results['weighted_f1_std']:.4f}")
        print(f"  Test accuracy:    {eval_results['accuracy_mean']:.4f} +/- {eval_results['accuracy_std']:.4f}")

        all_results[backbone] = {
            "best_val_macro_f1": float(best_val_f1),
            "best_params": best_params,
            "tuning_time_s": round(tuning_time, 1),
            "test_results": eval_results,
        }

        # Free GPU
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # Save results (merge with existing if running per-backbone)
    out_path = results_dir / "finetune_results.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'Backbone':<20} {'Macro F1':>12} {'Weighted F1':>14} {'Accuracy':>12}")
    print(f"{'='*70}")
    for bb, res in all_results.items():
        tr = res["test_results"]
        print(
            f"  {bb:<20} "
            f"{tr['macro_f1_mean']:.4f}+/-{tr['macro_f1_std']:.4f} "
            f"{tr['weighted_f1_mean']:.4f}+/-{tr['weighted_f1_std']:.4f} "
            f"{tr['accuracy_mean']:.4f}+/-{tr['accuracy_std']:.4f}"
        )
    print(f"{'='*70}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
