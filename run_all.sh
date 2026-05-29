#!/usr/bin/env bash
#
# Run the full PBC-classification pipeline end-to-end, in order.
#
# WARNING: a full run is ~1 DAY on a workstation (CellPose ~12h, XGBoost tuning
# ~5h, plus feature extraction). Every stage is idempotent / resumable, so
# re-running fast-forwards through completed stages (00 skips on md5 match, 02b
# skips if its npz exists, 02d resumes, 03/03b merge). Heavy/optional arms
# (fine-tuning, external validation) are behind flags.
#
# Usage:
#   bash run_all.sh                  # core pipeline: data -> features -> classifiers -> eval
#   bash run_all.sh --dry-run        # print every command in order WITHOUT running it
#   bash run_all.sh --with-external  # also: download Acevedo + CellPose masks + external validation
#   bash run_all.sh --with-finetune  # also: end-to-end fine-tuning (needs a GPU; very slow)
#   bash run_all.sh --with-external --with-finetune --dry-run   # flags compose
#
# Run from anywhere — the script cd's to its own directory (the repo root).
set -euo pipefail

cd "$(dirname "$0")"

DRY_RUN=0
WITH_FINETUNE=0
WITH_EXTERNAL=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --with-finetune) WITH_FINETUNE=1 ;;
    --with-external) WITH_EXTERNAL=1 ;;
    -h|--help)       sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 1 ;;
  esac
done

# run <cmd...> : echo the stage, then execute it (unless --dry-run).
run() {
  echo
  echo "=============================================================="
  echo ">>> $*"
  echo "=============================================================="
  [[ "$DRY_RUN" -eq 1 ]] || "$@"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] printing the pipeline; nothing will be executed."
fi

# ── 1. Data + EDA ────────────────────────────────────────────────────────────
run python -u scripts/00_download_data.py
run python -u scripts/01_data_exploration.py

# ── 2. Features (learned + handcrafted) ──────────────────────────────────────
run python -u scripts/02_feature_extraction.py            # 5 frozen backbones -> *_features.npz
run python -u scripts/02d_cellpose_masks.py               # CellPose whole-cell masks (cached, resumable)
run python -u scripts/02b_handcrafted_features.py                                   # convex-hull seg
run python -u scripts/02b_handcrafted_features.py --segmentation cellpose --force   # CellPose seg

# ── 3. Lightweight classifiers on frozen features ────────────────────────────
run python -u scripts/03_xgboost_training.py
run python -u scripts/03b_linear_probe.py
# Handcrafted 2x2 ablation: Tavakoli-51 subset (vs the default full 65)
run python -u scripts/03_xgboost_training.py --backbone handcrafted          --feature-set tavakoli
run python -u scripts/03_xgboost_training.py --backbone handcrafted_cellpose --feature-set tavakoli
run python -u scripts/03b_linear_probe.py    --backbone handcrafted          --feature-set tavakoli
run python -u scripts/03b_linear_probe.py    --backbone handcrafted_cellpose --feature-set tavakoli

# ── 4. Optional: end-to-end fine-tuning (GPU, very slow) ──────────────────────
if [[ "$WITH_FINETUNE" -eq 1 ]]; then
  run python -u scripts/04_fine_tune.py
fi

# ── 5. Evaluation + explainability ───────────────────────────────────────────
run python -u scripts/05_evaluation.py        # comparison tables + McNemar/bootstrap significance
run env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python -u scripts/06_explainability.py

# ── 6. Optional: external validation on Acevedo ──────────────────────────────
if [[ "$WITH_EXTERNAL" -eq 1 ]]; then
  run python -u scripts/00b_download_acevedo.py
  run python -u scripts/02d_cellpose_masks.py --acevedo-dir data/acevedo/PBC_dataset_normal_DIB
  run python -u scripts/07_external_validation.py
fi

echo
echo "Pipeline complete."
