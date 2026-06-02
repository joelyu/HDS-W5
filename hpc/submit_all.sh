#!/bin/bash
#
# Submit the full pipeline to CSD3 with dependency chaining.
# Each stage waits for its prerequisites to complete successfully.
#
# Usage:
#   bash hpc/submit_all.sh                  # core pipeline only
#   bash hpc/submit_all.sh --with-external  # + Acevedo external validation
#   bash hpc/submit_all.sh --with-finetune  # + end-to-end fine-tuning
#   bash hpc/submit_all.sh --dry-run        # print sbatch commands without submitting
#   bash hpc/submit_all.sh --with-external --with-finetune   # flags compose
#
# Prerequisites:
#   1. Run hpc/setup_csd3.sh once
#   2. Data in data/raw/ (run 00_download_data.py)
#   3. (If --with-external) Acevedo in data/acevedo/ (run 00b_download_acevedo.py)

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
WITH_FINETUNE=0
WITH_EXTERNAL=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --with-finetune) WITH_FINETUNE=1 ;;
    --with-external) WITH_EXTERNAL=1 ;;
    -h|--help)       sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 1 ;;
  esac
done

submit() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] sbatch $*"
    echo "dry_$(echo "$1" | sed 's|.*/||;s|\.slurm||')"
  else
    sbatch --parsable "$@"
  fi
}

echo "=== PBC Classification Pipeline — CSD3 Submission ==="
echo ""

# ── Stage 1: Feature extraction (parallel, no dependencies) ──────────────
echo "--- Stage 1: Feature extraction ---"
jid_feat=$(submit hpc/01_feature_extraction.slurm)
echo "  Backbone features:    job $jid_feat"

jid_cp=$(submit hpc/01c_cellpose_masks.slurm)
echo "  CellPose masks:       job $jid_cp"

# ── Stage 2: Handcrafted features (after CellPose masks) ─────────────────
echo "--- Stage 2: Handcrafted features ---"
jid_hc=$(submit --dependency=afterok:$jid_cp hpc/01b_handcrafted_features.slurm)
echo "  Handcrafted features: job $jid_hc (after $jid_cp)"

# ── Stage 3: Classifiers (after all features) ────────────────────────────
echo "--- Stage 3: Classifiers ---"
jid_xgb=$(submit --dependency=afterok:$jid_feat:$jid_hc hpc/02_xgboost_training.slurm)
echo "  XGBoost training:     job $jid_xgb (after $jid_feat, $jid_hc)"

jid_lp=$(submit --dependency=afterok:$jid_feat:$jid_hc hpc/02b_linear_probe.slurm)
echo "  Linear probe:         job $jid_lp (after $jid_feat, $jid_hc)"

# ── Stage 4: Evaluation + explainability (after classifiers) ─────────────
echo "--- Stage 4: Evaluation + explainability ---"
jid_eval=$(submit --dependency=afterok:$jid_xgb:$jid_lp hpc/05_evaluation.slurm)
echo "  Evaluation:           job $jid_eval (after $jid_xgb, $jid_lp)"

jid_expl=$(submit --dependency=afterok:$jid_xgb hpc/06_explainability.slurm)
echo "  Explainability:       job $jid_expl (after $jid_xgb)"

# ── Optional: fine-tuning (after backbone features) ──────────────────────
if [[ "$WITH_FINETUNE" -eq 1 ]]; then
  echo "--- Optional: Fine-tuning ---"
  for bb in resnet50 efficientnet_b0; do
    jid_ft=$(submit --dependency=afterok:$jid_feat hpc/03_fine_tuning.slurm "$bb")
    echo "  Fine-tune $bb: job $jid_ft (after $jid_feat)"
  done
fi

# ── Optional: external validation (after training + Acevedo masks) ───────
if [[ "$WITH_EXTERNAL" -eq 1 ]]; then
  echo "--- Optional: External validation ---"
  jid_cp_ace=$(submit --dependency=afterok:$jid_cp hpc/01c_cellpose_masks.slurm --acevedo)
  echo "  CellPose Acevedo:     job $jid_cp_ace (after $jid_cp)"

  jid_ext=$(submit --dependency=afterok:$jid_xgb:$jid_lp:$jid_cp_ace hpc/04_external_validation.slurm)
  echo "  External validation:  job $jid_ext (after $jid_xgb, $jid_lp, $jid_cp_ace)"
fi

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs will appear in logs/"
