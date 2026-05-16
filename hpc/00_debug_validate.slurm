#!/bin/bash
#SBATCH --job-name=pbc_debug
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU
#SBATCH -q INTR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

# DEBUG MODE: -q INTR gives near-top-of-queue priority, max 1 hour.
# Use this to validate the pipeline works on CSD3 before submitting full jobs.
#
# Runs: feature extraction (DinoBloom only, smallest) + quick XGBoost (5 trials, TPE only)
# Should complete in ~15-20 min on A100.

module purge
module load rhel8/default-amp
module load cuda/12.1

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs results

echo "=== DEBUG VALIDATION RUN ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

# Step 1: Extract features for DinoBloom only (384-dim, fastest)
echo "--- Feature Extraction (DinoBloom-S only) ---"
python -u scripts/02_feature_extraction.py \
    --data-dir data/raw \
    --results-dir results \
    --device cuda \
    --batch-size 128 \
    --backbone dinobloom_s

echo ""

# Step 2: Quick XGBoost validation (5 trials, TPE only, DinoBloom only)
echo "--- XGBoost Quick Validation (5 trials, TPE, DinoBloom) ---"
python -u scripts/03_xgboost_training.py \
    --results-dir results \
    --backbone dinobloom_s \
    --n-trials 5 \
    --patience 5 \
    --skip-gp

echo ""
echo "=== DEBUG DONE: $(date) ==="
echo "If this succeeded, submit the full jobs:"
echo "  sbatch hpc/01_feature_extraction.slurm"
echo "  sbatch hpc/02_xgboost_training.slurm"
