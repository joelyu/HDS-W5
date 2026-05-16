#!/bin/bash
#SBATCH --job-name=pbc_features
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/features_%j.out
#SBATCH --error=logs/features_%j.err

# Feature extraction on GPU (ampere A100)
# Runs all 3 backbones — ~20 min total on A100 vs ~2 hrs on M1 Pro

module purge
module load rhel8/default-amp
module load cuda/12.1

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== Feature Extraction ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

python scripts/02_feature_extraction.py \
    --data-dir data/raw \
    --results-dir results \
    --device cuda \
    --batch-size 128

echo ""
echo "Done: $(date)"
