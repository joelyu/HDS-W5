#!/bin/bash
#SBATCH --job-name=pbc_explain
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/explain_%j.out
#SBATCH --error=logs/explain_%j.err

# Explainability: UMAP, SHAP, DinoBloom attention maps
# GPU needed for DinoBloom attention map inference.
#
# Usage:
#   sbatch hpc/06_explainability.slurm

module purge
module load rhel8/default-amp
module load cuda/12.1

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== Explainability ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

python -u scripts/06_explainability.py \
    --data-dir data/raw \
    --results-dir results

echo ""
echo "Done: $(date)"
