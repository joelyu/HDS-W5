#!/bin/bash
#SBATCH --job-name=pbc_external
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/external_%j.out
#SBATCH --error=logs/external_%j.err

# External validation on Acevedo dataset (GPU for backbone inference)
# Prerequisites:
#   1. Acevedo downloaded (00b_download_acevedo.py)
#   2. CellPose Acevedo masks generated (01c with --acevedo)
#   3. KU-Optofil XGBoost + linear probe training complete (02/03)
#
# Usage:
#   sbatch hpc/04_external_validation.slurm

module purge
module load rhel8/default-amp
module load cuda/12.1

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== External Validation (Acevedo) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

python -u scripts/07_external_validation.py \
    --acevedo-dir data/acevedo/PBC_dataset_normal_DIB \
    --results-dir results \
    --batch-size 64

echo ""
echo "Done: $(date)"
