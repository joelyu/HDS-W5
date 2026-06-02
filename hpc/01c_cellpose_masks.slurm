#!/bin/bash
#SBATCH --job-name=pbc_cellpose
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/cellpose_%j.out
#SBATCH --error=logs/cellpose_%j.err

# CellPose whole-cell segmentation masks (GPU accelerated)
# Produces masks for KU-Optofil and optionally Acevedo.
# Resumable — skips images whose mask PNG already exists.
#
# Usage:
#   sbatch hpc/01c_cellpose_masks.slurm              # KU-Optofil only
#   sbatch hpc/01c_cellpose_masks.slurm --acevedo    # also Acevedo masks

module purge
module load rhel8/default-amp
module load cuda/12.1

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

WITH_ACEVEDO=0
for arg in "$@"; do
  case "$arg" in
    --acevedo) WITH_ACEVEDO=1 ;;
  esac
done

echo "=== CellPose Mask Generation ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

echo "--- KU-Optofil masks ---"
python -u scripts/02d_cellpose_masks.py \
    --data-dir data/raw \
    --results-dir results

if [[ "$WITH_ACEVEDO" -eq 1 ]]; then
    echo ""
    echo "--- Acevedo masks ---"
    python -u scripts/02d_cellpose_masks.py \
        --acevedo-dir data/acevedo/PBC_dataset_normal_DIB \
        --results-dir results
fi

echo ""
echo "Done: $(date)"
