#!/bin/bash
#SBATCH --job-name=pbc_handcrafted
#SBATCH --partition=icelake
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/handcrafted_%j.out
#SBATCH --error=logs/handcrafted_%j.err

# Handcrafted feature extraction (CPU)
# Runs both segmentation approaches: convex hull + CellPose.
# CellPose masks must already exist (run 01c_cellpose_masks.slurm first).
#
# Usage:
#   sbatch hpc/01b_handcrafted_features.slurm

module purge
module load rhel8/default-icl

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== Handcrafted Feature Extraction ==="
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Start: $(date)"
echo ""

echo "--- Convex hull segmentation ---"
python -u scripts/02b_handcrafted_features.py \
    --data-dir data/raw \
    --results-dir results

echo ""
echo "--- CellPose segmentation ---"
python -u scripts/02b_handcrafted_features.py \
    --data-dir data/raw \
    --results-dir results \
    --segmentation cellpose \
    --force

echo ""
echo "Done: $(date)"
