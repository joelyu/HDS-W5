#!/bin/bash
#SBATCH --job-name=pbc_linprobe
#SBATCH --partition=icelake
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/linprobe_%j.out
#SBATCH --error=logs/linprobe_%j.err

# Linear probe (logistic regression) on all frozen features
# Fast — typically completes in a few minutes.
#
# Usage:
#   sbatch hpc/02b_linear_probe.slurm

module purge
module load rhel8/default-icl

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== Linear Probe ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

python -u scripts/03b_linear_probe.py \
    --results-dir results

echo ""
echo "Done: $(date)"
