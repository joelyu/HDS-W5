#!/bin/bash
#SBATCH --job-name=pbc_eval
#SBATCH --partition=icelake
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Evaluation: comparison tables, McNemar + bootstrap significance
# Requires: xgboost_results.json, linear_probe_results.json
#           (and optionally finetune_results.json)
#
# Usage:
#   sbatch hpc/05_evaluation.slurm

module purge
module load rhel8/default-icl

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== Evaluation ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

python -u scripts/05_evaluation.py \
    --results-dir results

echo ""
echo "Done: $(date)"
