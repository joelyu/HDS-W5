#!/bin/bash
#SBATCH --job-name=pbc_xgboost
#SBATCH --partition=icelake
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/xgboost_%j.out
#SBATCH --error=logs/xgboost_%j.err

# XGBoost HP tuning on CPU (icelake — 32 cores)
# TPE only — GP sampler dropped (surrogate fitting dominated walltime)
#
# Usage:
#   sbatch hpc/02_xgboost_training.slurm              # all backbones
#   sbatch hpc/02_xgboost_training.slurm vit_s16       # single backbone

module purge
module load rhel8/default-icl

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

BACKBONE_ARG=""
if [[ -n "$1" ]]; then
    BACKBONE_ARG="--backbone $1"
fi

echo "=== XGBoost Training (TPE) ${1:-all backbones} ==="
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Start: $(date)"
echo ""

python -u scripts/03_xgboost_training.py \
    --results-dir results \
    --n-trials 100 \
    --patience 20 \
    $BACKBONE_ARG

echo ""
echo "Done: $(date)"
