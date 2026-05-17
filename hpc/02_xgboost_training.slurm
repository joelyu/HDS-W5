#!/bin/bash
#SBATCH --job-name=pbc_xgboost
#SBATCH --partition=icelake
#SBATCH --account=TORABI-SL3-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/xgboost_%j.out
#SBATCH --error=logs/xgboost_%j.err

# XGBoost HP tuning on CPU (icelake — 8 cores)
# TPE only — GP sampler dropped (surrogate fitting dominated walltime)

module purge
module load rhel8/default-icl

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== XGBoost Training (TPE) ==="
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Start: $(date)"
echo ""

python -u scripts/03_xgboost_training.py \
    --results-dir results \
    --n-trials 100 \
    --patience 20

echo ""
echo "Done: $(date)"
