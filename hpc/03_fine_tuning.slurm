#!/bin/bash
#SBATCH --job-name=pbc_finetune
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# Fine-tuning on A100 (ampere partition)
# Submit per-backbone for parallel execution + fault isolation:
#   sbatch hpc/03_fine_tuning.slurm resnet50
#   sbatch hpc/03_fine_tuning.slurm efficientnet_b0
#   sbatch hpc/03_fine_tuning.slurm dinobloom_s
#
# Debug mode:
#   sbatch -q INTR --time=01:00:00 hpc/03_fine_tuning.slurm --debug

BACKBONE="${1:?Usage: sbatch hpc/03_fine_tuning.slurm <backbone|--debug>}"

module purge
module load rhel8/default-amp
module load cuda/12.1

export PATH="$HOME/miniforge3/envs/classileukotion/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
mkdir -p logs

echo "=== Fine-Tuning: ${BACKBONE} ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo ""

if [[ "$BACKBONE" == "--debug" ]]; then
    echo "[DEBUG MODE] resnet50, 2 trials, 3 epochs"
    python -u scripts/04_fine_tune.py \
        --data-dir data/raw \
        --results-dir results \
        --backbone resnet50 \
        --n-trials 2 \
        --n-epochs 3 \
        --patience 2 \
        --batch-size 32 \
        --device cuda
else
    python -u scripts/04_fine_tune.py \
        --data-dir data/raw \
        --results-dir results \
        --backbone "$BACKBONE" \
        --n-trials 30 \
        --n-epochs 50 \
        --patience 10 \
        --batch-size 32 \
        --device cuda
fi

echo ""
echo "Done: $(date)"
