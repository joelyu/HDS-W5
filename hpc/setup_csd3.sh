#!/bin/bash
# Setup script for CSD3 — run once to create environment and transfer data
#
# Usage (from login node):
#   bash hpc/setup_csd3.sh
#
# Prerequisites:
#   1. Clone/copy repo to ~/rds/hpc-work/w5/HDS_W5_cyy36
#   2. Copy dataset.zip + metadata CSVs to data/raw/
#      (or run 00_download_data.py on login node)

set -e

WORK_DIR="$HOME/rds/hpc-work/w5/HDS_W5_cyy36"
DATA_DIR="$WORK_DIR/data/raw"

echo "=== CSD3 Setup for PBC Classification ==="
echo "Work dir: $WORK_DIR"
echo ""

# ── Create directory structure ────────────────────────────────────────────
mkdir -p "$DATA_DIR"
mkdir -p "$WORK_DIR/results"
mkdir -p "$WORK_DIR/logs"

# ── Create conda environment ─���───────────────────────────────────────────
module load miniconda/3

if conda env list | grep -q "classileukotion"; then
    echo "Environment 'classileukotion' already exists."
else
    echo "Creating conda environment from environment.yml..."
    conda env create -f "$WORK_DIR/environment.yml"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Transfer data to $DATA_DIR:"
echo "     scp data/raw/dataset.zip cyy36@login-e-16.hpc.cam.ac.uk:$DATA_DIR/"
echo "     scp data/raw/metadata*.csv cyy36@login-e-16.hpc.cam.ac.uk:$DATA_DIR/"
echo ""
echo "  2. Unzip on CSD3 (or run 00_download_data.py):"
echo "     cd $DATA_DIR && unzip dataset.zip"
echo ""
echo "  3. Submit jobs:"
echo "     cd $WORK_DIR"
echo "     sbatch hpc/01_feature_extraction.slurm"
echo "     # Wait for features, then:"
echo "     sbatch hpc/02_xgboost_training.slurm"
echo ""
echo "  4. Monitor:"
echo "     squeue -u cyy36"
echo "     tail -f logs/features_*.out"
