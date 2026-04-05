#!/bin/bash

set -e  # stop on error

echo "[SETUP INFO] Cloning PaDT repository..."
git clone https://github.com/Gorilla-Lab-SCUT/PaDT.git

echo "[SETUP INFO] Creating conda environment..."
conda create -y -n PaDT python=3.11

echo "[SETUP INFO] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PaDT

echo "[SETUP INFO] Installing dependencies..."
cd PaDT

echo "[SETUP INFO] Running PaDT setup..."
bash setup.sh

echo "[SETUP INFO] Setup complete. Activate with: conda activate PaDT"
