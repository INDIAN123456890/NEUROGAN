#!/bin/bash
# run_train.sh

set -e

conda activate synthetic_mri_gan

echo "Starting data preprocessing..."
python src/preprocess.py

echo "Starting model training..."
python src/train.py

echo "Training complete."