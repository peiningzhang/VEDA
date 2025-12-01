#!/bin/bash
# Example script for using the trained QM9 model
# Model: /home/phz24002/Drug_Discovery/VEDA/wandb/equinv-qm9/p5gilxmq/checkpoints/last.ckpt
# Dataset: /home/phz24002/Drug_Discovery/semla-flow/data/qm9

MODEL_PATH="/home/phz24002/Drug_Discovery/VEDA/wandb/equinv-qm9/p5gilxmq/checkpoints/last.ckpt"
DATA_PATH="/home/phz24002/Drug_Discovery/semla-flow/data/qm9/smol"

# Example 1: Evaluate the model on test set
echo "=== Evaluating model on QM9 test set ==="
python evaluate.py \
  --ckpt_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --dataset qm9 \
  --dataset_split test \
  --n_molecules 1000 \
  --n_replicates 1 \
  --integration_steps 100 \
  --coord_noise_std_dev 0.4

# # Example 2: Generate new molecules
# echo "=== Generating new molecules ==="
# python predict.py \
#   --ckpt_path "$MODEL_PATH" \
#   --data_path "$DATA_PATH" \
#   --dataset qm9 \
#   --n_molecules 1000 \
#   --integration_steps 50 \
#   --save_file predictions_qm9.smol \
#   --save_dir ./outputs \
#   --mask_rate_strategy log_uniform

