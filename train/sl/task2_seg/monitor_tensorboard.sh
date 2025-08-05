#!/bin/bash

# TensorBoard Monitoring Script for MIM-Med3D Training
# This script launches TensorBoard to monitor the training progress

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch_p38

# Configuration
EXPERIMENT_NAME=fomo_seg
# VERSION=vitsimmim_base_p16_m0.3_full_shallow 
LOG_DIR=/home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/lightning/logs/$EXPERIMENT_NAME/
PORT=6007

# Launch TensorBoard
tensorboard --logdir=$LOG_DIR --port=$PORT

## print connection info
# echo "TensorBoard is running at http://localhost:$PORT"
