#!/bin/bash

# TensorBoard Monitoring Script for MIM-Med3D Training
# This script launches TensorBoard to monitor the training progress

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch_p38

# Configuration
REPO_DIR=/home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/MIM-Med3D
LOG_DIR=/home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/lightning/logs
EXPERIMENT_NAME=fomo60k
VERSION=vitsimmim_base_p16_m0.3_full_shallow 
PORT=6006

# Launch TensorBoard
tensorboard --logdir=$LOG_DIR --port=$PORT

## print connection info
# echo "TensorBoard is running at http://localhost:$PORT"
