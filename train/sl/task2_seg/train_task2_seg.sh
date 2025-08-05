#!/bin/bash

mkdir -p logs

nohup bash -c '{
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate pytorch_p38

    REPO_DIR=/home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/MIM-Med3D

    # Change to the code directory to ensure proper imports
    cd $REPO_DIR/code
    
    # Add current directory to PYTHONPATH as backup
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    
    # # Debug: print current directory and Python path
    # echo "Current directory: $(pwd)"
    # echo "Python path: $PYTHONPATH"
    # echo "Listing current directory:"
    # ls -la

    MAIN_FILE=experiments/sl/single_seg_main.py
    CONFIG_FILE=configs/sl/fomo_seg/unetr_base_vitmae_m0.5.yaml

    python3 $MAIN_FILE fit --config $CONFIG_FILE

    conda deactivate
}' > logs/train_task2_seg.log 2>&1 &
# > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &