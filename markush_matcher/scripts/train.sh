#!/bin/bash

python train_group_pred.py \
    --base_model_name "laituan245/molt5-large" \
    --checkpoint_dir $root \
    --dataset_path data/train.json \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
;