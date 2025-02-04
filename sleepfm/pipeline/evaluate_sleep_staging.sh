#!/bin/bash

model_path="../checkpoints/SetTransformer/leave_one_out_128_patch_size_3840/SleepEventLSTMClassifier_mesa_sleep_stages_BAS_RESP_EKG_EMG"
dataset="mesa"
split="test"

python evaluate_sleep_staging.py \
    --output_path $model_path \
    --split $split \
    --dataset $dataset