#!/bin/bash

dataset_name=mesa
splits=train,validation,test
num_workers=16
model_path="../checkpoints//SetTransformer/leave_one_out_128_patch_size_3840"

python generate_embeddings.py \
    --num_workers $num_workers \
    --batch_size 16 \
    --model_path $model_path \
    --dataset_name $dataset_name \
    --splits $splits \
    --num_workers $num_workers