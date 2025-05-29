#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1


config_path="../configs/config_finetune_disease_prediction.yaml"

python finetune_disease_prediction.py --config_path $config_path