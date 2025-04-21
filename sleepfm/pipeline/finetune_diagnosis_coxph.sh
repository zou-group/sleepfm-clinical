#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

config_path="../configs/config_finetune_diagnosis_coxph.yaml"

python finetune_diagnosis_coxph.py --config_path $config_path