#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# Activate the conda environment 
source activate psg_fm

model_paths=(
    /oak/stanford/groups/jamesz/rthapa84/psg_fm/08_26_24/models/SetTransformer/leave_one_out/test_run_20240828_161247/DiagnosisFinetuneFullLSTMCOXPH_ssc_stanford_phewas_tte_prediction_10_18_24_num_labels_1065__BAS_RESP_EKG_EMG__ep_20_bs_32/combined_dataset_split_10_14_28
)

splits=(
    "train" 
    "test"
)

for split in "${splits[@]}"; do
    for model_path in "${model_paths[@]}"; do
        echo "Running evaluation for model: $model_path on split: $split"
        python evaluate_disease_prediction.py \
            --output_path "$model_path" \
            --split "$split" \
            --dataset ssc_stanford
    done
done

