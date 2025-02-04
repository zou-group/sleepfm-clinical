#!/bin/bash

root_dir=/oak/stanford/groups/mignot/psg/NSRR/mesa
target_dir=/oak/stanford/groups/jamesz/rthapa84/psg_fm/data_new_128/mesa

num_threads=10
num_files=-1
resample_rate=128

python3 preprocessing.py \
    --root_dir $root_dir \
    --target_dir $target_dir \
    --num_threads $num_threads \
    --num_files $num_files \
    --resample_rate $resample_rate