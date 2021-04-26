#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
    --data_root ../data/places2/data_large \
    --target_size 384 \
    --model_path places2_imsize384_recur10_linear_act_dingshengmod/weight.pth \
    --model_save_path places2_imsize384_recur10_linear_act_dingshengmod \
    --num_iters 3000000 \
    --batch_size 8 \
    --n_threads 20 \
    --multi_gpu \
    --finetune
