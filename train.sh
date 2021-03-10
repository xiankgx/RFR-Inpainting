#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python run.py \
    --data_root ../data/places2/data_large \
    --mask_root ../data/mask/testing_mask_dataset \
    --target_size 384 \
    --model_path places2_imsize384_recur10_sigmoid_act/g_40000.pth \
    --model_save_path places2_imsize384_recur10_sigmoid_act \
    --num_iters 1000000 \
    --batch_size 6 \
    --n_threads 12 \
    --multi_gpu
