#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python run.py \
    --data_root ../data/places2/data_large \
    --mask_root ../data/mask/testing_mask_dataset \
    --target_size 384 \
    --model_path checkpoint/g_610000.pth \
    --num_iters 1000000 \
    --batch_size 9 \
    --n_threads 12 \
    --multi_gpu
