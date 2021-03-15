#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python run.py \
    --data_root ../datasets/places2/data_large \
    --target_size 256 \
    --model_path checkpoint_RFRv7/g_155000.pth \
    --model_save_path checkpoint_RFRv7 \
    --num_iters 1000000 \
    --batch_size 16 \
    --n_threads 16 \
    --multi_gpu \
    --fp16