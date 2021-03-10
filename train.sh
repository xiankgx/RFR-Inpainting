#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python run.py \
    --data_root ../datasets/places2/data_large \
    --target_size 256 \
    --model_path checkpoint_RFRv6/g_10000.pth \
    --model_save_path checkpoint_RFRv6 \
    --num_iters 1000000 \
    --batch_size 6 \
    --n_threads 16 \
    --multi_gpu