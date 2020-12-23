#!/bin/bash

python run.py \
    --data_root /home/gx/datasets/coco/test2017/ \
    --mask_root /home/gx/datasets/mask/test_mask_datset/ \
    --model_path checkpoint/g_480000.pth \
    --test \
    --target_size 384 \
    --mask_mode 1