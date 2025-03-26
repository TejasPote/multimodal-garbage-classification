#!/bin/bash

cd .. || exit

python train.py \
  --train_dir "garbage_data/Train" \
  --val_dir "garbage_data/Val" \
  --test_dir "garbage_data/Test" \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --weight_decay 1e-4 \
  --max_len 32 \
  --num_workers 4 \
  --checkpoint_dir "checkpoint"