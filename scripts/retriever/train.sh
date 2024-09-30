#!/bin/bash

echo "Running $PYTHON_TRAIN_SCRIPT"

DATA_DIR="data/retriever/dataset"
TRAIN_PATH="$DATA_DIR/train.jsonl"
DEV_PATH="$DATA_DIR/dev.jsonl"

python "zebra/retriever/train.py" \
    --train_data_path "$TRAIN_PATH" \
    --dev_data_path "$DEV_PATH" \
    --wandb_online_mode \