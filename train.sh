#!/bin/bash

# Set experiment name and number of channels
EXPERIMENT="rgbham10000v1"
CHANNELS_NUM=3  # grey->1

# Split dataset using Python
echo "Splitting dataset..."
# python3 split_dataset.py

# Run training
echo "Starting training..."
python3 train.py  --DENOISER efficient_Unet \
                  --num-workers 6 \
                  --EXPERIMENT ${EXPERIMENT} \
                  --json-file configs/ADL_train.json \
                  --CHANNELS-NUM ${CHANNELS_NUM} \
                  --train-dirs 'data/train' \
                  --test-dirs  'data/test'
