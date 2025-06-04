#!/bin/bash

export HF_HOME=/your/HF/path/

torchrun --nproc_per_node=8 \
    train_image_reconstruction.py \
    --encoder_id google/siglip2-so400m-patch16-512 \
    --output_dir /your/output/folder/


