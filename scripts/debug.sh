#!/bin/bash

export HF_HOME=/your/hf/home/

VISION_MODEL=/your/vqsiglip/path

AR_BACKBONE=Qwen/Qwen3-0.6B
DIFFUSION=Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers

LR=5e-5
RUN_NAME="debug"

echo "AR_BACKBONE: ${AR_BACKBONE}"
echo "DIFFUSION: ${DIFFUSION}"
echo "RUN_NAME: ${RUN_NAME}"

LOCAL_DIR="models/${RUN_NAME}"


torchrun \
--nproc_per_node=4 \
--nnodes=1 \
--master_port=29509 \
blip3o/train/train.py \
--deepspeed scripts/zero1.json \
--num_image_tokens 65536 \
--num_scale_tokens 3 \
--load_embeddings_from_vision True \
--model_name_or_path $AR_BACKBONE \
--diffusion_name_or_path   $DIFFUSION \
--version "qwen_1_5" \
--dataset_cls 'mix' \
--dispatch_batches False \
--vision_tower ${VISION_MODEL} \
--mm_vision_select_layer -2 \
--mm_use_im_start_end True \
--group_by_modality_length True \
--image_aspect_ratio square \
--mm_patch_merge_type flat \
--bf16 True \
--run_name $RUN_NAME \
--output_dir ${LOCAL_DIR} \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 1 \
--learning_rate ${LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 5 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none \
--torch_compile True \
--torch_compile_backend inductor \
--dataloader_drop_last True 
