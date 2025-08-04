#!/bin/bash
#SBATCH --job-name=sana15    # Job name
#SBATCH --nodes=4                         # Number of nodes
#SBATCH --gres=gpu:8                         # Number of GPUs per node
#SBATCH --time=96:00:00                      # Time limit (hh:mm:ss)
#SBATCH --output=/fsx/home/jiuhai.chen/Tar/log/%j.out         # Standard output log
#SBATCH --error=/fsx/home/jiuhai.chen/Tar/log/%j.err          # Standard error log
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --account=xgen-mm
#SBATCH --exclude=ip-10-3-20-53 


conda  activate  your env

export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'
export HF_HOME=/fsx/sfr/data/jiuhai


VISION_MODEL=/fsx/sfr/data/jiuhai/hub/models--csuhan--TA-Tok/snapshots/ac4dbb76a42d8b8bc92532b6fdb90c42e2d0a447/ta_tok.pth

AR_BACKBONE=Your_Pretrain_CKPT
DIFFUSION=Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers

LR=5e-5
RUN_NAME="SFT"

echo "AR_BACKBONE: ${AR_BACKBONE}"
echo "DIFFUSION: ${DIFFUSION}"
echo "RUN_NAME: ${RUN_NAME}"

LOCAL_DIR="models/${RUN_NAME}"


srun torchrun --nproc_per_node=8  --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29501 blip3o/train/train.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path $AR_BACKBONE \
    --diffusion_name_or_path  DIFFUSION \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True 
