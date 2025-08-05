#!/bin/bash
#SBATCH --job-name=grpo    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:8                         # Number of GPUs per node
#SBATCH --time=96:00:00                      # Time limit (hh:mm:ss)


conda  activate  your env

export HF_HOME=/your/hf/home/

export WANDB_API_KEY='your_wandb_key'

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

srun --nodes=1 --ntasks=1  \
  accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero1.yaml \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_ip ${NODELIST[0]} \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    train_grpo.py
