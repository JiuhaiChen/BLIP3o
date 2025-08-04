#!/bin/bash
#SBATCH --job-name=grpo    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:8                         # Number of GPUs per node
#SBATCH --time=96:00:00                      # Time limit (hh:mm:ss)
#SBATCH --output=/fsx/home/jiuhai.chen/trl/log/%j.out         # Standard output log
#SBATCH --error=/fsx/home/jiuhai.chen/trl/log/%j.err          # Standard error log
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --account=xgen-mm


export CONDA_ROOT=/fsx/home/jiuhai.chen/envs/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda  activate  grpo_test

export HF_HOME=/fsx/sfr/data/jiuhai

export TRITON_CACHE_DIR=/fsx/sfr/data/jiuhai/triton-cache/
export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   /fsx/home/jiuhai.chen/envs/miniconda3/envs/grpo/bin/accelerate  launch   train_grpo.py



# Tell PyTorch where to rendezvous across nodes:
# export MASTER_PORT=29500
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Now launch across 4 machines, 8 GPUs each
srun --nodes=1 --ntasks=1  \
  /fsx/home/jiuhai.chen/envs/miniconda3/envs/grpo/bin/accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero1.yaml \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_ip ${NODELIST[0]} \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    train_grpo.py
