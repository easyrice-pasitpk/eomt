#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16   			    # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1		    # Specify number of GPU per task
#SBATCH --ntasks-per-node=4		    # Specify tasks per node
#SBATCH -t 5:00:00                # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200377               	# Specify project name
#SBATCH -J eomt               	# Specify job name

# module load Mamba/23.11.0-0         # Load the conda module
# conda activate eomt		# Activate your environment
source .venv/bin/activate
export WANDB_MODE=offline
export WANDB_ENTITY="easyrice"
export HF_HUB_CACHE="/home/pjakkraw/.cache/huggingface/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python main.py fit \
  -c configs/dinov3/coco/instance/eomt_small_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 64 \
  --data.path /project/lt200377-mpind/segment