#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16   			    # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=4		    # Specify tasks per node
#SBATCH --gpus=4	                # Specify total number of GPUs
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

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python main.py fit \
  -c configs/dinov3/coco/instance/eomt_small_640.yaml \
  --compile_disabled \
  --trainer.devices 4 \
  --trainer.num_nodes 1 \
  --data.num_workers 2 \
  --data.batch_size 32 \
  --data.path /project/lt200377-mpind/segment