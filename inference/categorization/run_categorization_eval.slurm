#!/bin/bash
#SBATCH -A A-ccsc
#SBATCH --time=02:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -p gh
#SBATCH --mail-user=YOUR@EMAIL.ADDRESS
#SBATCH --mail-type=all

# Define the number of GPUs per node
GPUS_PER_NODE=1
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

# Define the master port
export MASTER_PORT=12345
echo "MASTER_PORT=${MASTER_PORT}"

# Auto define the node list and the master address
NODE_LIST=$(srun hostname)
NODE_LIST_UNIQUE=$(echo "${NODE_LIST[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')
master_addr=$(scontrol show hostnames "$NODE_LIST_UNIQUE" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python3 main.py \
        --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --json_path /path/to/data/eval.json \
        --output_file ./eval_categorized.json \
        --max_new_tokens 1024 \
        -t 0.6 \