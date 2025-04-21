#!/bin/bash
#SBATCH -A A-ccsc
#SBATCH --time=04:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -p gh
#SBATCH --mail-user=YOUR@EMAIL.ADDRESS
#SBATCH --mail-type=all

GPUS_PER_NODE=1
NODE_LIST=$(srun hostname)
NODE_LIST_UNIQUE=$(echo "${NODE_LIST[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')

export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$NODE_LIST_UNIQUE" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT=${MASTER_PORT}"

srun python3 eval_chatbot.py \
    --model_path /path/to/the/SFT/model/path \
    --eval_json_path /path/to/data/eval.json \
    --output_file ./eval_answered_k_2_distributed.json \
    --max_new_tokens 1024 \
    --max_context_len 4096 \
    --RAG_file_path ./RAG_md \

srun python3 combine_all_rank_results.py