#!/bin/bash
#SBATCH -A NAIRR240352
#SBATCH --time=05:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -p gh
#SBATCH --mail-user=fl374@rutgers.edu
#SBATCH --mail-type=all

cd /scratch/09979/esther_lh/hpcgpt/zmk/metrics/relevance/run_relevance_1

source /work/09979/esther_lh/vista/python_venvs/hpcgpt_eval/bin/activate

module load cuda/12.4

METHOD="SFT+RAG+ICL"

INPUT_FILE="/scratch/09979/esther_lh/hpcgpt/zmk/eval/SFT_results/${METHOD}_answer.json"
OUTPUT_DIR="./relevance_scores_v3/${METHOD}"

echo "Running evaluation on $INPUT_FILE"

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

srun python3 relevance.py --input "$INPUT_FILE" --output "$OUTPUT_DIR" 
