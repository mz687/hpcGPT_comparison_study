#!/bin/bash
#SBATCH -A A-ccsc
#SBATCH --time=14:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH --mail-user=YOUR@EMAIL.ADDRESS
#SBATCH --mail-type=all

python3 eval.py \
        --model_path /path/to/the/SFT/model/path \
        --eval_json_path /path/to/data/eval.json \
        --output_file ./eval_answered.json \
        --max_new_tokens 1024 \