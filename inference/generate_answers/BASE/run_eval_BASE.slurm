#!/bin/bash
#SBATCH -A NAIRR240352
#SBATCH --time=05:00:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH --mail-user=fl374@rutgers.edu
#SBATCH --mail-type=all

source /work/09308/zhengmk/python_vir_envs/vista/hpcgpt/bin/activate

cd /scratch/09979/esther_lh/hpcgpt/zmk/eval/BASE

export HF_HOME=/scratch/09979/esther_lh/cache/huggingface_cache

ml cuda/12.4
export CUDA_HOME=/opt/apps/cuda/12.4

export LD_LIBRARY_PATH=/opt/apps/gcc14/openmpi/5.0.5/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/comm_libs/12.3/hpcx/hpcx-2.17.1/hcoll/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/home1/09308/zhengmk/usr:/opt/apps/nvidia_math/12.4/targets/sbsa-linux/include:/opt/apps/cuda/12.4/targets/sbsa-linux/include:~/cutlass/include
export CXX=g++
export TORCH_CUDA_ARCH_LIST=9.0

export PATH=~:$PATH
export PATH=~/work/pdsh_vista/bin:$PATH

python3 /scratch/09979/esther_lh/hpcgpt/zmk/eval/BASE/eval_chatbot.py \
	--model_path meta-llama/Llama-3.1-8B-Instruct \
        --eval_json_path /scratch/09979/esther_lh/hpcgpt/zmk/data/eval.json \
        --output_file /scratch/09979/esther_lh/hpcgpt/zmk/eval/BASE/eval_results_chatbot.json \
        --max_new_tokens 1024 \
	--max_context_len 8192 \
