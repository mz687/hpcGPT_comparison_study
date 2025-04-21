#!/bin/bash
#SBATCH -A A-ccsc
#SBATCH --time=00:30:00
#SBATCH -o rtx-%J.out
#SBATCH -e rtx-%J.out
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -p gh
#SBATCH --mail-user=YOUR@EMAIL.ADDRESS
#SBATCH --mail-type=all

GPUS_PER_NODE=1
NODE_LIST=$(srun hostname)
NODE_LIST_UNIQUE=$(echo "${NODE_LIST[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ')

host_name=$(hostname)

export GPUS_PER_NODE=$GPUS_PER_NODE
export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NNODES * $GPUS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$NODE_LIST_UNIQUE" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT=${MASTER_PORT}"

NODE_LIST_UNIQUE=$(echo "${NODE_LIST[@]}" | tr ' ' '\n' | sort -u )
echo $NODE_LIST_UNIQUE

HOSTFILE_DIR=./
echo "$NODE_LIST_UNIQUE" > $HOSTFILE_DIR/hostfile

NODE_LIST_UNIQUE=$(cat $HOSTFILE_DIR/hostfile | while read line; do echo ${line}" slots=${GPUS_PER_NODE}"; done)
echo "$NODE_LIST_UNIQUE" > $HOSTFILE_DIR/hostfile

NODE_RANK=$(( $(echo $NODE_LIST_UNIQUE | grep  -n -F -w $host_name | cut -d ":" -f 1) -1 ))
echo "NODE_RANK=${NODE_RANK}"

huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama3.1_8b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT


deepspeed --launcher PDSH --hostfile $HOSTFILE_DIR/hostfile  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT main_ckpt.py \
   --data_path Dahoas/full-hh-rlhf local/jsonfile  \
   --data_split 10,0,0 \
   --model_name_or_path meta-llama/Llama-3.1-8B  \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-5 \
   --weight_decay 0. \
   --num_train_epochs 8  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --dtype bf16 \
   --print_loss \
   --output_dir /path/to/where/the/model/checkpoints/will/be/saved \
   --compute_fp32_loss \
   --offload \
   |& tee $OUTPUT/training.log