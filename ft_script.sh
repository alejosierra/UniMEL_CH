#!/bin/bash

set -euo pipefail

seed="${1:-}"

if [[ -z "${seed}" ]]; then
    echo "Usage: bash ft_script.sh <seed>"
    exit 1
fi

dataset_path="dataset/wikimusa/wikimusa_${seed}/llm_finetune/train.jsonl"
echo "Training dataset path: ${dataset_path}"

python code/wikimusa_prepare_ft.py --config ./config/wikimusa.yaml --seed "${seed}"

swift sft \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --train_type lora \
    --dataset "${dataset_path}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --output_dir "output_ft/swift_ft/logs/seed_${seed}" \
    --warmup_ratio 0.03 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --seed "${seed}" \
    --use_hf True