#!/bin/bash

seed="$1"

swift sft \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --train_type lora \
    --dataset 'dataset/wikimusa_${seed}/llm_finetune/train.jsonl' \
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