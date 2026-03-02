#!/bin/bash
# Optimize rotation for Qwen2.5-0.5B on single GPU
# Usage: bash scripts/run_optimize_rotation.sh

MODEL_PATH="/data/disk1/guohaoran/models/Qwen2.5-0.5B"
OUTPUT_DIR="/data/disk1/guohaoran/QwenSpinQuant/output"
ROTATION_DIR="/data/disk1/guohaoran/QwenSpinQuant/output/rotation"
LOG_DIR="/data/disk1/guohaoran/QwenSpinQuant/output/logs"

export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6

source /data/disk1/guohaoran/QwenSpinQuant/.venv/bin/activate

# W16A8KV8: only quantize activations for rotation optimization
# (when using GPTQ later, optimize with W16 first)
torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
--input_model $MODEL_PATH \
--output_rotation_path "$ROTATION_DIR" \
--output_dir "$OUTPUT_DIR/" \
--logging_dir "$LOG_DIR/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits 16 \
--a_bits 8 \
--k_bits 8 \
--v_bits 8 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 64 \
--v_groupsize 64
