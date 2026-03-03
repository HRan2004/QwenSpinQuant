#!/bin/bash
# Simple inference script for SpinQuant quantized models

MODEL_PATH="/data/disk1/guohaoran/models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"

uv run inference.py \
    --model_path ${MODEL_PATH} \
    --w_bits 4 \
    --a_bits 8 \
    --k_bits 8 \
    --v_bits 8 \
    --max_length 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --device cuda
