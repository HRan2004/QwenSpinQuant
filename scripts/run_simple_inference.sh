#!/bin/bash
# Simple inference script using pre-quantized model

MODEL_PATH="/data/disk1/guohaoran/models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8"

uv run simple_inference.py \
    --model_path ${MODEL_PATH} \
    --max_length 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --device cuda
