#!/bin/bash
# Wrapper script for training with structured output directories
# Usage: bash scripts/train_with_output.sh <model_path> <w_bits> <a_bits> <k_bits> <v_bits> <method> <weight_quant>

set -e

MODEL_PATH=$1
W_BITS=$2
A_BITS=$3
K_BITS=$4
V_BITS=$5
METHOD=${6:-"SpinQuant"}  # SpinQuant or Baseline
WEIGHT_QUANT=${7:-"GPTQ"}  # GPTQ or RTN

# Extract model name from path
MODEL_NAME=$(basename $MODEL_PATH)

# Generate output directory
TIMESTAMP=$(date +"%Y%m%d.%H%M%S")
EXP_TYPE="${MODEL_NAME}_W${W_BITS}A${A_BITS}K${K_BITS}V${V_BITS}_${METHOD}-${WEIGHT_QUANT}"
BASE_OUTPUT="/data/disk1/guohaoran/QwenSpinQuant/output"
RUN_DIR="${BASE_OUTPUT}/${EXP_TYPE}/${TIMESTAMP}.0"

# Check for conflicts and increment counter if needed
COUNTER=0
while [ -d "${BASE_OUTPUT}/${EXP_TYPE}/${TIMESTAMP}.${COUNTER}" ]; do
    COUNTER=$((COUNTER + 1))
done
RUN_DIR="${BASE_OUTPUT}/${EXP_TYPE}/${TIMESTAMP}.${COUNTER}"

mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Experiment: $EXP_TYPE"
echo "Run ID: ${TIMESTAMP}.${COUNTER}"
echo "Output: $RUN_DIR"
echo "=========================================="

# Save run metadata
cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "w_bits": $W_BITS,
  "a_bits": $A_BITS,
  "k_bits": $K_BITS,
  "v_bits": $V_BITS,
  "method": "$METHOD",
  "weight_quant": "$WEIGHT_QUANT",
  "timestamp": "$TIMESTAMP",
  "run_id": "${TIMESTAMP}.${COUNTER}",
  "command": "$0 $@"
}
EOF

# Set environment
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6

cd /data/disk1/guohaoran/QwenSpinQuant
source .venv/bin/activate

# Determine rotation path based on method
if [ "$METHOD" = "SpinQuant" ]; then
    ROTATION_PATH="${RUN_DIR}/rotation"
    mkdir -p "$ROTATION_PATH"
    ROTATE_FLAG="--rotate"
else
    ROTATION_PATH=""
    ROTATE_FLAG=""
fi

# Training: optimize rotation matrices (W16 for GPTQ)
TRAIN_W_BITS=$W_BITS
if [ "$WEIGHT_QUANT" = "GPTQ" ] && [ $W_BITS -lt 16 ]; then
    TRAIN_W_BITS=16
    echo "Using W16 for rotation training (GPTQ mode)"
fi

if [ "$METHOD" = "SpinQuant" ]; then
    echo "Starting rotation training..."
    torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
        --input_model "$MODEL_PATH" \
        --output_rotation_path "$ROTATION_PATH" \
        --output_dir "$RUN_DIR" \
        --logging_dir "$RUN_DIR/logs" \
        --model_max_length 2048 \
        --fp16 False \
        --bf16 True \
        --log_on_each_node False \
        --per_device_train_batch_size 1 \
        --logging_steps 10 \
        --learning_rate 1.5 \
        --weight_decay 0. \
        --lr_scheduler_type cosine \
        --gradient_checkpointing True \
        --save_safetensors False \
        --max_steps 100 \
        --w_bits $TRAIN_W_BITS \
        --a_bits $A_BITS \
        --k_bits $K_BITS \
        --v_bits $V_BITS \
        --w_clip \
        --a_asym \
        --k_asym \
        --v_asym \
        --k_groupsize 64 \
        --v_groupsize 64 \
        2>&1 | tee "$RUN_DIR/training.log"

    echo "Rotation training completed. R.bin saved to $ROTATION_PATH"
fi

echo "=========================================="
echo "Training completed!"
echo "Output directory: $RUN_DIR"
echo "=========================================="
