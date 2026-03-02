#!/bin/bash
# Wrapper script for evaluation with structured output directories
# Usage: bash scripts/eval_with_output.sh <model_path> <w_bits> <a_bits> <k_bits> <v_bits> <method> <weight_quant> [rotation_path]

set -e

MODEL_PATH=$1
W_BITS=$2
A_BITS=$3
K_BITS=$4
V_BITS=$5
METHOD=${6:-"Baseline"}  # SpinQuant or Baseline
WEIGHT_QUANT=${7:-"GPTQ"}  # GPTQ or RTN
ROTATION_PATH=$8  # Optional: path to R.bin for SpinQuant

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
  "rotation_path": "$ROTATION_PATH",
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

# Build rotation flags
ROTATE_FLAGS=""
if [ "$METHOD" = "SpinQuant" ]; then
    if [ -z "$ROTATION_PATH" ]; then
        echo "Error: SpinQuant method requires rotation_path argument"
        exit 1
    fi
    ROTATE_FLAGS="--rotate --optimized_rotation_path $ROTATION_PATH"
fi

# Build weight quantization flags
WEIGHT_FLAGS="--w_clip"
if [ "$WEIGHT_QUANT" = "RTN" ]; then
    WEIGHT_FLAGS="$WEIGHT_FLAGS --w_rtn"
fi

echo "Starting PTQ evaluation..."
torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model "$MODEL_PATH" \
    --do_train False \
    --do_eval True \
    --per_device_eval_batch_size 4 \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --save_safetensors False \
    --output_dir "$RUN_DIR" \
    --w_bits $W_BITS \
    --a_bits $A_BITS \
    --k_bits $K_BITS \
    --v_bits $V_BITS \
    $WEIGHT_FLAGS \
    --a_asym \
    --k_asym \
    --v_asym \
    --k_groupsize 64 \
    --v_groupsize 64 \
    $ROTATE_FLAGS \
    2>&1 | tee "$RUN_DIR/eval.log"

# Extract perplexity from log
PPL=$(grep "wiki2 ppl is:" "$RUN_DIR/eval.log" | tail -1 | awk '{print $NF}')
echo "$PPL" > "$RUN_DIR/perplexity.txt"

echo "=========================================="
echo "Evaluation completed!"
echo "Perplexity: $PPL"
echo "Output directory: $RUN_DIR"
echo "=========================================="
