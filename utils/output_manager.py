# coding=utf-8
# Output directory management for QwenSpinQuant experiments

import os
from datetime import datetime
from pathlib import Path


def generate_run_id(base_dir: str, timestamp: str = None) -> str:
    """
    Generate a unique run ID with format: YYYYMMDD.HHMMSS.N
    where N is auto-incremented if multiple runs start in the same second.

    Args:
        base_dir: Base directory to check for existing runs
        timestamp: Optional timestamp string (YYYYMMDD.HHMMSS), auto-generated if None

    Returns:
        Unique run ID string
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")

    # Check for existing runs with same timestamp
    counter = 0
    while True:
        run_id = f"{timestamp}.{counter}"
        run_path = os.path.join(base_dir, run_id)
        if not os.path.exists(run_path):
            return run_id
        counter += 1


def get_output_dir(
    model_name: str,
    w_bits: int,
    a_bits: int,
    k_bits: int,
    v_bits: int,
    method: str,
    weight_quant: str = "GPTQ",
    base_output: str = "/data/disk1/guohaoran/QwenSpinQuant/output",
    timestamp: str = None,
) -> tuple[str, str]:
    """
    Generate structured output directory path.

    Structure:
        output/
        └── {model}_W{w}A{a}KV{kv}_{method}-{weight_quant}/
            └── {run_id}/
                ├── checkpoint-{step}/
                │   ├── R.bin (rotation matrices)
                │   ├── pytorch_model.bin
                │   ├── optimizer.pt
                │   └── ...
                ├── training.log
                └── eval.log

    Args:
        model_name: Model name (e.g., "Qwen2.5-0.5B")
        w_bits: Weight bits
        a_bits: Activation bits
        k_bits: Key bits
        v_bits: Value bits
        method: "SpinQuant" or "Baseline"
        weight_quant: "GPTQ" or "RTN"
        base_output: Base output directory
        timestamp: Optional timestamp for run_id

    Returns:
        (experiment_type_dir, run_dir) tuple
    """
    # Extract model short name (e.g., "Qwen2.5-0.5B" -> "Qwen2.5-0.5B")
    model_short = model_name.split("/")[-1]

    # Build experiment type string
    exp_type = f"{model_short}_W{w_bits}A{a_bits}K{k_bits}V{v_bits}_{method}-{weight_quant}"

    # Create experiment type directory
    exp_type_dir = os.path.join(base_output, exp_type)
    os.makedirs(exp_type_dir, exist_ok=True)

    # Generate unique run ID
    run_id = generate_run_id(exp_type_dir, timestamp)
    run_dir = os.path.join(exp_type_dir, run_id)

    return exp_type_dir, run_dir


def get_checkpoint_dir(run_dir: str, step: int) -> str:
    """
    Get checkpoint directory path for a specific step.

    Args:
        run_dir: Run directory path
        step: Training step number

    Returns:
        Checkpoint directory path
    """
    return os.path.join(run_dir, f"checkpoint-{step}")


def save_run_metadata(run_dir: str, args: dict):
    """
    Save run metadata (hyperparameters, config) to run directory.

    Args:
        run_dir: Run directory path
        args: Dictionary of arguments/config
    """
    import json

    os.makedirs(run_dir, exist_ok=True)
    metadata_path = os.path.join(run_dir, "run_metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(args, f, indent=2, default=str)


# Example usage:
if __name__ == "__main__":
    # Training run
    exp_dir, run_dir = get_output_dir(
        model_name="Qwen2.5-0.5B",
        w_bits=16,
        a_bits=8,
        k_bits=8,
        v_bits=8,
        method="SpinQuant",
        weight_quant="GPTQ",
    )
    print(f"Experiment dir: {exp_dir}")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint 100: {get_checkpoint_dir(run_dir, 100)}")

    # Evaluation run
    exp_dir, run_dir = get_output_dir(
        model_name="Qwen2.5-0.5B",
        w_bits=4,
        a_bits=8,
        k_bits=8,
        v_bits=8,
        method="Baseline",
        weight_quant="GPTQ",
    )
    print(f"\nEval experiment dir: {exp_dir}")
    print(f"Eval run dir: {run_dir}")
