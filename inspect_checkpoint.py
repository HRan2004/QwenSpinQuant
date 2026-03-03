#!/usr/bin/env python
# coding=utf-8
"""
Inspect Meta format checkpoint to see all weights, their keys, dtypes, and shapes.
"""

import argparse
import json
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to consolidated.00.pth file or directory containing it",
    )
    return parser.parse_args()


def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint and print all weight information."""
    checkpoint_path = Path(checkpoint_path)

    # If it's a directory, look for consolidated.00.pth
    if checkpoint_path.is_dir():
        pth_file = checkpoint_path / "consolidated.00.pth"
        params_file = checkpoint_path / "params.json"
    else:
        pth_file = checkpoint_path
        params_file = checkpoint_path.parent / "params.json"

    print(f"Loading checkpoint from {pth_file}...")
    print("="*80)

    # Load params if available
    if params_file.exists():
        with open(params_file, "r") as f:
            params = json.load(f)
        print("\nModel Parameters (from params.json):")
        print("-"*80)
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()

    # Load checkpoint
    checkpoint = torch.load(pth_file, map_location="cpu", weights_only=False)

    print(f"\nCheckpoint Type: {type(checkpoint)}")
    print(f"Number of keys: {len(checkpoint)}")
    print("\n" + "="*80)
    print("WEIGHT DETAILS")
    print("="*80)

    # Group keys by category
    categories = {
        "Embeddings": [],
        "Output/LM Head": [],
        "Norm": [],
        "Attention": [],
        "Feed Forward": [],
        "Other": []
    }

    for key in sorted(checkpoint.keys()):
        if "tok_embeddings" in key:
            categories["Embeddings"].append(key)
        elif "output" in key:
            categories["Output/LM Head"].append(key)
        elif "norm" in key and "layers" not in key:
            categories["Norm"].append(key)
        elif "attention" in key:
            categories["Attention"].append(key)
        elif "feed_forward" in key:
            categories["Feed Forward"].append(key)
        else:
            categories["Other"].append(key)

    # Print by category
    for category, keys in categories.items():
        if not keys:
            continue

        print(f"\n{'='*80}")
        print(f"{category.upper()} ({len(keys)} keys)")
        print('='*80)

        for key in keys:
            value = checkpoint[key]

            if isinstance(value, torch.Tensor):
                dtype = value.dtype
                shape = tuple(value.shape)
                numel = value.numel()

                # Calculate memory size
                element_size = value.element_size()
                memory_mb = (numel * element_size) / (1024 * 1024)

                # Get some statistics
                if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    min_val = value.min().item()
                    max_val = value.max().item()
                    mean_val = value.float().mean().item()
                    std_val = value.float().std().item()
                    stats = f"min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}"
                else:
                    stats = "N/A (not float type)"

                print(f"\n{key}")
                print(f"  Shape: {shape}")
                print(f"  Dtype: {dtype}")
                print(f"  Elements: {numel:,}")
                print(f"  Memory: {memory_mb:.2f} MB")
                print(f"  Stats: {stats}")
            else:
                print(f"\n{key}")
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_params = 0
    total_memory = 0
    dtype_counts = {}

    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            total_params += value.numel()
            total_memory += value.numel() * value.element_size()

            dtype_str = str(value.dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total memory: {total_memory / (1024**3):.2f} GB")
    print(f"\nData types distribution:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count} tensors")

    # Check for special keys (quantization-related)
    print("\n" + "="*80)
    print("SPECIAL KEYS (quantization-related)")
    print("="*80)

    quant_keys = [k for k in checkpoint.keys() if any(x in k for x in ['scale', 'zero', 'quant', 'int', 'bit'])]
    if quant_keys:
        print(f"\nFound {len(quant_keys)} quantization-related keys:")
        for key in sorted(quant_keys):
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.dtype}, {tuple(value.shape)}")
            else:
                print(f"  {key}: {type(value)}, {value}")
    else:
        print("\nNo obvious quantization-related keys found.")

    print("\n" + "="*80)


def main():
    args = parse_args()
    inspect_checkpoint(args.checkpoint_path)


if __name__ == "__main__":
    main()
