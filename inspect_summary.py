#!/usr/bin/env python
# coding=utf-8
"""
简洁地显示模型权重的关键信息
"""

import argparse
import json
import torch
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)

    if checkpoint_path.is_dir():
        pth_file = checkpoint_path / "consolidated.00.pth"
        params_file = checkpoint_path / "params.json"
    else:
        pth_file = checkpoint_path
        params_file = checkpoint_path.parent / "params.json"

    print(f"加载检查点: {pth_file}")
    print("="*80)

    # 加载参数
    with open(params_file, "r") as f:
        params = json.load(f)

    print("\n模型参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 加载检查点
    checkpoint = torch.load(pth_file, map_location="cpu", weights_only=False)

    print(f"\n检查点包含 {len(checkpoint)} 个键")
    print("\n"+"="*80)

    # 统计每一层的权重
    layer_stats = defaultdict(lambda: {"weight": [], "scales": [], "norm": []})

    for key in sorted(checkpoint.keys()):
        value = checkpoint[key]

        if key.startswith("layers."):
            # 提取层号
            parts = key.split(".")
            layer_num = parts[1]
            component = ".".join(parts[2:])

            if key.endswith(".weight") and value.dtype == torch.int8:
                layer_stats[layer_num]["weight"].append({
                    "name": component,
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "size_mb": value.numel() / (1024**2)
                })
            elif key.endswith(".scales"):
                layer_stats[layer_num]["scales"].append({
                    "name": component,
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype)
                })
            elif "norm" in key:
                layer_stats[layer_num]["norm"].append({
                    "name": component,
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype)
                })

    # 显示第0层的详细信息作为示例
    print("第 0 层详细信息 (示例):")
    print("-"*80)

    if "0" in layer_stats:
        print("\n权重 (INT8):")
        for w in layer_stats["0"]["weight"]:
            print(f"  {w['name']:40s} {str(w['shape']):25s} {w['dtype']:15s} {w['size_mb']:.2f} MB")

        print("\nScales (FP32):")
        for s in layer_stats["0"]["scales"]:
            print(f"  {s['name']:40s} {str(s['shape']):25s} {s['dtype']}")

        print("\nNorm (BF16):")
        for n in layer_stats["0"]["norm"]:
            print(f"  {n['name']:40s} {str(n['shape']):25s} {n['dtype']}")

    # 统计信息
    print("\n" + "="*80)
    print("统计信息:")
    print("-"*80)

    total_int8_params = 0
    total_fp32_params = 0
    total_bf16_params = 0

    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.int8:
                total_int8_params += value.numel()
            elif value.dtype == torch.float32:
                total_fp32_params += value.numel()
            elif value.dtype == torch.bfloat16:
                total_bf16_params += value.numel()

    print(f"\nINT8 参数: {total_int8_params:,} ({total_int8_params/(1024**3):.2f} GB)")
    print(f"FP32 参数: {total_fp32_params:,} ({total_fp32_params*4/(1024**3):.2f} GB)")
    print(f"BF16 参数: {total_bf16_params:,} ({total_bf16_params*2/(1024**3):.2f} GB)")
    print(f"总参数: {total_int8_params + total_fp32_params + total_bf16_params:,}")

    total_memory = (total_int8_params + total_fp32_params*4 + total_bf16_params*2) / (1024**3)
    print(f"总内存: {total_memory:.2f} GB")

    # 显示所有层的结构
    print("\n" + "="*80)
    print("所有层的结构:")
    print("-"*80)

    for layer_num in sorted(layer_stats.keys(), key=lambda x: int(x)):
        num_weights = len(layer_stats[layer_num]["weight"])
        num_scales = len(layer_stats[layer_num]["scales"])
        num_norms = len(layer_stats[layer_num]["norm"])
        print(f"Layer {layer_num}: {num_weights} 权重, {num_scales} scales, {num_norms} norms")

    # Embeddings 和 Output
    print("\n" + "="*80)
    print("Embeddings 和 Output:")
    print("-"*80)

    for key in ["tok_embeddings.weight", "tok_embeddings.scales", "output.weight", "output.scales", "norm.weight"]:
        if key in checkpoint:
            value = checkpoint[key]
            print(f"{key:30s} {str(tuple(value.shape)):25s} {str(value.dtype):15s} {value.numel()/(1024**2):.2f} MB")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
