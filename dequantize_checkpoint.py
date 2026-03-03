#!/usr/bin/env python
# coding=utf-8
"""
正确处理 INT4/INT8 混合量化的反量化脚本
"""

import argparse
import json
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def dequantize_weight(weight_int, scales, group_size=32):
    """
    反量化 INT4/INT8 权重

    Args:
        weight_int: INT8 容器中的量化权重 (out_features, in_features)
        scales: FP32 scale 张量
        group_size: 每组的元素数量

    Returns:
        反量化后的 FP16 权重
    """
    out_features, in_features = weight_int.shape

    # 处理不同的 scale 形状
    if scales.dim() == 2 and scales.shape[1] > 1:
        # Per-channel scales: (out_features, num_groups)
        num_groups = scales.shape[1]
        expected_group_size = in_features // num_groups

        # Reshape weight to (out_features, num_groups, group_size)
        weight_reshaped = weight_int.view(out_features, num_groups, expected_group_size)

        # 转换为 float 并应用 scales
        # scales shape: (out_features, num_groups) -> (out_features, num_groups, 1)
        weight_fp = weight_reshaped.float() * scales.unsqueeze(-1)

        # Reshape back
        weight_fp = weight_fp.view(out_features, in_features)

    elif scales.dim() == 1 or (scales.dim() == 2 and scales.shape[1] == 1):
        # Per-row scales: (out_features,) or (out_features, 1)
        if scales.dim() == 2:
            scales = scales.squeeze(-1)

        # 简单的 per-row scaling
        weight_fp = weight_int.float() * scales.unsqueeze(-1)

    else:
        raise ValueError(f"Unsupported scales shape: {scales.shape}")

    return weight_fp.half()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)

    if checkpoint_path.is_dir():
        pth_file = checkpoint_path / "consolidated.00.pth"
        params_file = checkpoint_path / "params.json"
    else:
        pth_file = checkpoint_path
        params_file = checkpoint_path.parent / "params.json"

    print(f"加载检查点: {pth_file}")
    checkpoint = torch.load(pth_file, map_location="cpu", weights_only=False)

    with open(params_file, "r") as f:
        params = json.load(f)

    print(f"模型参数: {params}")

    group_size = params.get("quantization_args", {}).get("group_size", 32)
    print(f"Group size: {group_size}")

    print(f"\n开始反量化...")
    print("="*80)

    dequantized = {}

    for key, value in checkpoint.items():
        if key.endswith(".weight") and value.dtype == torch.int8:
            # 查找对应的 scale
            scale_key = key.replace(".weight", ".scales")

            if scale_key in checkpoint:
                scales = checkpoint[scale_key]

                # 检查是否是 INT4 (只有16个唯一值)
                unique_vals = len(torch.unique(value))
                is_int4 = unique_vals <= 16

                quant_type = "INT4" if is_int4 else "INT8"

                print(f"反量化 {key}")
                print(f"  类型: {quant_type} (唯一值: {unique_vals})")
                print(f"  权重形状: {tuple(value.shape)}")
                print(f"  Scale形状: {tuple(scales.shape)}")

                # 反量化
                dequantized[key] = dequantize_weight(value, scales, group_size)

                print(f"  -> FP16 {tuple(dequantized[key].shape)}")
                print()
            else:
                print(f"警告: 没有找到 {key} 的 scales，保持原样")
                dequantized[key] = value

        elif key.endswith(".scales"):
            # 跳过 scales，已经使用过了
            continue

        else:
            # 保持其他张量不变 (norms 等)
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.bfloat16:
                    dequantized[key] = value.half()
                else:
                    dequantized[key] = value
            else:
                dequantized[key] = value

    print("="*80)
    print(f"反量化完成！")
    print(f"保存到: {output_path}")

    # 保存反量化后的检查点
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dequantized, output_path)

    # 也保存 params.json
    params_output = output_path.parent / "params.json"
    with open(params_output, "w") as f:
        json.dump(params, f, indent=2)

    print(f"参数文件保存到: {params_output}")

    # 统计信息
    total_params = sum(v.numel() for v in dequantized.values() if isinstance(v, torch.Tensor))
    total_memory = sum(v.numel() * v.element_size() for v in dequantized.values() if isinstance(v, torch.Tensor))

    print(f"\n统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  总内存: {total_memory / (1024**3):.2f} GB")
    print("\n完成！")


if __name__ == "__main__":
    main()
