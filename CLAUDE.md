# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

1. 对话时使用中文，代码及输出日志使用英文。
2. 单次写入大量代码至文件中失败时，分次，一次只写几行。

## rg 命令

用于连接 FDU_GZZ 远程服务器（10.176.56.244），定义在 `~/.bashrc` 中。

### 用法

- `rg` — 同步后登录服务器，工作目录为 `/data/disk1/guohaoran/transformers`
- `rg "command"` — 同步后在远程执行命令，如 `rg "python run.py"`

### 原理

每次调用 `rg` 时，先用 rsync 将本地 `/c/Projects/transformers` 增量同步到远程对应目录（排除 `.git`、`__pycache__`、`*.pyc`），然后通过 SSH 执行命令或登录。

### 实现

1. 从 `~/.ssh/config` 中读取 FDU_GZZ 的连接信息，参照已有的 `rt` 函数模板编写 `rg` 函数
2. 本地 Git Bash 没有 rsync，从 MSYS2 仓库下载了 rsync 3.3.0 包（`.pkg.tar.zst`），用 Python zstandard 解压，提取 `rsync.exe` 到 `~/bin/`
3. 补齐了缺失的依赖 `msys-xxhash-0.dll`（同样从 MSYS2 下载提取）

## Remote Server (FDU_GZZ)

- 地址：`guohaoran@10.176.56.244`
- 项目路径：`/data/disk1/guohaoran/QwenSpinQuant`
- 模型路径：`/data/disk1/guohaoran/models/`（当前有 Qwen2.5-0.5B）
- 输出路径：`/data/disk1/guohaoran/QwenSpinQuant/output/`
  - 旋转矩阵：`output/rotation/R.bin`
  - 训练日志：`output/rotation_training.log`
  - 评估日志：`output/eval_ptq.log`
- 本地同步：`rsync` 从 `C:\Projects\QwenSpinQuant` 同步到远程项目路径
- SSH 连接：`ssh guohaoran@10.176.56.244 'command'`（单引号避免本地 PATH 展开）

### 训练（旋转矩阵优化）

```bash
# 自定义单卡脚本（Qwen2.5-0.5B, W16A8KV8）
bash scripts/run_optimize_rotation.sh
```

入口：`optimize_rotation.py` → 调用 `train_utils/main.py:prepare_model()` 准备量化模型 → SGDG 优化器训练 R1/R2 → 保存 `R.bin`

### 评估（PTQ perplexity）

```bash
# 带旋转矩阵
torchrun --nproc_per_node=1 ptq.py --input_model <model> --rotate --optimized_rotation_path output/rotation/R.bin --w_bits 16 --a_bits 8 --k_bits 8 --v_bits 8 ...

# 无旋转 baseline
torchrun --nproc_per_node=1 ptq.py --input_model <model> --w_bits 16 --a_bits 8 --k_bits 8 --v_bits 8 ...

# FP16 baseline（无量化）
torchrun --nproc_per_node=1 ptq.py --input_model <model> --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 ...
```

入口：`ptq.py` → 调用 `eval_utils/main.py:ptq_model()` 应用旋转+量化 → `utils/eval_utils.py:evaluator()` 计算 WikiText2 perplexity

### 当前实验结果（Qwen2.5-0.5B, W16A8KV8）

| 配置 | WikiText2 PPL |
|------|--------------|
| FP16 原模型 | 14.25 |
| W16A8KV8 无旋转 | 14.58 |
| W16A8KV8 + SpinQuant | 14.48 |

## Project Overview

QwenSpinQuant implements SpinQuant, a learned-rotation-based LLM quantization technique (https://arxiv.org/pdf/2405.16406). It uses Cayley-optimized rotation matrices to remove outliers in LLMs, enabling W4A4KV4 quantization with minimal accuracy loss. Originally for LLaMA models, extended to support Qwen2/Qwen2.5 families.

## Running

There is no test suite. The two main workflows are:

**Step 1 — Optimize rotation matrices:**

```bash
# Single-node (7B/8B models): uses 8 GPUs via torchrun
bash scripts/10_optimize_rotation.sh <model_name> <w_bits> <a_bits> <kv_bits>
# Example: bash scripts/10_optimize_rotation.sh meta-llama/Llama-2-7b 4 4 4

# Multi-node FSDP (70B models):
bash scripts/11_optimize_rotation_fsdp.sh <model_name> <w_bits> <a_bits> <kv_bits>
```

**Step 2 — PTQ evaluation (perplexity on WikiText2):**

```bash
bash scripts/2_eval_ptq.sh <model_name> <w_bits> <a_bits> <kv_bits>
```

**ExecuTorch export:**

```bash
bash scripts/31_optimize_rotation_executorch.sh <model_name>
bash scripts/32_eval_ptq_executorch.sh <model_name>
```

Note: When using GPTQ for weight+activation quantization, optimize rotations with W16 first (only activations quantized), then evaluate with the target W bits. E.g., optimize with `16 4 4`, evaluate with `4 4 4`.

## Architecture

Two main entry points:

- `optimize_rotation.py` — Trains rotation matrices (R1 global, R2 per-head) using SGDG optimizer on the Stiefel manifold. Outputs `R.bin`.
- `ptq.py` — Applies rotations, fuses layer norms, quantizes weights (GPTQ or RTN), adds activation quantization wrappers, evaluates perplexity.

### Directory layout

- **`utils/`** — Shared core: quantization primitives (`quant_utils.py`), Hadamard transforms (`hadamard_utils.py`), argument parsing (`process_args.py`), data loading (`data_utils.py`), layer norm fusion (`fuse_norm_utils.py`), ExecuTorch export (`convert_to_executorch.py`).
- **`train_utils/`** — Rotation optimization: FSDP trainer (`fsdp_trainer.py`), SGDG optimizer for orthogonal matrices (`optimizer.py`), quantized model wrappers (`modeling_llama_quant.py`, `modeling_qwen2_quant.py`), custom quantized linear layer (`quant_linear.py`).
- **`eval_utils/`** — PTQ evaluation: layer-by-layer inference (`main.py`), GPTQ implementation (`gptq_utils.py`), rotation application (`rotation_utils.py`), model definitions (`modeling_llama.py`, `modeling_qwen2.py`).
- **`scripts/`** — Shell scripts that invoke `torchrun` with standard configurations.

### Key patterns

- `ActQuantWrapper` wraps `nn.Linear` layers to transparently quantize activations during forward pass.
- `RotateModule` wraps learnable rotation matrices, constrained to stay orthogonal via Cayley transform in the SGDG optimizer.
- Hadamard transforms are applied to down-projection inputs and Q/K for KV-cache quantization. Precomputed matrices live in `hadamard_utils.py` (large file, ~2MB of lookup tables).
- Layer norm weights are fused into adjacent linear layers before quantization (`fuse_norm_utils.py`).
- Model support is split: LLaMA models use `modeling_llama*.py`, Qwen2 models use `modeling_qwen2*.py`. Both follow the same quantization pipeline.

### Quantization configurations

Bit-widths for weights (`--w_bits`), activations (`--a_bits`), keys (`--k_bits`), values (`--v_bits`) can each be 4, 8, or 16. Common configs: W4A4KV4, W4A4KV16, W4A16KV16, W4A8. Qwen2/2.5 currently works best with W4A8 (W4A4 has channel outlier issues).

## Supported Models

LLaMA-2 (7B/13B/70B), LLaMA-3 (8B/70B), LLaMA-3.2 (1B/3B/8B), Qwen2, Qwen2.5 (0.5B through 72B).

## License

CC-BY-NC 4.0 (non-commercial use only).
