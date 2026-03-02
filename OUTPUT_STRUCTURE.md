# Output Directory Structure

## Overview

所有实验输出统一管理在结构化目录中，避免冲突和混乱。

## Directory Structure

```
output/
└── {model}_{config}_{method}-{quant}/
    └── {timestamp}.{counter}/
        ├── run_metadata.json          # 运行元数据
        ├── training.log               # 训练日志（训练任务）
        ├── eval.log                   # 评估日志（评估任务）
        ├── perplexity.txt            # 最终 PPL 结果（评估任务）
        ├── checkpoint-{step}/         # 训练检查点
        │   ├── R.bin                  # 旋转矩阵（SpinQuant）
        │   ├── pytorch_model.bin      # 模型权重
        │   ├── optimizer.pt           # 优化器状态
        │   ├── scheduler.pt           # 学习率调度器
        │   ├── trainer_state.json     # Trainer 状态
        │   └── ...
        ├── rotation/                  # 旋转矩阵目录（训练任务）
        │   └── R.bin
        └── logs/                      # TensorBoard 日志
            └── events.out.tfevents.*
```

## Naming Convention

### Experiment Type (实验类型目录)

格式：`{model}_W{w}A{a}K{k}V{v}_{method}-{quant}`

- `{model}`: 模型名称，如 `Qwen2.5-0.5B`
- `W{w}`: 权重比特数，如 `W4`, `W16`
- `A{a}`: 激活比特数，如 `A4`, `A8`
- `K{k}`: Key 比特数，如 `K8`, `K4`
- `V{v}`: Value 比特数，如 `V8`, `V4`
- `{method}`: `SpinQuant` 或 `Baseline`
- `{quant}`: 权重量化方法，`GPTQ` 或 `RTN`

示例：
- `Qwen2.5-0.5B_W4A8K8V8_SpinQuant-GPTQ`
- `Qwen2.5-0.5B_W4A8K8V8_Baseline-GPTQ`
- `Qwen2.5-0.5B_W4A4K4V4_SpinQuant-RTN`

### Run ID (运行 ID)

格式：`{timestamp}.{counter}`

- `{timestamp}`: `YYYYMMDD.HHMMSS`，如 `20260302.085100`
- `{counter}`: 同一秒内启动的任务自增计数，从 `0` 开始

示例：
- `20260302.085100.0` — 第一个任务
- `20260302.085100.1` — 同一秒启动的第二个任务

## Usage

### 使用 Shell 脚本（推荐）

#### 训练

```bash
bash scripts/train_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 16 8 8 8 SpinQuant GPTQ
```

参数：
1. 模型路径
2. W bits
3. A bits
4. K bits
5. V bits
6. 方法（SpinQuant 或 Baseline）
7. 权重量化方法（GPTQ 或 RTN）

#### 评估

```bash
# SpinQuant 评估（需要指定 R.bin 路径）
bash scripts/eval_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 4 8 8 8 SpinQuant GPTQ /data/disk1/guohaoran/QwenSpinQuant/output/Qwen2.5-0.5B_W16A8K8V8_SpinQuant-GPTQ/20260302.085100.0/rotation/R.bin

# Baseline 评估（不需要 R.bin）
bash scripts/eval_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 4 8 8 8 Baseline GPTQ
```

参数：
1. 模型路径
2. W bits
3. A bits
4. K bits
5. V bits
6. 方法（SpinQuant 或 Baseline）
7. 权重量化方法（GPTQ 或 RTN）
8. 旋转矩阵路径（仅 SpinQuant 需要）

### 使用 Python API

```python
from utils.output_manager import get_output_dir, get_checkpoint_dir, save_run_metadata

# 生成输出目录
exp_dir, run_dir = get_output_dir(
    model_name="Qwen2.5-0.5B",
    w_bits=4,
    a_bits=8,
    k_bits=8,
    v_bits=8,
    method="SpinQuant",
    weight_quant="GPTQ",
)

# 保存元数据
save_run_metadata(run_dir, {
    "learning_rate": 1.5,
    "max_steps": 100,
    # ... 其他参数
})

# 获取检查点目录
ckpt_dir = get_checkpoint_dir(run_dir, step=100)
```

## Examples

### 完整实验流程

```bash
# 1. 训练旋转矩阵（W16A8K8V8）
bash scripts/train_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 16 8 8 8 SpinQuant GPTQ

# 输出：output/Qwen2.5-0.5B_W16A8K8V8_SpinQuant-GPTQ/20260302.090000.0/
# 旋转矩阵：output/Qwen2.5-0.5B_W16A8K8V8_SpinQuant-GPTQ/20260302.090000.0/rotation/R.bin

# 2. 评估 W4A8K8V8 + SpinQuant
bash scripts/eval_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 4 8 8 8 SpinQuant GPTQ output/Qwen2.5-0.5B_W16A8K8V8_SpinQuant-GPTQ/20260302.090000.0/rotation/R.bin

# 输出：output/Qwen2.5-0.5B_W4A8K8V8_SpinQuant-GPTQ/20260302.090500.0/
# PPL：output/Qwen2.5-0.5B_W4A8K8V8_SpinQuant-GPTQ/20260302.090500.0/perplexity.txt

# 3. 评估 W4A8K8V8 Baseline
bash scripts/eval_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 4 8 8 8 Baseline GPTQ

# 输出：output/Qwen2.5-0.5B_W4A8K8V8_Baseline-GPTQ/20260302.090600.0/
```

### 查找实验结果

```bash
# 列出所有 W4A8K8V8 实验
ls -d output/Qwen2.5-0.5B_W4A8K8V8_*/

# 查看最新的 SpinQuant 实验
ls -td output/Qwen2.5-0.5B_W4A8K8V8_SpinQuant-GPTQ/*/ | head -1

# 提取所有 PPL 结果
find output -name "perplexity.txt" -exec echo {} \; -exec cat {} \;

# 查看特定实验的元数据
cat output/Qwen2.5-0.5B_W4A8K8V8_SpinQuant-GPTQ/20260302.090500.0/run_metadata.json
```

## Migration from Old Structure

旧的输出文件（`output/eval_*.log`, `output/rotation/R.bin` 等）可以保留作为参考，新实验使用新的结构化目录。

如需迁移旧结果：

```bash
# 创建对应的实验类型目录
mkdir -p output/Qwen2.5-0.5B_W16A8K8V8_SpinQuant-GPTQ/20260301.220000.0

# 移动旧文件
mv output/rotation output/Qwen2.5-0.5B_W16A8K8V8_SpinQuant-GPTQ/20260301.220000.0/
mv output/rotation_training.log output/Qwen2.5-0.5B_W16A8K8V8_SpinQuant-GPTQ/20260301.220000.0/training.log
```

## Benefits

1. **避免冲突** — 每次运行有独立目录，timestamp + counter 确保唯一性
2. **易于追溯** — 目录名包含完整配置信息，一眼看出实验设置
3. **便于对比** — 同一配置的多次运行在同一实验类型目录下
4. **自动化友好** — 脚本自动管理目录结构，无需手动创建
5. **元数据完整** — `run_metadata.json` 记录所有参数，便于复现
