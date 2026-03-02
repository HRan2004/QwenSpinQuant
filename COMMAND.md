# Command Reference

## General Setup

```bash
cd /data/disk1/guohaoran/QwenSpinQuant && source .venv/bin/activate && export PATH=/usr/local/cuda-12.6/bin:$PATH && export CUDA_HOME=/usr/local/cuda-12.6
```

---

## Structured Output

所有命令使用新的输出管理脚本，自动创建结构化目录。详见 [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md)。

## Tasks

### W4A8K8V8 SpinQuant GPTQ

#### Evaluation

```bash
bash scripts/eval_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 4 8 8 8 SpinQuant GPTQ /data/disk1/guohaoran/QwenSpinQuant/output/rotation/R.bin
```

### W4A8K8V8 Baseline GPTQ

#### Evaluation

```bash
bash scripts/eval_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 4 8 8 8 Baseline GPTQ
```

### W4A4K4V4 SpinQuant GPTQ

#### Training

```bash
bash scripts/train_with_output.sh /data/disk1/guohaoran/models/Qwen2.5-0.5B 16 4 4 4 SpinQuant GPTQ
```

## Quick Reference

### 查看实验结果

```bash
# 列出所有实验
ls -d output/*/

# 查看最新的 W4A8KV8 SpinQuant 实验
ls -td output/Qwen2.5-0.5B_W4A8KV8_SpinQuant-GPTQ/*/ | head -1

# 提取所有 PPL 结果
find output -name "perplexity.txt" -exec echo {} \; -exec cat {} \;

# 查看特定实验的元数据
cat output/Qwen2.5-0.5B_W4A8KV8_SpinQuant-GPTQ/20260302.100000.0/run_metadata.json
```

### 脚本参数说明

#### train_with_output.sh

```bash
bash scripts/train_with_output.sh <model_path> <w_bits> <a_bits> <k_bits> <v_bits> <method> <weight_quant>
```

- `model_path`: 模型路径
- `w_bits`: 权重比特数（训练时 GPTQ 自动用 W16）
- `a_bits`: 激活比特数
- `k_bits`: Key 比特数
- `v_bits`: Value 比特数
- `method`: `SpinQuant` 或 `Baseline`
- `weight_quant`: `GPTQ` 或 `RTN`

#### eval_with_output.sh

```bash
bash scripts/eval_with_output.sh <model_path> <w_bits> <a_bits> <k_bits> <v_bits> <method> <weight_quant> [rotation_path]
```

- 前 7 个参数同 train_with_output.sh
- `rotation_path`: R.bin 路径（仅 SpinQuant 需要）
