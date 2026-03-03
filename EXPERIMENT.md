# Experiment Log

## Model: Qwen2.5-0.5B

| Config | W bits | A bits | KV bits | Method | Rotation Steps | Train Time | Train Loss | Final Epoch | PPL | Δ vs FP16 | Δ vs No Rotation | Notes |
|--------|--------|--------|---------|--------|----------------|------------|------------|-------------|-----|-----------|------------------|-------|
| FP16 baseline | 16 | 16 | 16 | - | - | - | - | - | 14.25 | - | - | No quantization |
| W16A8KV8 baseline | 16 | 8 | 8 | Asymmetric | - | - | - | - | 14.58 | +0.33 | - | No rotation, k/v groupsize=64 |
| W16A8KV8 + SpinQuant | 16 | 8 | 8 | Asymmetric | 100 | 6m43s | 16.81 | 0.08 | 14.48 | +0.23 | -0.10 | 30% recovery rate, 4.05s/it |
| W16A8KV8 + SpinQuant | 16 | 8 | 8 | Asymmetric | 2000 | 2h15m10s | 21.45 | 1.63 | 14.48 | +0.23 | -0.10 | No improvement vs 100 steps, possible overfitting |
| W4A8KV8 + SpinQuant + GPTQ | 4 | 8 | 8 | GPTQ | 100 | 6m43s | 16.81 | 0.08 | 15.60 | +1.35 | TBD | Reused W16A8KV8 rotation, need baseline |

## Training Config (Common)

- Script: `scripts/run_optimize_rotation.sh`
- GPU: 1x (single node)
- Batch size: 1
- Learning rate: 1.5 (cosine schedule)
- Logging steps: 1 (100 steps) / 10 (2000 steps)

## Key Findings

1. **100 steps is sufficient for W16A8KV8**: 20x more training (2000 steps) yielded no improvement, train loss increased suggesting overfitting
2. **W16A8KV8 quantization degradation is small**: Only +0.33 ppl, SpinQuant recovers 30% (-0.10 ppl)
3. **W4A8KV8 shows larger degradation**: +1.35 ppl vs FP16, need to test baseline without rotation to measure SpinQuant effectiveness

## TODO

- [ ] Test W4A8KV8 baseline (no rotation) to measure SpinQuant effectiveness
- [ ] Test W4A8KV8 + SpinQuant + RTN (compare with GPTQ)
- [ ] Test W4A4KV4 quantization (SpinQuant's target scenario)
- [ ] Try multi-GPU training (8x) to increase effective batch size
