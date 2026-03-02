# Experiment Log

## Model: Qwen2.5-0.5B

### Experiment 1: W16A8KV8 Rotation Training (100 steps)

**Training config:**
- Script: `scripts/run_optimize_rotation.sh`
- GPU: 1x (single node)
- Batch size: 1
- Learning rate: 1.5 (cosine schedule)
- Max steps: 100
- Quantization: W16A8KV8 (asymmetric, k/v groupsize=64)
- Training time: 6m43s (4.05s/it)
- Final epoch: 0.08
- Train loss: 16.81

**PTQ Evaluation (WikiText2 perplexity):**

| Config | PPL |
|--------|-----|
| FP16 (no quantization) | 14.25 |
| W16A8KV8 (no rotation) | 14.58 |
| W16A8KV8 + SpinQuant (100 steps) | 14.48 |

**Analysis:**
- Quantization degradation: +0.33 ppl (14.25 → 14.58)
- SpinQuant recovery: -0.10 ppl (14.58 → 14.48)
- Recovery rate: 30% of quantization loss recovered

---

### Experiment 2: W16A8KV8 Rotation Training (2000 steps, 20x)

**Training config:**
- Same as Experiment 1, except:
- Max steps: 2000
- Logging steps: 10
- Training time: 2h15m10s (4.04s/it)
- Final epoch: 1.63
- Train loss: 21.45 (higher than 100-step run)

**PTQ Evaluation (WikiText2 perplexity):**

| Config | PPL |
|--------|-----|
| FP16 (no quantization) | 14.25 |
| W16A8KV8 (no rotation) | 14.58 |
| W16A8KV8 + SpinQuant (100 steps) | 14.48 |
| W16A8KV8 + SpinQuant (2000 steps) | 14.48 |

**Analysis:**
- 20x more training steps yielded no improvement (14.482 → 14.479)
- Train loss increased from 16.8 to 21.4 over longer training, suggesting overfitting or instability
- W16A8KV8 quantization degradation is too small (0.33 ppl) for rotation to make a significant difference
- 100 steps is sufficient for this configuration

---

## Summary

| Config | Steps | Train Loss | PPL | Δ vs FP16 | Δ vs No Rotation |
|--------|-------|-----------|-----|-----------|-----------------|
| FP16 | - | - | 14.25 | - | - |
| W16A8KV8 | - | - | 14.58 | +0.33 | - |
| W16A8KV8 + SpinQuant | 100 | 16.81 | 14.48 | +0.23 | -0.10 |
| W16A8KV8 + SpinQuant | 2000 | 21.45 | 14.48 | +0.23 | -0.10 |

## TODO

- [ ] Test with W4A8 quantization (more aggressive, larger degradation expected)
- [ ] Test with W4A4KV4 quantization (SpinQuant's target scenario)
- [ ] Try multi-GPU training (8x) to increase effective batch size
