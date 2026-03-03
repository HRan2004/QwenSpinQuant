#!/usr/bin/env python
# coding=utf-8
"""
Inference script that properly handles INT8 quantized weights with per-channel scales.
This script dequantizes weights on-the-fly during inference.
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from sentencepiece import SentencePieceProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def dequantize_weight(weight_int8, scales, group_size=32):
    """
    Dequantize INT8 weight using per-channel scales.

    Args:
        weight_int8: INT8 weight tensor, shape (out_features, in_features)
        scales: FP32 scale tensor, shape (out_features, num_groups)
        group_size: number of elements per group

    Returns:
        Dequantized FP16 weight tensor
    """
    out_features, in_features = weight_int8.shape
    num_groups = scales.shape[1]

    # Reshape weight to (out_features, num_groups, group_size)
    weight_reshaped = weight_int8.view(out_features, num_groups, group_size)

    # Convert to float and apply scales
    # scales shape: (out_features, num_groups) -> (out_features, num_groups, 1)
    weight_fp = weight_reshaped.float() * scales.unsqueeze(-1)

    # Reshape back to (out_features, in_features)
    weight_fp = weight_fp.view(out_features, in_features)

    return weight_fp.half()


def load_and_dequantize_checkpoint(checkpoint_path, device="cuda"):
    """Load INT8 checkpoint and dequantize all weights."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        pth_file = checkpoint_path / "consolidated.00.pth"
        params_file = checkpoint_path / "params.json"
    else:
        pth_file = checkpoint_path
        params_file = checkpoint_path.parent / "params.json"

    print(f"Loading checkpoint from {pth_file}...")
    checkpoint = torch.load(pth_file, map_location="cpu", weights_only=False)

    with open(params_file, "r") as f:
        params = json.load(f)

    print(f"Model params: {params}")
    print(f"Dequantizing {len(checkpoint)} tensors...")

    # Dequantize all INT8 weights
    dequantized = {}
    group_size = params.get("quantization_args", {}).get("group_size", 32)

    for key, value in checkpoint.items():
        if key.endswith(".weight") and value.dtype == torch.int8:
            # Find corresponding scale
            scale_key = key.replace(".weight", ".scales")
            if scale_key in checkpoint:
                scales = checkpoint[scale_key]
                print(f"  Dequantizing {key}: {value.shape} INT8 -> FP16")
                dequantized[key] = dequantize_weight(value, scales, group_size)
            else:
                print(f"  Warning: No scales found for {key}, keeping as INT8")
                dequantized[key] = value
        elif key.endswith(".scales"):
            # Skip scales, they're already used
            continue
        else:
            # Keep other tensors as-is (norms, etc.)
            dequantized[key] = value.half() if value.dtype == torch.bfloat16 else value

    print(f"Dequantization complete. Moving to {device}...")

    # Move to device
    for key in dequantized:
        dequantized[key] = dequantized[key].to(device)

    return dequantized, params


class SimpleLLaMAModel:
    """Minimal LLaMA model for inference with dequantized weights."""

    def __init__(self, checkpoint, params, device="cuda"):
        self.checkpoint = checkpoint
        self.params = params
        self.device = device

        self.dim = params["dim"]
        self.n_layers = params["n_layers"]
        self.n_heads = params["n_heads"]
        self.n_kv_heads = params.get("n_kv_heads", self.n_heads)
        self.vocab_size = params["vocab_size"]
        self.head_dim = self.dim // self.n_heads

        print(f"Model: {self.n_layers} layers, {self.n_heads} heads, dim={self.dim}")

    def forward_single_token(self, token_id, cache=None):
        """
        Forward pass for a single token.
        This is a simplified implementation - full implementation would need
        proper attention, RoPE, etc.
        """
        # Get embedding
        x = self.checkpoint["tok_embeddings.weight"][token_id].unsqueeze(0)  # (1, dim)

        # For now, just return logits from output layer
        # A full implementation would process through all layers
        logits = torch.matmul(x, self.checkpoint["output.weight"].t())  # (1, vocab_size)

        return logits

    def generate(self, input_ids, max_new_tokens=50):
        """Generate tokens autoregressively."""
        generated = input_ids.tolist()

        for _ in range(max_new_tokens):
            # Get last token
            last_token = generated[-1]

            # Forward pass
            with torch.no_grad():
                logits = self.forward_single_token(last_token)

            # Greedy decoding
            next_token = logits.argmax(dim=-1).item()

            generated.append(next_token)

            # Stop if EOS
            if next_token == 128001:  # Llama 3 EOS token
                break

        return torch.tensor(generated)


def main():
    args = parse_args()

    # Load and dequantize checkpoint
    checkpoint, params = load_and_dequantize_checkpoint(args.model_path, args.device)

    # Load tokenizer
    tokenizer_path = Path(args.model_path) / "tokenizer.model"
    if not tokenizer_path.exists():
        print(f"Error: tokenizer not found at {tokenizer_path}")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))

    # Create model
    model = SimpleLLaMAModel(checkpoint, params, args.device)

    print("\n" + "="*50)
    print("INT8 Quantized LLaMA Inference")
    print(f"Model: {args.model_path}")
    print("Type 'quit' to exit")
    print("="*50 + "\n")

    print("Note: This is a simplified implementation.")
    print("Full inference requires proper layer-by-layer processing.\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            # Tokenize
            input_ids = tokenizer.encode(user_input)
            print(f"Tokens: {input_ids[:10]}... (length: {len(input_ids)})")

            # For demonstration, just show that dequantization worked
            print(f"\nDequantized weights loaded successfully!")
            print(f"Example weight stats:")
            print(f"  tok_embeddings: {checkpoint['tok_embeddings.weight'].shape}, {checkpoint['tok_embeddings.weight'].dtype}")
            print(f"  output: {checkpoint['output.weight'].shape}, {checkpoint['output.weight'].dtype}")
            print(f"  layer 0 wq: {checkpoint['layers.0.attention.wq.weight'].shape}, {checkpoint['layers.0.attention.wq.weight'].dtype}")

            print("\nNote: Full generation requires implementing the complete forward pass.")
            print("This would include: RoPE, attention, MLP, layer norms, etc.\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
