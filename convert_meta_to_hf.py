#!/usr/bin/env python
# coding=utf-8
"""
Convert Meta format LLaMA checkpoint to HuggingFace format.
Based on transformers convert_llama_weights_to_hf.py
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing consolidated.00.pth and params.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace format",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="1B",
        help="Model size (1B, 3B, 7B, etc.)",
    )
    return parser.parse_args()


def convert_meta_checkpoint_to_hf(input_dir, output_dir, model_size="1B"):
    """Convert Meta format checkpoint to HuggingFace format."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {input_dir}...")

    # Load params
    with open(input_dir / "params.json", "r") as f:
        params = json.load(f)

    print(f"Model params: {params}")

    # Calculate intermediate_size from checkpoint
    checkpoint = torch.load(
        input_dir / "consolidated.00.pth",
        map_location="cpu",
        weights_only=False
    )

    # Get actual intermediate_size from the checkpoint
    intermediate_size = checkpoint["layers.0.feed_forward.w1.weight"].shape[0]
    print(f"Detected intermediate_size: {intermediate_size}")

    # Create HuggingFace config
    config = LlamaConfig(
        vocab_size=params["vocab_size"],
        hidden_size=params["dim"],
        intermediate_size=intermediate_size,
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params.get("n_kv_heads", params["n_heads"]),
        rms_norm_eps=params.get("norm_eps", 1e-5),
        rope_theta=params.get("rope_theta", 10000.0),
        max_position_embeddings=params.get("max_seq_len", 2048),
    )

    # Initialize model with config
    print("Initializing model...")
    model = LlamaForCausalLM(config)

    # Convert state dict keys from Meta format to HuggingFace format
    print("Converting state dict...")
    hf_state_dict = {}

    for key, value in checkpoint.items():
        # Map Meta keys to HuggingFace keys
        new_key = key

        # tok_embeddings -> model.embed_tokens
        if key == "tok_embeddings.weight":
            new_key = "model.embed_tokens.weight"

        # output -> lm_head
        elif key == "output.weight":
            new_key = "lm_head.weight"

        # norm -> model.norm
        elif key == "norm.weight":
            new_key = "model.norm.weight"

        # layers.X -> model.layers.X
        elif key.startswith("layers."):
            new_key = "model." + key

            # attention.wq/wk/wv/wo -> self_attn.q_proj/k_proj/v_proj/o_proj
            new_key = new_key.replace("attention.wq", "self_attn.q_proj")
            new_key = new_key.replace("attention.wk", "self_attn.k_proj")
            new_key = new_key.replace("attention.wv", "self_attn.v_proj")
            new_key = new_key.replace("attention.wo", "self_attn.o_proj")

            # feed_forward -> mlp
            new_key = new_key.replace("feed_forward.w1", "mlp.gate_proj")
            new_key = new_key.replace("feed_forward.w2", "mlp.down_proj")
            new_key = new_key.replace("feed_forward.w3", "mlp.up_proj")

            # attention_norm -> input_layernorm
            new_key = new_key.replace("attention_norm", "input_layernorm")

            # ffn_norm -> post_attention_layernorm
            new_key = new_key.replace("ffn_norm", "post_attention_layernorm")

        hf_state_dict[new_key] = value
        print(f"  {key} -> {new_key}")

    # Load converted weights
    print("Loading converted weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    # Save model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Try to copy tokenizer if it exists in HF format
    print("Checking for tokenizer...")
    try:
        # Try to load from original Meta model path
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            token=None
        )
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved!")
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        print("You may need to manually add tokenizer files.")

    print(f"\nConversion complete! Model saved to {output_dir}")


def main():
    args = parse_args()
    convert_meta_checkpoint_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_size
    )


if __name__ == "__main__":
    main()
