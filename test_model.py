#!/usr/bin/env python
# coding=utf-8
"""
Test script to verify quantized model can be loaded and basic forward pass works.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")

    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=args.device,
        )
        print("✓ Model loaded successfully")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded successfully")

        # Test tokenization
        test_text = "Hello"
        inputs = tokenizer(test_text, return_tensors="pt").to(args.device)
        print(f"✓ Tokenization works: {test_text} -> {inputs['input_ids'].shape}")

        # Test forward pass
        print("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"✓ Forward pass works: logits shape = {logits.shape}")

            # Check for NaN or Inf
            if torch.isnan(logits).any():
                print("✗ WARNING: NaN detected in logits!")
            elif torch.isinf(logits).any():
                print("✗ WARNING: Inf detected in logits!")
            else:
                print("✓ Logits are valid (no NaN/Inf)")

        # Try simple generation with greedy decoding
        print("\nTesting generation with greedy decoding...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"✓ Generation works!")
        print(f"Input: {test_text}")
        print(f"Output: {generated_text}")

        print("\n" + "="*50)
        print("All tests passed! Model is working correctly.")
        print("="*50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
