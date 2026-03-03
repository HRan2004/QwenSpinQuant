#!/usr/bin/env python
# coding=utf-8
"""
Simple inference script for quantized LLaMA models.
Loads pre-quantized model directly without applying PTQ.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Model Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on"
    )
    return parser.parse_args()


def load_model(model_path, device):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, args):
    """Generate response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, args.device)

    print("\n" + "="*50)
    print("Simple Interactive Inference")
    print(f"Model: {args.model_path}")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")

    # Interactive loop
    while True:
        try:
            user_input = input("User: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Generate response
            response = generate_response(model, tokenizer, user_input, args)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
