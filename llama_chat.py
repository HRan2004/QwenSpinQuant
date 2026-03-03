#!/usr/bin/env python
# coding=utf-8
"""
Interactive inference with proper Llama 3.2 Instruct format.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def format_prompt(message):
    """Format message using Llama 3.2 Instruct template."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!\n")
    print("="*50)
    print("Llama 3.2 1B Instruct (Quantized)")
    print("Type 'quit' to exit")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            # Format with Llama 3.2 template
            prompt = format_prompt(user_input)

            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

            print("Assistant: ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode and extract only the assistant's response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract text after the assistant header
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            else:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(user_input):].strip()

            print(response + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


if __name__ == "__main__":
    main()
