#!/usr/bin/env python
# coding=utf-8
"""
Simple inference script for SpinQuant quantized LLaMA models.
Supports interactive Q&A with quantized models.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from eval_utils.modeling_llama import LlamaForCausalLM
from eval_utils.main import ptq_model
from utils.process_args import PTQArgs


def parse_args():
    parser = argparse.ArgumentParser(description="SpinQuant Model Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the quantized model checkpoint",
    )
    parser.add_argument(
        "--w_bits", type=int, default=4, help="Weight quantization bits"
    )
    parser.add_argument(
        "--a_bits", type=int, default=8, help="Activation quantization bits"
    )
    parser.add_argument(
        "--k_bits", type=int, default=8, help="Key quantization bits"
    )
    parser.add_argument(
        "--v_bits", type=int, default=8, help="Value quantization bits"
    )
    parser.add_argument(
        "--rotate", action="store_true", help="Apply rotation matrices"
    )
    parser.add_argument(
        "--optimized_rotation_path",
        type=str,
        default=None,
        help="Path to optimized rotation matrices (R.bin)",
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


def load_model(args):
    """Load and quantize the model."""
    print(f"Loading model from {args.model_path}...")

    config = AutoConfig.from_pretrained(args.model_path)

    # Handle tied word embeddings
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    # Load model in FP16
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
    )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = model.to(args.device)

    # Create PTQ args
    ptq_args = PTQArgs(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        rotate=args.rotate,
        optimized_rotation_path=args.optimized_rotation_path,
    )

    # Apply quantization
    print("Applying quantization...")
    model = ptq_model(ptq_args, model, None, None)
    model.eval()

    print("Model loaded and quantized successfully!")
    return model


def load_tokenizer(model_path):
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
    model = load_model(args)
    tokenizer = load_tokenizer(args.model_path)

    print("\n" + "="*50)
    print("SpinQuant Interactive Inference")
    print(f"Model: {args.model_path}")
    print(f"Quantization: W{args.w_bits}A{args.a_bits}K{args.k_bits}V{args.v_bits}")
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
