#!/usr/bin/env python
# coding=utf-8
"""
Inference script for Meta format LLaMA models (consolidated.00.pth).
"""

import argparse
import json
import torch
from pathlib import Path
from sentencepiece import SentencePieceProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Meta Format Model Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
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
    return parser.parse_args()


class SimpleLLaMAInference:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        model_path = Path(model_path)

        # Load checkpoint
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(
            model_path / "consolidated.00.pth",
            map_location=device,
            weights_only=False
        )

        # Load params
        with open(model_path / "params.json", "r") as f:
            self.params = json.load(f)

        print(f"Model params: {self.params}")

        # Load tokenizer
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.load(str(model_path / "tokenizer.model"))

        self.checkpoint = checkpoint
        print("Model loaded successfully!")

    def encode(self, text):
        """Encode text to token ids."""
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        """Decode token ids to text."""
        return self.tokenizer.decode(tokens)

    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text from prompt."""
        print(f"\nPrompt: {prompt}")
        print(f"Generating with max_length={max_length}, temperature={temperature}, top_p={top_p}")

        # Encode prompt
        tokens = self.encode(prompt)
        print(f"Encoded tokens: {tokens[:10]}... (length: {len(tokens)})")

        # For now, just return the prompt + a simple message
        # Full generation would require implementing the model forward pass
        return prompt + " [Note: This is a placeholder. Full generation requires implementing the model architecture.]"


def main():
    args = parse_args()

    # Initialize model
    model = SimpleLLaMAInference(args.model_path)

    print("\n" + "="*50)
    print("Meta Format LLaMA Inference")
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
            response = model.generate(
                user_input,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
