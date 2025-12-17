"""
Text Generation Script

Load a trained model and generate text interactively or from prompts.
"""

import argparse
import torch

from model import GPT, ModelConfig
from tokenizer import CharTokenizer, BPETokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model config
    model_config = ModelConfig(**checkpoint["model_config"])

    # Create and load model
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model with {model.count_parameters():,} parameters")

    return model, model_config


def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda"
) -> str:
    """Generate text from a prompt."""
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    # Decode
    generated_tokens = output_ids[0].tolist()
    text = tokenizer.decode(generated_tokens)

    return text


def interactive_mode(model, tokenizer, args):
    """Run interactive generation loop."""
    print("\n" + "="*50)
    print("Interactive text generation")
    print("Type a prompt and press Enter to generate")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break

            if not prompt:
                continue

            print("\nGenerating...\n")
            text = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )

            print("-" * 40)
            print(text)
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained model")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="char",
                        choices=["char", "bpe"],
                        help="Tokenizer type")
    parser.add_argument("--vocab-path", type=str, default=None,
                        help="Path to vocabulary file (for char tokenizer)")

    # Generation
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (interactive mode if not provided)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling parameter")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load model
    model, model_config = load_model(args.checkpoint, args.device)

    # Load tokenizer
    if args.tokenizer == "char":
        if args.vocab_path:
            tokenizer = CharTokenizer.load(args.vocab_path)
        else:
            # Try to find vocab file next to checkpoint
            import os
            vocab_path = args.checkpoint.replace('.pt', '_vocab.json')
            if os.path.exists(vocab_path):
                tokenizer = CharTokenizer.load(vocab_path)
            else:
                # Default character set
                tokenizer = CharTokenizer()
        print(f"Using char tokenizer with vocab size: {tokenizer.vocab_size}")
    else:
        tokenizer = BPETokenizer()
        print(f"Using BPE tokenizer with vocab size: {tokenizer.vocab_size}")

    # Generate
    if args.prompt:
        # Single generation
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        print("\nGenerated text:")
        print("-" * 40)
        print(text)
        print("-" * 40)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()
