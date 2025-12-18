"""
Simple Tokenizer Implementation

Provides:
- Character-level tokenizer (for quick testing)
- BPE tokenizer wrapper (using tiktoken or sentencepiece)
"""

import json
import os
from typing import List, Optional


class CharTokenizer:
    """
    Simple character-level tokenizer.
    Useful for testing and debugging.
    """

    def __init__(self, chars: Optional[str] = None):
        if chars is None:
            # Default ASCII printable characters + newline
            chars = ''.join(chr(i) for i in range(32, 127)) + '\n\t'

        self.chars = sorted(list(set(chars)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.idx_to_char.get(t, '') for t in tokens)

    @classmethod
    def from_text(cls, text: str) -> 'CharTokenizer':
        """Create tokenizer from training text."""
        chars = sorted(list(set(text)))
        return cls(chars)

    def save(self, path: str):
        """Save tokenizer vocabulary."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'chars': self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """Load tokenizer vocabulary."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data['chars'])


class BPETokenizer:
    """
    Wrapper for tiktoken (OpenAI's BPE tokenizer).
    Falls back to sentencepiece if tiktoken is not available.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize BPE tokenizer.

        Args:
            encoding_name: tiktoken encoding name
                - "cl100k_base": GPT-4, ChatGPT encoding
                - "p50k_base": GPT-3 encoding
                - "r50k_base": Codex encoding
        """
        self.encoding_name = encoding_name
        self._tokenizer = None
        self._backend = None

    def _ensure_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(self.encoding_name)
            self._backend = "tiktoken"
            self.vocab_size = self._tokenizer.n_vocab
        except ImportError:
            try:
                import sentencepiece as spm
                # Use a default model if available
                self._tokenizer = spm.SentencePieceProcessor()
                # This would require a trained model file
                self._backend = "sentencepiece"
                raise NotImplementedError("SentencePiece requires a trained model file")
            except ImportError:
                raise ImportError(
                    "Neither tiktoken nor sentencepiece is installed.\n"
                    "tiktoken requires Rust compiler to build from source.\n\n"
                    "Options:\n"
                    "1. Install pre-built wheel: pip install tiktoken --only-binary :all:\n"
                    "2. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n"
                    "3. Use CharTokenizer instead (no external deps):\n"
                    "   from tokenizer import CharTokenizer\n"
                    "   tok = CharTokenizer.from_file('data/train.txt')"
                )

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        self._ensure_tokenizer()
        if self._backend == "tiktoken":
            return self._tokenizer.encode(text)
        else:
            return self._tokenizer.encode(text, out_type=int)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        self._ensure_tokenizer()
        if self._backend == "tiktoken":
            return self._tokenizer.decode(tokens)
        else:
            return self._tokenizer.decode(tokens)


def get_tokenizer(name: str = "char", text: Optional[str] = None):
    """
    Get tokenizer by name.

    Args:
        name: Tokenizer type
            - "char": Character-level tokenizer
            - "bpe" or "tiktoken": BPE tokenizer (tiktoken)
        text: Training text for character tokenizer

    Returns:
        Tokenizer instance
    """
    if name == "char":
        if text is not None:
            return CharTokenizer.from_text(text)
        return CharTokenizer()
    elif name in ("bpe", "tiktoken"):
        return BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


if __name__ == "__main__":
    # Test tokenizers
    print("Testing CharTokenizer:")
    text = "Hello, World! 你好世界"
    char_tok = CharTokenizer.from_text(text)
    print(f"  Vocab size: {char_tok.vocab_size}")
    encoded = char_tok.encode(text)
    print(f"  Encoded: {encoded[:20]}...")
    decoded = char_tok.decode(encoded)
    print(f"  Decoded: {decoded}")
    assert decoded == text

    print("\nTesting BPETokenizer:")
    try:
        bpe_tok = BPETokenizer()
        encoded = bpe_tok.encode(text)
        print(f"  Vocab size: {bpe_tok.vocab_size}")
        print(f"  Encoded: {encoded}")
        decoded = bpe_tok.decode(encoded)
        print(f"  Decoded: {decoded}")
    except ImportError as e:
        print(f"  Skipped (tiktoken not installed): {e}")
