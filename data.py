"""
Data Preparation Utilities

Provides utilities to:
- Download sample datasets
- Tokenize and prepare data for training
- Create binary data files for efficient loading
"""

import os
import urllib.request
from typing import Optional

import numpy as np


def download_file(url: str, output_path: str, show_progress: bool = True):
    """Download a file with optional progress bar."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if show_progress:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()
    else:
        urllib.request.urlretrieve(url, output_path)


def download_shakespeare(output_dir: str = "data") -> str:
    """
    Download the tiny Shakespeare dataset.
    ~1MB of Shakespeare plays text.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = os.path.join(output_dir, "shakespeare.txt")

    if not os.path.exists(output_path):
        print(f"Downloading tiny Shakespeare to {output_path}...")
        download_file(url, output_path)
    else:
        print(f"Shakespeare dataset already exists at {output_path}")

    return output_path


def download_wikitext(output_dir: str = "data", subset: str = "2") -> str:
    """
    Download WikiText dataset.

    Args:
        output_dir: Output directory
        subset: "2" for WikiText-2, "103" for WikiText-103
    """
    # Note: This is a placeholder - actual WikiText requires proper download
    raise NotImplementedError(
        "WikiText download requires the datasets library. "
        "Install with: pip install datasets"
    )


def prepare_data(
    input_path: str,
    output_path: str,
    tokenizer=None,
    dtype=np.uint16
):
    """
    Tokenize text file and save as binary.

    Args:
        input_path: Path to input text file
        output_path: Path to output binary file
        tokenizer: Tokenizer instance (uses char-level if None)
        dtype: Data type for tokens (uint16 supports vocab up to 65k)
    """
    from tokenizer import CharTokenizer

    # Read text
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"  Total characters: {len(text):,}")

    # Tokenize
    print("Tokenizing...")
    if tokenizer is None:
        tokenizer = CharTokenizer.from_text(text)
        print(f"  Created char tokenizer with vocab size: {tokenizer.vocab_size}")

    tokens = tokenizer.encode(text)
    print(f"  Total tokens: {len(tokens):,}")

    # Convert to numpy and save
    tokens_np = np.array(tokens, dtype=dtype)
    print(f"  Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tokens_np.tofile(output_path)

    # Also save vocabulary if using char tokenizer
    if isinstance(tokenizer, CharTokenizer):
        vocab_path = output_path.replace('.bin', '_vocab.json')
        tokenizer.save(vocab_path)
        print(f"  Saved vocabulary to {vocab_path}")

    return output_path, tokenizer


def prepare_shakespeare(output_dir: str = "data"):
    """
    Download and prepare tiny Shakespeare dataset.
    Returns path to binary data file.
    """
    # Download
    text_path = download_shakespeare(output_dir)

    # Prepare
    bin_path = os.path.join(output_dir, "shakespeare.bin")
    prepare_data(text_path, bin_path)

    return bin_path


class StreamingTextDataset:
    """
    Memory-mapped dataset for large text files.
    Loads data lazily from disk.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        dtype=np.uint16
    ):
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=dtype, mode='r')
        self.n_tokens = len(self.data)

    def __len__(self):
        return self.n_tokens - self.seq_len - 1

    def __getitem__(self, idx):
        import torch
        x = torch.from_numpy(self.data[idx:idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + self.seq_len + 1].astype(np.int64))
        return x, y

    def get_batch(self, batch_size: int, device: str = 'cpu'):
        """Get a random batch."""
        import torch
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([self[i][0] for i in ix])
        y = torch.stack([self[i][1] for i in ix])
        return x.to(device), y.to(device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare"],
                        help="Dataset to prepare")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory")

    args = parser.parse_args()

    if args.dataset == "shakespeare":
        bin_path = prepare_shakespeare(args.output_dir)
        print(f"\nData prepared at: {bin_path}")
        print("You can now train with: python train.py --data-path", bin_path)
