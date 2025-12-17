# Step 2: Data Preparation

This guide covers how to prepare training data for the language model.

## Inputs

- Raw text file(s) for training
- (Or) Use the built-in Shakespeare dataset

## Outputs

After this step, you will have:
- `data/[name].bin` - Binary tokenized data file
- `data/[name]_vocab.json` - Vocabulary file (for character tokenizer)

---

## Data Format Overview

```
┌──────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Raw Text       │ ──► │  Tokenizer   │ ──► │  Binary File    │
│   (.txt)         │     │              │     │  (.bin)         │
│                  │     │              │     │                 │
│ "Hello World"    │     │ encode()     │     │ [7,4,11,11,14]  │
└──────────────────┘     └──────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ Vocabulary   │
                        │ (.json)      │
                        │              │
                        │ {"H":7, ...} │
                        └──────────────┘
```

---

## Option 1: Use Built-in Shakespeare Dataset (Quickstart)

### Command

```bash
python data.py --dataset shakespeare
```

### What This Does

1. Downloads tiny Shakespeare dataset (~1 MB) from GitHub
2. Creates character-level tokenizer from the text
3. Tokenizes the entire text
4. Saves binary data and vocabulary

### Expected Output

```
Downloading tiny Shakespeare to data/shakespeare.txt...
Downloading: 100%
Reading data/shakespeare.txt...
  Total characters: 1,115,394
Tokenizing...
  Created char tokenizer with vocab size: 65
  Total tokens: 1,115,394
  Saving to data/shakespeare.bin...
  Saved vocabulary to data/shakespeare_vocab.json

Data prepared at: data/shakespeare.bin
You can now train with: python train.py --data-path data/shakespeare.bin
```

### Output Files

```
data/
├── shakespeare.txt        # Raw text (1.1 MB)
├── shakespeare.bin        # Tokenized data (2.2 MB, uint16)
└── shakespeare_vocab.json # Character vocabulary (65 chars)
```

### Verify the Data

```bash
python -c "
import numpy as np
data = np.memmap('data/shakespeare.bin', dtype=np.uint16, mode='r')
print(f'Total tokens: {len(data):,}')
print(f'First 20 tokens: {data[:20].tolist()}')
"
```

**Expected Output:**
```
Total tokens: 1,115,394
First 20 tokens: [18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 0, 14, 43, 44, 53, 56]
```

---

## Option 2: Prepare Custom Text Data

### Step 2.1: Prepare Your Text File

Create a text file with your training data:

```bash
# Example: combine multiple files
cat book1.txt book2.txt book3.txt > data/my_corpus.txt

# Or download from internet
wget https://example.com/corpus.txt -O data/my_corpus.txt
```

**Requirements:**
- Plain text file (UTF-8 encoding)
- Recommended: at least 1 MB of text
- Larger datasets = better models

### Step 2.2: Tokenize the Data

**Using Character-level Tokenizer (Simple, Good for Testing):**

```bash
python -c "
from data import prepare_data
prepare_data('data/my_corpus.txt', 'data/my_corpus.bin')
"
```

**Using BPE Tokenizer (Better for Production):**

```bash
python -c "
from data import prepare_data
from tokenizer import BPETokenizer

tokenizer = BPETokenizer('cl100k_base')  # GPT-4's tokenizer
prepare_data('data/my_corpus.txt', 'data/my_corpus.bin', tokenizer=tokenizer)
"
```

### Expected Output

```
Reading data/my_corpus.txt...
  Total characters: 5,234,567
Tokenizing...
  Total tokens: 1,234,567
  Saving to data/my_corpus.bin...
```

---

## Option 3: Using Python API

### Full Control Over Data Preparation

```python
from data import prepare_data, StreamingTextDataset
from tokenizer import CharTokenizer, BPETokenizer

# Method 1: Auto character tokenizer
prepare_data(
    input_path="data/my_text.txt",
    output_path="data/my_text.bin",
    tokenizer=None  # Auto-create char tokenizer
)

# Method 2: Pre-built BPE tokenizer
tokenizer = BPETokenizer("cl100k_base")
prepare_data(
    input_path="data/my_text.txt",
    output_path="data/my_text.bin",
    tokenizer=tokenizer
)

# Method 3: Custom character tokenizer
with open("data/my_text.txt", "r") as f:
    text = f.read()
tokenizer = CharTokenizer.from_text(text)
prepare_data(
    input_path="data/my_text.txt",
    output_path="data/my_text.bin",
    tokenizer=tokenizer
)
```

---

## Data Split (Train/Validation)

The training script automatically splits data:
- **90%** for training
- **10%** for validation

This is controlled in `train.py`:
```python
TextDataset(data_path, seq_len, split="train", train_split=0.9)
TextDataset(data_path, seq_len, split="val", train_split=0.9)
```

---

## Understanding the Binary Format

### File Structure

The `.bin` file is a flat array of token IDs stored as `uint16`:

```
┌────┬────┬────┬────┬────┬────┬────┬─────┐
│ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ ... │
└────┴────┴────┴────┴────┴────┴────┴─────┘
  2B   2B   2B   2B   2B   2B   2B

T0, T1, ... = Token IDs (0 to 65535)
2B = 2 bytes per token (uint16)
```

### Reading Binary Data

```python
import numpy as np

# Memory-mapped reading (efficient for large files)
data = np.memmap('data/shakespeare.bin', dtype=np.uint16, mode='r')

# Get statistics
print(f"Total tokens: {len(data):,}")
print(f"File size: {len(data) * 2 / 1e6:.1f} MB")
print(f"Vocabulary range: {data.min()} to {data.max()}")
```

---

## How DataLoader Uses the Data

During training, the data is loaded in sequences:

```
Binary file: [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, ...]

With seq_len=4:
┌─────────────────┐
│ Batch 1         │
│ x: [T0,T1,T2,T3]│  Input
│ y: [T1,T2,T3,T4]│  Target (shifted by 1)
└─────────────────┘

┌─────────────────┐
│ Batch 2         │
│ x: [T5,T6,T7,T8]│
│ y: [T6,T7,T8,T9]│
└─────────────────┘
```

---

## Recommended Dataset Sizes

| Dataset Size | Training Time | Model Quality |
|--------------|---------------|---------------|
| 1 MB | Minutes | Poor (overfits) |
| 10 MB | Hours | Basic |
| 100 MB | Days | Good |
| 1 GB+ | Weeks | Production |

For learning purposes, Shakespeare (~1 MB) is sufficient.

---

## Troubleshooting

### Issue: `FileNotFoundError: data/shakespeare.txt`

**Solution:** Run data preparation first:
```bash
python data.py --dataset shakespeare
```

### Issue: `UnicodeDecodeError`

**Cause:** Text file is not UTF-8 encoded.

**Solution:** Convert to UTF-8:
```bash
iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt
```

### Issue: Data file is too large

**Solution:** Use memory-mapped loading (already default):
```python
data = np.memmap('data/large_file.bin', dtype=np.uint16, mode='r')
```

### Issue: Out of memory when tokenizing

**Solution:** Process in chunks:
```python
# For very large files, modify data.py to process in chunks
chunk_size = 10_000_000  # characters
# Process and append to binary file incrementally
```

---

## Summary

| Input | Command | Output |
|-------|---------|--------|
| (none) | `python data.py --dataset shakespeare` | `data/shakespeare.bin`, `data/shakespeare_vocab.json` |
| `my_text.txt` | `prepare_data('my_text.txt', 'my_text.bin')` | `my_text.bin`, `my_text_vocab.json` |

---

## Next Step

Now that data is prepared, learn about [Tokenizers](03-tokenizer.md) →

Or skip directly to [Training](04-training.md) →
