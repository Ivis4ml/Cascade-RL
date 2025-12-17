# Step 3: Tokenizer

This guide explains tokenization and how to use the included tokenizers.

## What is Tokenization?

Tokenization converts text into numbers (tokens) that the model can process:

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ "Hello World"   │ ──► │  Tokenizer   │ ──► │ [15496, 2159]   │
│                 │     │              │     │                 │
│ (Text)          │     │  encode()    │     │ (Token IDs)     │
└─────────────────┘     └──────────────┘     └─────────────────┘
                               │
                               │ decode()
                               ▼
                        ┌─────────────────┐
                        │ "Hello World"   │
                        └─────────────────┘
```

## Available Tokenizers

| Tokenizer | Vocab Size | Best For | Speed |
|-----------|------------|----------|-------|
| `CharTokenizer` | ~100 | Testing, small data | Fast |
| `BPETokenizer` | ~100k | Production | Medium |

---

## Option 1: Character-level Tokenizer

### How It Works

Each character becomes a token:

```
"Hello" → ['H', 'e', 'l', 'l', 'o'] → [7, 4, 11, 11, 14]
```

### Usage

```python
from tokenizer import CharTokenizer

# Create from text
text = "Hello, World! This is a test."
tokenizer = CharTokenizer.from_text(text)

print(f"Vocab size: {tokenizer.vocab_size}")
# Output: Vocab size: 19

# Encode
tokens = tokenizer.encode("Hello")
print(f"Tokens: {tokens}")
# Output: Tokens: [0, 7, 11, 11, 14]

# Decode
text = tokenizer.decode(tokens)
print(f"Text: {text}")
# Output: Text: Hello
```

### Save and Load

```python
# Save vocabulary
tokenizer.save("vocab.json")

# Load vocabulary
tokenizer = CharTokenizer.load("vocab.json")
```

### Vocabulary File Format

```json
{
  "chars": [" ", "!", ",", ".", "H", "T", "W", "a", "d", "e", "h", "i", "l", "o", "r", "s", "t"]
}
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Simple to understand | Long sequences |
| Works with any language | Inefficient for large data |
| Small vocabulary | No subword understanding |
| Fast training | Poor generalization |

---

## Option 2: BPE Tokenizer (tiktoken)

### How It Works

Byte Pair Encoding groups common character sequences:

```
"Hello" → ["Hello"] → [15496]  (single token!)
"unhappiness" → ["un", "happiness"] → [359, 99847]
```

### Usage

```python
from tokenizer import BPETokenizer

# Use GPT-4's tokenizer
tokenizer = BPETokenizer("cl100k_base")

print(f"Vocab size: {tokenizer.vocab_size}")
# Output: Vocab size: 100277

# Encode
tokens = tokenizer.encode("Hello, World!")
print(f"Tokens: {tokens}")
# Output: Tokens: [9906, 11, 4435, 0]

# Decode
text = tokenizer.decode(tokens)
print(f"Text: {text}")
# Output: Text: Hello, World!
```

### Available Encodings

| Encoding | Models | Vocab Size |
|----------|--------|------------|
| `cl100k_base` | GPT-4, ChatGPT | 100,277 |
| `p50k_base` | GPT-3 | 50,281 |
| `r50k_base` | Codex | 50,281 |

### Pros and Cons

| Pros | Cons |
|------|------|
| Efficient encoding | Requires tiktoken library |
| Better generalization | Larger vocabulary |
| Industry standard | More complex |
| Handles rare words | Fixed vocabulary |

---

## Choosing a Tokenizer

### Use CharTokenizer When:
- Learning/experimenting
- Small datasets (< 10 MB)
- Quick testing
- Custom characters needed

### Use BPETokenizer When:
- Production training
- Large datasets (> 100 MB)
- English or common languages
- Need efficient sequences

---

## Tokenization Examples

### Character Tokenizer

```python
from tokenizer import CharTokenizer

text = """First Citizen:
Before we proceed any further, hear me speak."""

tokenizer = CharTokenizer.from_text(text)
tokens = tokenizer.encode(text)

print(f"Original length: {len(text)} characters")
print(f"Token length: {len(tokens)} tokens")
print(f"Vocab size: {tokenizer.vocab_size}")
```

**Output:**
```
Original length: 64 characters
Token length: 64 tokens
Vocab size: 28
```

### BPE Tokenizer

```python
from tokenizer import BPETokenizer

text = """First Citizen:
Before we proceed any further, hear me speak."""

tokenizer = BPETokenizer("cl100k_base")
tokens = tokenizer.encode(text)

print(f"Original length: {len(text)} characters")
print(f"Token length: {len(tokens)} tokens")
print(f"Compression ratio: {len(text)/len(tokens):.1f}x")
```

**Output:**
```
Original length: 64 characters
Token length: 16 tokens
Compression ratio: 4.0x
```

---

## Special Tokens

### Character Tokenizer
No special tokens by default. All characters in the training text become tokens.

### BPE Tokenizer
Has special tokens built-in:
- End of text markers
- Padding tokens
- Unknown token handling

---

## Vocabulary Management

### Check Vocabulary Contents

```python
from tokenizer import CharTokenizer

tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")

# List all tokens
for idx, char in tokenizer.idx_to_char.items():
    print(f"{idx}: '{char}' ({repr(char)})")
```

**Output:**
```
0: '
' ('\n')
1: ' ' (' ')
2: '!' ('!')
3: '$' ('$')
...
```

### Vocabulary Size Impact

| Vocab Size | Sequence Length | Training Speed | Model Quality |
|------------|-----------------|----------------|---------------|
| ~100 (char) | Long | Slow | Lower |
| ~32k (BPE) | Medium | Medium | Good |
| ~100k (BPE) | Short | Fast | Best |

---

## Common Operations

### Get Token for Character

```python
tokenizer = CharTokenizer.load("vocab.json")
token_id = tokenizer.char_to_idx.get('A', -1)
print(f"Token for 'A': {token_id}")
```

### Get Character for Token

```python
char = tokenizer.idx_to_char.get(42, '<UNK>')
print(f"Character for token 42: {char}")
```

### Check if Character Exists

```python
if 'ñ' in tokenizer.char_to_idx:
    print("Spanish character supported")
else:
    print("Spanish character not in vocabulary")
```

---

## Using with Model Configuration

The model's vocabulary size must match the tokenizer:

```python
from model import ModelConfig
from tokenizer import CharTokenizer

# Load tokenizer
tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")

# Create matching model config
config = ModelConfig.small()
config.vocab_size = tokenizer.vocab_size  # Must match!

print(f"Model vocab size: {config.vocab_size}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
```

---

## Troubleshooting

### Issue: `KeyError` when encoding

**Cause:** Character not in vocabulary.

**Solution:** Use a tokenizer trained on your data:
```python
tokenizer = CharTokenizer.from_text(your_training_text)
```

### Issue: `ImportError: tiktoken`

**Solution:** Install tiktoken:
```bash
pip install tiktoken
```

### Issue: Different results after save/load

**Cause:** Vocabulary order changed.

**Solution:** Always use the same vocabulary file for training and inference.

### Issue: Unknown characters appear as wrong tokens

**Solution:** Ensure training data contains all characters you need:
```python
# Check what's in your vocabulary
with open("vocab.json") as f:
    import json
    vocab = json.load(f)
    print(f"Vocabulary: {vocab['chars']}")
```

---

## Summary

| Operation | CharTokenizer | BPETokenizer |
|-----------|---------------|--------------|
| Create | `CharTokenizer.from_text(text)` | `BPETokenizer("cl100k_base")` |
| Encode | `tokenizer.encode(text)` | `tokenizer.encode(text)` |
| Decode | `tokenizer.decode(tokens)` | `tokenizer.decode(tokens)` |
| Save | `tokenizer.save("vocab.json")` | N/A (uses pretrained) |
| Load | `CharTokenizer.load("vocab.json")` | `BPETokenizer("cl100k_base")` |

---

## Next Step

Now that you understand tokenization, proceed to [Training](04-training.md) →
