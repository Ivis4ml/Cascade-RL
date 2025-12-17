# Step 5: Text Generation

This guide covers how to generate text using trained models.

## Inputs

- Trained model checkpoint (`checkpoints/ckpt_*.pt`)
- Vocabulary file (`data/*_vocab.json`) for char tokenizer
- Text prompt

## Outputs

- Generated text continuation

---

## Quick Start

### Single Prompt Generation

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_2000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "ROMEO: " \
    --max-tokens 200
```

### Expected Output

```
Loading checkpoint from checkpoints/ckpt_2000.pt...
Loaded model with 124,668,672 parameters
Using char tokenizer with vocab size: 65

Generated text:
----------------------------------------
ROMEO: O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
Like a rich jewel in an Ethiope's ear;
Beauty too rich for use, for earth too dear!
----------------------------------------
```

---

## Generation Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    Autoregressive Generation                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: "ROMEO: "                                               │
│         ↓                                                       │
│  ┌─────────────┐                                                │
│  │  Tokenize   │ → [48, 47, 45, 17, 47, 10, 1]                 │
│  └─────────────┘                                                │
│         ↓                                                       │
│  ┌─────────────┐     ┌─────────────┐                           │
│  │   Model     │ ──► │   Logits    │ → Probability distribution │
│  │  Forward    │     │  [vocab_sz] │                           │
│  └─────────────┘     └─────────────┘                           │
│         ↓                    ↓                                  │
│  ┌─────────────┐     ┌─────────────┐                           │
│  │   Sample    │ ◄── │  Sampling   │ ← temperature, top_k, top_p│
│  │ Next Token  │     │  Strategy   │                           │
│  └─────────────┘     └─────────────┘                           │
│         ↓                                                       │
│  Append token → Repeat until max_tokens                        │
│         ↓                                                       │
│  ┌─────────────┐                                                │
│  │   Decode    │ → "ROMEO: O, she doth teach..."              │
│  └─────────────┘                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--prompt` | None | Text prompt (interactive if not set) |
| `--max-tokens` | 200 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-k` | 50 | Top-k sampling |
| `--top-p` | 0.9 | Nucleus sampling |
| `--tokenizer` | `char` | Tokenizer type: `char` or `bpe` |
| `--vocab-path` | None | Path to vocabulary file |
| `--device` | `cuda` | Device: `cuda`, `cpu`, `mps` |

---

## Sampling Parameters

### Temperature

Controls randomness in generation:

```
temperature = 0.0  → Deterministic (always pick highest probability)
temperature = 0.7  → Balanced (some creativity)
temperature = 1.0  → Original distribution
temperature = 1.5  → More random/creative
```

**Visual:**
```
Low temperature (0.2):    High temperature (1.5):
████████░░ "the"          ███░░░░░░░ "the"
██░░░░░░░░ "a"            ██░░░░░░░░ "elephant"
░░░░░░░░░░ "elephant"     ██░░░░░░░░ "a"
                          █░░░░░░░░░ "banana"
```

### Top-k Sampling

Only consider the k most likely tokens:

```
top_k = 1   → Greedy (only most likely)
top_k = 10  → Conservative
top_k = 50  → Balanced
top_k = 100 → More variety
```

### Top-p (Nucleus) Sampling

Only consider tokens until cumulative probability reaches p:

```
top_p = 0.5  → Conservative
top_p = 0.9  → Balanced
top_p = 0.95 → More variety
top_p = 1.0  → No filtering
```

---

## Usage Examples

### Deterministic Generation (Greedy)

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_2000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "To be or not to be" \
    --temperature 0.0
```

Always produces the same output for the same prompt.

### Creative Generation

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_2000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "To be or not to be" \
    --temperature 1.0 \
    --top-k 100
```

More varied, creative output.

### Balanced Generation (Recommended)

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_2000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "To be or not to be" \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9
```

Good balance of coherence and creativity.

---

## Interactive Mode

Run without `--prompt` to enter interactive mode:

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_2000.pt \
    --vocab-path data/shakespeare_vocab.json
```

### Interactive Session

```
==================================================
Interactive text generation
Type a prompt and press Enter to generate
Type 'quit' or 'exit' to stop
==================================================

Prompt: HAMLET:

Generating...

----------------------------------------
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them.
----------------------------------------

Prompt: JULIET:

Generating...

----------------------------------------
JULIET: O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
----------------------------------------

Prompt: quit

Goodbye!
```

---

## Python API

### Basic Generation

```python
from generate import load_model, generate_text
from tokenizer import CharTokenizer

# Load model
model, config = load_model("checkpoints/ckpt_2000.pt", device="cuda")

# Load tokenizer
tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")

# Generate
text = generate_text(
    model, tokenizer,
    prompt="ROMEO: ",
    max_new_tokens=200,
    temperature=0.8
)
print(text)
```

### Direct Model Generation

```python
import torch
from model import GPT, ModelConfig

# Load checkpoint
checkpoint = torch.load("checkpoints/ckpt_2000.pt")
config = ModelConfig(**checkpoint["model_config"])

# Create model
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to("cuda")

# Tokenize prompt
from tokenizer import CharTokenizer
tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")
tokens = tokenizer.encode("ROMEO: ")
input_ids = torch.tensor([tokens], device="cuda")

# Generate
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50
    )

# Decode
text = tokenizer.decode(output_ids[0].tolist())
print(text)
```

---

## KV Cache

The model uses KV (Key-Value) cache for efficient generation:

```
Without KV Cache:
Step 1: Process [T0]
Step 2: Process [T0, T1]         (recompute T0)
Step 3: Process [T0, T1, T2]     (recompute T0, T1)
... O(n²) computation

With KV Cache:
Step 1: Process [T0], cache KV
Step 2: Process [T1], use cached KV, update cache
Step 3: Process [T2], use cached KV, update cache
... O(n) computation
```

This is handled automatically in `model.generate()`.

---

## Generation Quality

### Signs of Good Training

```
ROMEO: O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
Like a rich jewel in an Ethiope's ear;
```
- Coherent sentences
- Follows style/format
- Correct grammar

### Signs of Undertrained Model

```
ROMEO: the the the the the and and
to be to be to be to be
```
- Repetitive
- No coherence
- Random tokens

### Signs of Overtrained Model

```
ROMEO: O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
```
- Exact quotes from training data
- No variation
- Memorization

---

## Multiple Samples

Generate multiple completions for comparison:

```python
from generate import load_model, generate_text
from tokenizer import CharTokenizer

model, config = load_model("checkpoints/ckpt_2000.pt")
tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")

prompt = "HAMLET: To be or not to be"

for i in range(3):
    print(f"\n=== Sample {i+1} ===")
    text = generate_text(
        model, tokenizer,
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.9
    )
    print(text)
```

---

## Batch Generation

Generate multiple prompts at once:

```python
import torch
from model import GPT, ModelConfig
from tokenizer import CharTokenizer

# Load model
checkpoint = torch.load("checkpoints/ckpt_2000.pt")
config = ModelConfig(**checkpoint["model_config"])
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.eval().to("cuda")

# Load tokenizer
tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")

# Multiple prompts
prompts = ["ROMEO: ", "JULIET: ", "HAMLET: "]

# Tokenize (pad to same length)
max_len = max(len(tokenizer.encode(p)) for p in prompts)
batch = []
for p in prompts:
    tokens = tokenizer.encode(p)
    tokens = tokens + [0] * (max_len - len(tokens))  # Pad
    batch.append(tokens)

input_ids = torch.tensor(batch, device="cuda")

# Generate
with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=50)

# Decode each
for i, ids in enumerate(output_ids):
    text = tokenizer.decode(ids.tolist())
    print(f"\n{prompts[i]}\n{text}")
```

---

## Troubleshooting

### Issue: Output is garbage/random

**Causes:**
1. Model not trained enough
2. Wrong vocabulary file

**Solutions:**
- Train longer: `--max-iters 5000`
- Use correct vocab: `--vocab-path data/shakespeare_vocab.json`

### Issue: Output is repetitive

**Causes:**
1. Temperature too low
2. Top-k too small

**Solutions:**
```bash
python generate.py --temperature 0.9 --top-k 100
```

### Issue: Output is too random/incoherent

**Causes:**
1. Temperature too high
2. Model undertrained

**Solutions:**
```bash
python generate.py --temperature 0.7 --top-k 40
```

### Issue: `KeyError` when decoding

**Cause:** Vocabulary mismatch

**Solution:** Use the vocabulary from training:
```bash
python generate.py --vocab-path data/shakespeare_vocab.json
```

### Issue: Slow generation

**Solutions:**
1. Use GPU: `--device cuda`
2. Reduce max tokens: `--max-tokens 100`
3. Use torch.compile during training

---

## Summary

| Goal | Command |
|------|---------|
| Single prompt | `python generate.py --checkpoint ... --prompt "text"` |
| Interactive | `python generate.py --checkpoint ...` |
| Deterministic | `python generate.py --checkpoint ... --temperature 0.0` |
| Creative | `python generate.py --checkpoint ... --temperature 1.0` |

---

## Next Step

Learn about the [Model Architecture](06-architecture.md) →
