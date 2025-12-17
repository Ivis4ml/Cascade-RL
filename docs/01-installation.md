# Step 1: Installation

This guide covers environment setup and dependency installation.

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| PyTorch | 2.0+ | 2.1+ |
| CUDA | 11.7+ | 12.0+ |
| GPU VRAM | 4 GB | 16+ GB |
| RAM | 8 GB | 32+ GB |

## Inputs

- A working Python installation
- (Optional) NVIDIA GPU with CUDA support

## Outputs

After this step, you will have:
- All required Python packages installed
- Verified PyTorch installation with GPU support (if available)

---

## Option 1: Basic Installation (Recommended)

### Step 1.1: Create Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 1.2: Install PyTorch

Choose based on your hardware:

**With NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**With NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**macOS (Apple Silicon):**
```bash
pip install torch  # MPS support included by default
```

### Step 1.3: Install Other Dependencies

```bash
pip install numpy
pip install tiktoken  # Optional: for BPE tokenizer
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## Option 2: Using Conda

```bash
# Create conda environment
conda create -n minimal-llm python=3.10 -y
conda activate minimal-llm

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install numpy tiktoken
```

---

## Verification

### Check PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

**Expected Output:**
```
PyTorch version: 2.1.0
```

### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected Output (with GPU):**
```
CUDA available: True
```

### Check GPU Information

```bash
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU available, will use CPU')
"
```

**Expected Output:**
```
GPU: NVIDIA GeForce RTX 3090
VRAM: 24.0 GB
```

### Test Model Import

```bash
python -c "from model import GPT, ModelConfig; print('Model import OK')"
```

**Expected Output:**
```
Model import OK
```

### Quick Functionality Test

```bash
python model.py
```

**Expected Output:**
```
Model config: ModelConfig(vocab_size=32000, n_layers=12, ...)
Total parameters: 124,668,672
Input shape: torch.Size([2, 128])
Logits shape: torch.Size([2, 128, 32000])
Loss: 10.XXXX
```

---

## Memory Requirements by Model Size

| Model | Parameters | Training (bf16) | Inference |
|-------|------------|-----------------|-----------|
| small | ~125M | ~4 GB | ~1 GB |
| medium | ~350M | ~8 GB | ~2 GB |
| large | ~760M | ~16 GB | ~4 GB |
| gpt-oss | ~2.7B | ~48 GB | ~12 GB |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install PyTorch:
```bash
pip install torch
```

### Issue: `CUDA out of memory`

**Solution:** Reduce batch size or use a smaller model:
```bash
python train.py --batch-size 4 --model-size small
```

### Issue: `torch.cuda.is_available()` returns `False`

**Possible causes:**
1. No NVIDIA GPU installed
2. CUDA drivers not installed
3. Wrong PyTorch version (CPU-only)

**Solution:** Reinstall PyTorch with CUDA support:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Slow training on Apple Silicon

**Solution:** Ensure MPS is being used:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

The training script automatically uses MPS when available.

---

## Next Step

Once installation is verified, proceed to [Data Preparation](02-data-preparation.md) →
