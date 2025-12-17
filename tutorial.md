# 最小 LLM 训练教程

本教程详细介绍如何使用这个最小化的 LLM 训练代码库，包括每个组件的原理和使用方法。

## 目录

1. [项目概述](#1-项目概述)
2. [模型架构 (model.py)](#2-模型架构-modelpy)
3. [训练脚本 (train.py)](#3-训练脚本-trainpy)
4. [分词器 (tokenizer.py)](#4-分词器-tokenizerpy)
5. [数据准备 (data.py)](#5-数据准备-datapy)
6. [文本生成 (generate.py)](#6-文本生成-generatepy)
7. [完整训练示例](#7-完整训练示例)
8. [架构设计解析](#8-架构设计解析)

---

## 1. 项目概述

这是一个完全使用 PyTorch 实现的最小化 LLM 训练代码，参考了 [nanochat](https://github.com/karpathy/nanochat) 的设计理念和 [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) 的架构特性。

### 核心特性

| 特性 | 传统 GPT | 本实现 (GPT-OSS 风格) |
|------|----------|----------------------|
| 归一化 | LayerNorm | **RMSNorm** |
| 位置编码 | 学习式绝对位置 | **RoPE 旋转位置编码** |
| 注意力 | MHA | **GQA 分组查询注意力** |
| 激活函数 | GELU | **SwiGLU** |
| 线性层偏置 | 有 | **无** |

### 文件结构

```
├── model.py        # 模型定义 (GPT + 所有组件)
├── train.py        # 训练循环
├── tokenizer.py    # 分词器实现
├── data.py         # 数据加载和预处理
├── generate.py     # 推理和文本生成
└── requirements.txt
```

---

## 2. 模型架构 (model.py)

### 2.1 模型配置 (ModelConfig)

```python
from model import ModelConfig

# 使用预设配置
config = ModelConfig.small()   # ~125M 参数
config = ModelConfig.medium()  # ~350M 参数
config = ModelConfig.large()   # ~760M 参数
config = ModelConfig.gpt_oss_style()  # GPT-OSS 20B 风格

# 或自定义配置
config = ModelConfig(
    vocab_size=32000,      # 词表大小
    n_layers=12,           # Transformer 层数
    n_heads=12,            # 注意力头数
    n_kv_heads=4,          # KV 头数 (用于 GQA)
    d_model=768,           # 隐藏层维度
    d_ff=2048,             # FFN 中间层维度
    max_seq_len=2048,      # 最大序列长度
    dropout=0.0,           # Dropout 率
    rope_theta=10000.0     # RoPE 基础频率
)
```

### 2.2 RMSNorm (均方根归一化)

RMSNorm 是 LayerNorm 的简化版本，只做缩放不做平移：

```python
class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / RMS(x) * γ
    其中 RMS(x) = sqrt(mean(x²))

    优点：
    - 计算更快（无需计算均值）
    - 参数更少（无 bias）
    - 训练稳定性相当
    """
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

### 2.3 RoPE (旋转位置编码)

RoPE 通过旋转向量来编码位置信息：

```python
def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    """
    预计算 RoPE 频率

    原理：将位置信息编码为复数旋转
    - 位置 p 的编码: e^(i * p * θ)
    - θ = 10000^(-2k/d), k 是维度索引

    优点：
    - 相对位置信息天然嵌入
    - 可外推到更长序列
    - 无需学习位置参数
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(q, k, freqs_cis):
    """应用 RoPE 到 Q 和 K"""
    q_complex = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))
    # 复数乘法 = 旋转
    q_rotated = torch.view_as_real(q_complex * freqs_cis)
    k_rotated = torch.view_as_real(k_complex * freqs_cis)
    return q_rotated.flatten(-2), k_rotated.flatten(-2)
```

### 2.4 GQA (分组查询注意力)

GQA 让多个 Query 头共享同一个 Key/Value 头：

```python
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力

    传统 MHA: n_heads 个 Q, K, V 头 (1:1:1)
    MQA:      n_heads 个 Q 头, 1 个 K, V 头 (n:1:1)
    GQA:      n_heads 个 Q 头, n_kv_heads 个 K, V 头 (n:k:k)

    例如: n_heads=64, n_kv_heads=8
    每 8 个 Q 头共享 1 个 KV 头

    优点：
    - 减少 KV Cache 大小 (推理加速)
    - 保持模型质量
    """
    def __init__(self, config):
        self.n_rep = config.n_heads // config.n_kv_heads  # Q/KV 比例

        # Q 投影: d_model -> n_heads * head_dim
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        # K, V 投影: d_model -> n_kv_heads * head_dim (更小)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
```

### 2.5 SwiGLU (门控线性单元)

SwiGLU 结合了 Swish 激活和门控机制：

```python
class SwiGLU(nn.Module):
    """
    SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)

    其中 Swish(x) = x * sigmoid(x) = SiLU(x)

    对比传统 FFN:
    - 传统: GELU(xW1) @ W2
    - SwiGLU: (Swish(xW_gate) * xW_up) @ W_down

    优点：
    - 更好的梯度流动
    - 实践中效果更好
    """
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish 激活
        up = self.w_up(x)
        return self.w_down(gate * up)  # 门控 + 投影
```

### 2.6 完整模型使用

```python
from model import GPT, ModelConfig
import torch

# 创建模型
config = ModelConfig.small()
model = GPT(config)
print(f"参数量: {model.count_parameters():,}")  # ~125M

# 前向传播
input_ids = torch.randint(0, config.vocab_size, (2, 128))  # [batch, seq_len]
targets = torch.randint(0, config.vocab_size, (2, 128))

logits, loss, _ = model(input_ids, targets)
print(f"Logits 形状: {logits.shape}")  # [2, 128, vocab_size]
print(f"Loss: {loss.item():.4f}")

# 文本生成
generated = model.generate(
    input_ids[:, :10],  # 前10个token作为提示
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

---

## 3. 训练脚本 (train.py)

### 3.1 训练配置

```python
from train import TrainConfig

config = TrainConfig(
    # 数据
    batch_size=32,                      # 批次大小
    seq_len=512,                        # 序列长度
    gradient_accumulation_steps=4,      # 梯度累积步数

    # 训练
    max_iters=10000,                    # 总迭代次数
    eval_interval=500,                  # 评估间隔
    log_interval=10,                    # 日志间隔

    # 优化器
    learning_rate=3e-4,                 # 峰值学习率
    min_lr=3e-5,                        # 最小学习率
    weight_decay=0.1,                   # 权重衰减
    beta1=0.9, beta2=0.95,             # Adam 参数
    grad_clip=1.0,                      # 梯度裁剪

    # 学习率调度
    warmup_iters=500,                   # 预热步数
    lr_decay_iters=10000,               # 衰减结束步数

    # 系统
    device="cuda",
    dtype="bfloat16",                   # 混合精度
    compile=False,                      # torch.compile
)
```

### 3.2 学习率调度

使用 Warmup + Cosine Decay 调度：

```
学习率
  ^
  |      /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
  |     /                    \
  |    /                       \
  |   /                          \_____ min_lr
  |  /
  | /
  +----------------------------------------> 迭代
    warmup     constant      decay
```

```python
def get_lr(iter_num, config):
    # 1. 线性预热
    if iter_num < config.warmup_iters:
        return config.learning_rate * iter_num / config.warmup_iters

    # 2. 余弦衰减
    if iter_num > config.lr_decay_iters:
        return config.min_lr

    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)
```

### 3.3 命令行使用

```bash
# 基础训练（使用合成数据测试）
python train.py --model-size small --max-iters 1000

# 使用真实数据
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --batch-size 16 \
    --seq-len 256 \
    --max-iters 5000 \
    --learning-rate 1e-3

# 大模型训练
python train.py \
    --model-size large \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --compile  # 使用 torch.compile 加速

# 从检查点恢复
python train.py \
    --resume-from checkpoints/ckpt_5000.pt \
    --max-iters 10000
```

### 3.4 分布式训练

```bash
# 单机多卡 (DDP)
torchrun --nproc_per_node=4 train.py \
    --model-size medium \
    --batch-size 8

# 多机多卡
torchrun --nnodes=2 --nproc_per_node=8 \
    --node_rank=0 --master_addr=主节点IP \
    train.py --model-size large
```

---

## 4. 分词器 (tokenizer.py)

### 4.1 字符级分词器

适合快速测试和小数据集：

```python
from tokenizer import CharTokenizer

# 从文本创建
text = "Hello, World! 你好世界"
tokenizer = CharTokenizer.from_text(text)
print(f"词表大小: {tokenizer.vocab_size}")  # 字符数

# 编码/解码
tokens = tokenizer.encode("Hello")  # [7, 4, 11, 11, 14]
text = tokenizer.decode(tokens)     # "Hello"

# 保存/加载
tokenizer.save("vocab.json")
tokenizer = CharTokenizer.load("vocab.json")
```

### 4.2 BPE 分词器 (tiktoken)

适合生产环境：

```python
from tokenizer import BPETokenizer

tokenizer = BPETokenizer("cl100k_base")  # GPT-4 使用的编码
print(f"词表大小: {tokenizer.vocab_size}")  # ~100k

tokens = tokenizer.encode("Hello, World!")
text = tokenizer.decode(tokens)
```

---

## 5. 数据准备 (data.py)

### 5.1 下载示例数据

```python
from data import download_shakespeare, prepare_shakespeare

# 下载 Shakespeare 数据集 (~1MB)
text_path = download_shakespeare("data/")

# 下载并预处理（分词 + 保存为二进制）
bin_path = prepare_shakespeare("data/")
# 生成: data/shakespeare.bin, data/shakespeare_vocab.json
```

### 5.2 准备自定义数据

```python
from data import prepare_data
from tokenizer import CharTokenizer

# 准备文本文件
prepare_data(
    input_path="my_corpus.txt",
    output_path="data/my_corpus.bin",
    tokenizer=None  # 自动创建字符级分词器
)

# 或使用自定义分词器
from tokenizer import BPETokenizer
tokenizer = BPETokenizer()
prepare_data(
    input_path="my_corpus.txt",
    output_path="data/my_corpus.bin",
    tokenizer=tokenizer
)
```

### 5.3 数据集类

```python
from train import TextDataset, SyntheticDataset

# 文本数据集
dataset = TextDataset(
    data_path="data/shakespeare.txt",  # 或 .bin
    seq_len=512,
    split="train",      # "train" 或 "val"
    train_split=0.9     # 90% 训练, 10% 验证
)

# 合成数据集 (用于调试)
dataset = SyntheticDataset(
    vocab_size=32000,
    seq_len=512,
    num_samples=10000
)

# 获取一个样本
x, y = dataset[0]  # x: 输入, y: 目标 (右移一位)
```

---

## 6. 文本生成 (generate.py)

### 6.1 命令行使用

```bash
# 单次生成
python generate.py \
    --checkpoint checkpoints/ckpt_5000.pt \
    --prompt "To be or not to be" \
    --max-tokens 200 \
    --temperature 0.8

# 交互模式
python generate.py \
    --checkpoint checkpoints/ckpt_5000.pt \
    --tokenizer char \
    --vocab-path data/shakespeare_vocab.json
```

### 6.2 生成参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `temperature` | 控制随机性，越高越随机 | 0.7-1.0 |
| `top_k` | 只从概率最高的 k 个 token 采样 | 40-100 |
| `top_p` | 核采样，累积概率阈值 | 0.9-0.95 |

```python
# 确定性生成 (贪婪)
output = model.generate(input_ids, temperature=0.0)

# 创造性生成
output = model.generate(input_ids, temperature=1.0, top_k=50, top_p=0.9)

# 平衡生成
output = model.generate(input_ids, temperature=0.7, top_k=40)
```

### 6.3 Python API

```python
from generate import load_model, generate_text
from tokenizer import CharTokenizer

# 加载模型
model, config = load_model("checkpoints/ckpt_5000.pt", device="cuda")

# 加载分词器
tokenizer = CharTokenizer.load("data/shakespeare_vocab.json")

# 生成
text = generate_text(
    model, tokenizer,
    prompt="ROMEO: ",
    max_new_tokens=200,
    temperature=0.8
)
print(text)
```

---

## 7. 完整训练示例

### 7.1 训练 Shakespeare 模型

```bash
# 步骤 1: 准备数据
python data.py --dataset shakespeare

# 步骤 2: 训练
python train.py \
    --data-path data/shakespeare.bin \
    --model-size small \
    --batch-size 32 \
    --seq-len 256 \
    --max-iters 5000 \
    --learning-rate 1e-3 \
    --log-interval 100 \
    --eval-interval 500

# 步骤 3: 生成
python generate.py \
    --checkpoint checkpoints/ckpt_5000.pt \
    --vocab-path data/shakespeare_vocab.json \
    --prompt "HAMLET: To be or not"
```

### 7.2 训练更大的模型

```bash
# Medium 模型 (~350M) - 需要约 8GB 显存
python train.py \
    --data-path data/my_corpus.bin \
    --model-size medium \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --max-iters 50000 \
    --compile

# Large 模型 (~760M) - 需要约 16GB 显存
python train.py \
    --model-size large \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --compile
```

---

## 8. 架构设计解析

### 8.1 参数量计算

对于一个 Transformer 模型，主要参数分布：

```
词嵌入:      vocab_size × d_model
每层注意力:  4 × d_model × d_model (对于 MHA)
            (1 + 1/n_rep × 2) × d_model² (对于 GQA)
每层 FFN:    3 × d_model × d_ff (对于 SwiGLU)
输出层:      d_model × vocab_size

总参数 ≈ 12 × n_layers × d_model² (近似)
```

### 8.2 显存估算

训练时显存占用：

```
参数:          P bytes (fp32: 4P, bf16: 2P)
梯度:          P bytes
优化器状态:    8P bytes (Adam: m + v)
激活值:        取决于 batch_size × seq_len
```

粗略估算：
- Small (~125M): ~4GB
- Medium (~350M): ~8GB
- Large (~760M): ~16GB

### 8.3 训练技巧

1. **梯度累积**: 用小 batch 模拟大 batch
   ```bash
   --batch-size 4 --gradient-accumulation-steps 8  # 等效 batch=32
   ```

2. **混合精度**: 默认使用 bfloat16/float16

3. **梯度裁剪**: 默认 `grad_clip=1.0` 防止梯度爆炸

4. **学习率**:
   - Small 模型: 1e-3 ~ 3e-4
   - Large 模型: 3e-4 ~ 1e-4

5. **torch.compile**: PyTorch 2.0+ 可加速 20-30%
   ```bash
   --compile
   ```

---

## 常见问题

**Q: 为什么用 RMSNorm 而不是 LayerNorm?**
A: RMSNorm 计算更快，参数更少，且在 LLM 中效果相当甚至更好。

**Q: GQA 的 n_kv_heads 怎么选?**
A: 通常 n_heads / n_kv_heads = 4~8。例如 64 个 Q 头配 8 个 KV 头。

**Q: SwiGLU 的 d_ff 怎么设置?**
A: 通常 d_ff ≈ 2.67 × d_model（因为 SwiGLU 有 3 个矩阵而非 2 个）。

**Q: 训练不收敛怎么办?**
A: 尝试降低学习率、增加 warmup、检查数据质量。

---

## 参考资料

- [nanochat](https://github.com/karpathy/nanochat) - Karpathy 的简洁实现
- [GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) - OpenAI 的开源模型
- [RoPE 论文](https://arxiv.org/abs/2104.09864) - 旋转位置编码
- [GQA 论文](https://arxiv.org/abs/2305.13245) - 分组查询注意力
- [SwiGLU 论文](https://arxiv.org/abs/2002.05202) - GLU 变体
