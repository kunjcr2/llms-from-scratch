# LLMs from Scratch

A comprehensive repository covering Large Language Models from fundamentals to advanced architectures. Includes theory, hands-on notebooks, from-scratch implementations, fine-tuning pipelines, and trained artifacts. Built entirely in 2025.

## Highlight: MedAssistGPT

A 401M parameter medical domain LLM, pretrained from scratch on 2 million PubMed abstracts.

**Architecture:**
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) with 4 KV groups
- SwiGLU activation in feed-forward layers
- RMSNorm for layer normalization
- 24 transformer blocks, 1024 hidden dimension, 16 attention heads

**Training Features:**
- Flash Attention optimization for A100 GPU
- Memory-mapped datasets for zero RAM overhead
- Parallel data processing with multiprocessing
- Gradient accumulation (effective batch size: 128)
- Automatic checkpointing and HuggingFace uploads
- Weights and Biases integration

**Links:** [HuggingFace Model](https://huggingface.co/kunjcr2/MedAssist-GPT-401M)

See `MedAssistGPT/` for full implementation.

---

## Repository Structure

### tutorials/

#### DeepSeek (`tutorials/deepseek/`)
Complete implementation and documentation of DeepSeek architecture:

| Topic | Description |
|-------|-------------|
| Multi-Head Latent Attention (MLA) | KV cache compression via latent matrices, absorbed queries |
| Mixture of Experts (MoE) | Sparse expert routing, auxiliary loss, load balancing |
| Shared Experts | Always-activated experts for knowledge redundancy |
| Fine-grained Expert Segmentation | Specialized experts with smaller dimensions |
| Auxiliary Loss-Free Load Balancing | Bias-based routing without loss penalties |
| RoPE + MLA Integration | Decoupled RoPE application in latent attention |
| Multi-Token Prediction (MTP) | Predicting multiple future tokens |

**Code files:** Expert routing, MoE base, noisy top-k, RMSNorm, RoPE, complete DeepSeek implementation notebook.

#### GPT Pipeline (`tutorials/gpt/`)
End-to-end tutorials for building GPT from scratch:

1. **Tokenization** - tiktoken implementation
2. **Attention** - Self-attention and multi-head attention
3. **Architecture** - Full GPT model construction
4. **Training** - Training loops with gradient checkpointing
5. **Post-training** - Techniques after pretraining
6. **Fine-tuning** - LoRA and full fine-tuning methods

#### Mamba (`tutorials/mamba/`)
State Space Models as transformer alternatives:

- **State Space Models (SSM)** - Linear RNNs with discretization, A/B/C/Delta matrices, GPU-parallelizable convolutions
- **Selective State Space Models (SSSM)** - Input-dependent parameters, parallel associative scan, SRAM optimization

#### Mixture of Depths - Kimi (`tutorials/mod/`)
Google DeepMind's dynamic compute allocation:

- **What is MoD** - Top-k routing, per-token layer skipping, static computation graphs
- **How MoD Works** - Per-block routing, residual paths, MoDE (combined with MoE)
- Up to 50% FLOPs reduction while maintaining performance

---

### models/

| Model | Description |
|-------|-------------|
| **MedAssistGPT** | 401M medical LLM with RoPE, GQA, SwiGLU |
| **GPT-155M** | GPT from scratch, 155M parameters |
| **GPT-211M** | GPT from scratch, 211M parameters |
| **GatorGPT** | Modern transformer with GQA, RoPE, SwiGLU, vLLM ready |
| **Qwen2.5 DPO** | Qwen 2.5 fine-tuned with Direct Preference Optimization |
| **Flan-T5** | Fine-tuned Flan-T5 |
| **GPT2-Med** | Fine-tuned GPT-2 Medium |
| **LLaMA 3-3B LoRA** | LoRA adapters for LLaMA 3-3B on OpenHermes |
| **Tinymorfer** | Experimental compact architecture |

---

### docs/

#### RLHF (`docs/RLHF/`)
Complete implementations with theory:

| Component | Files |
|-----------|-------|
| Reward Model | Theory (`.md`) + Implementation (`.py`) |
| Proximal Policy Optimization (PPO) | Theory (`.md`) + Implementation (`.py`) |
| Direct Preference Optimization (DPO) | Theory (`.md`) + Implementation (`.py`) |

#### Reinforcement Learning (`docs/RL/`)
Foundations for LLM alignment: 4 lectures covering RL basics to RLHF.

#### Reasoning Models (`docs/Reasoning Models/`)
Notes on reasoning capabilities across 4 lectures with code.

#### ML and DL Fundamentals (`docs/ml-and-dl/`)

| Topic | Description |
|-------|-------------|
| Flash Attention | Memory-efficient attention |
| Sliding Window Attention | Local attention patterns |
| ALiBi | Attention with Linear Biases |
| Gated Linear Units (GLU) | SwiGLU, GeGLU activations |
| Encoder-only Architectures | BERT-style models |
| vLLM | High-throughput inference |
| Supervised Fine-tuning | Data preprocessing pipelines |
| Backpropagation | Theory and notebook |
| PyTorch Lightning | Training with LightningModule |

#### Optimization (`docs/optimization/`)
- Introduction to Optimization
- SGD Optimizer
- Momentum Gradient Descent
- RMSProp
- Adam Optimizer

---

### vision/

| Model | Files |
|-------|-------|
| Vision Transformers (ViT) | Demo notebook |
| Swin Transformers | Theory (`.md`) + Implementation (`.py`) |
| TinyViT | Implementation (`.py`) |
| DeiT | Data-efficient Image Transformers (`.md`) |
| CNNs | Convolutional Neural Networks (`.md`) |
| Contrastive Learning | CLIP-style (`.md` + `.py`) |
| Vision-Language Models | Theory (`.md`) + Implementation (`.py`) |

---

### eval/
- **BLEU** - Score implementation and documentation for text generation evaluation

### rag/
Retrieval-Augmented Generation notes and implementations.

### papers/
Research papers referenced throughout:
- Attention Is All You Need
- GPT-2
- DeepSeek V3
- DeepSeek MoE
- Mixture of Depths (MoD)
- Vision Transformers
- Layer Normalization
- Residual Connections
- Sparse Attention
- And more

---

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- transformers
- tiktoken
- vLLM (for deployment)

### Clone
```bash
git clone https://github.com/kunjcr2/llms-from-scratch.git
cd llms-from-scratch
```

### Recommended Learning Path

1. **Start with GPT:** `tutorials/gpt/` - Build a transformer from scratch
2. **Explore DeepSeek:** `tutorials/deepseek/` - Modern architecture innovations
3. **Understand Mamba:** `tutorials/mamba/` - Alternative to attention
4. **Study Kimi/MoD:** `tutorials/mod/` - Dynamic compute allocation
5. **Learn alignment:** `docs/RLHF/` - DPO, PPO, Reward Models
6. **See trained models:** `models/` - Working implementations
7. **Dive into vision:** `vision/` - ViT, Swin, contrastive learning

---

## Notebooks Index

| Topic | Path |
|-------|------|
| Tokenizer | `tutorials/gpt/1_tokenizer/LLM_tokenizer.ipynb` |
| Attention | `tutorials/gpt/2_attention/LLM_attention.ipynb` |
| Architecture | `tutorials/gpt/3_architecture/LLM_GPT_arch.ipynb` |
| Training | `tutorials/gpt/4_training/LLM_training.ipynb` |
| Post-training | `tutorials/gpt/5_post_training/LLM_post_training.ipynb` |
| LoRA Fine-tuning | `tutorials/gpt/6_finetune/LLM_LoRA_finetune.ipynb` |
| Full Fine-tuning | `tutorials/gpt/6_finetune/LLM_full_finetune.ipynb` |
| DeepSeek Complete | `tutorials/deepseek/Codes/deepseek_complete.ipynb` |
| Backpropagation | `docs/ml-and-dl/BackProp.ipynb` |
| ViT Demo | `vision/ViT_demo.ipynb` |
| Reasoning Models | `docs/Reasoning Models/Lec3_code.ipynb` |
| GPT-155M | `models/GPT_155M/LLM_155M.ipynb` |
| GPT-211M | `models/GPT_211M/LLM_211M.ipynb` |
| Qwen 2.5 DPO | `models/Qwen2.5_dpo.ipynb` |

---

## Contributing

Contributions welcome. Open an issue or submit a pull request.

## License

None

---

Created and maintained by Kunj Shah
