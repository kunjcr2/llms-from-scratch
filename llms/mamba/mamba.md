# Mamba Architecture

> **Prerequisite**: Read `StateSpaceModels.md` → `SelectiveStateSpaceModels.md` before this file.

Mamba is a **Selective State Space Model** introduced by Albert Gu and Tri Dao in *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"* (Dec 2023). It replaces the Transformer's attention mechanism entirely — no KV cache, no quadratic bottleneck — while matching or exceeding Transformer performance across language, audio, and genomics benchmarks.

---

## High-Level Architecture

A Mamba model stacks **N identical Mamba blocks**, each containing a selective SSM as its core. Unlike Transformers, there is **no attention layer** and **no MLP block** in the traditional sense — the Mamba block fuses both roles.

```
Input Tokens
     │
     ▼
┌────────────┐
│  Embedding │
└─────┬──────┘
      │
      ▼  (repeat × N)
┌─────────────────────────┐
│       Mamba Block       │
│  ┌─────────────────────┐│
│  │  Linear Projection  ││
│  │  (expand dim by E)  ││
│  └─────────┬───────────┘│
│     ┌──────┴──────┐     │
│     ▼             ▼     │
│  ┌──────┐   ┌─────────┐ │
│  │Conv1D│   │  SiLU   │ │
│  │+ SiLU│   │ (gate)  │ │
│  └──┬───┘   └────┬────┘ │
│     ▼            │      │
│  ┌──────────┐    │      │
│  │Selective │    │      │
│  │  SSM     │    │      │
│  └──┬───────┘    │      │
│     ▼            │      │
│   element-wise   │      │
│   multiply ◄─────┘      │
│     ▼                   │
│  ┌──────────┐           │
│  │ Linear   │           │
│  │(project  │           │
│  │  back)   │           │
│  └──┬───────┘           │
│     │ + residual        │
└─────┼───────────────────┘
      ▼
┌────────────┐
│  RMS Norm  │
│  + LM Head │
└────────────┘
```

---

## The Mamba Block in Detail

Each Mamba block applies these operations to an input $x \in \mathbb{R}^{B \times L \times D}$:

### 1. Dual Linear Projections

The input is projected into **two branches** with expansion factor $E$ (typically $E=2$):

$$x' = \text{Linear}(x) \in \mathbb{R}^{B \times L \times ED}$$
$$z = \text{Linear}(x) \in \mathbb{R}^{B \times L \times ED}$$

Branch $x'$ goes through the SSM path. Branch $z$ acts as a **gating signal**.

### 2. Depthwise Convolution

A short 1D causal convolution (kernel size $d=4$ typically) is applied along the sequence dimension:

$$x'' = \text{SiLU}(\text{Conv1D}(x'))$$

> This convolution provides **local context mixing** — a lightweight way to look at the few tokens immediately before the current position, compensating for the SSM's purely sequential nature.

### 3. Selective SSM (The Core)

This is the heart of Mamba. The model maintains a **hidden state** $h$ — think of it as a compressed summary of everything the model has seen. Each new token updates this summary.

From $x''$, three input-dependent projections produce the selective parameters:

$$B(x'') = \text{Linear}_B(x'') \in \mathbb{R}^{B \times L \times N}$$
$$C(x'') = \text{Linear}_C(x'') \in \mathbb{R}^{B \times L \times N}$$
$$\Delta(x'') = \text{softplus}(\text{Linear}_\Delta(x'')) \in \mathbb{R}^{B \times L \times ED}$$

Where $N$ is the SSM state dimension (typically 16). Then the discretized recurrence is:

$$\bar{A} = \exp(\Delta \cdot A) \quad \bar{B} = \Delta \cdot B$$
$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x''_t$$
$$y_t = C \cdot h_t$$

**What each parameter actually does:**

- **$B$** = the **"write gate"** — decides what part of the current token gets written into memory
- **$C$** = the **"read gate"** — decides what part of memory gets read out as the output
- **$\Delta$** = the **"care knob"** — decides how much to care about this specific token vs. keep coasting on memory (see dedicated section below)
- **$A$** = the **"decay pattern"** — how old memories fade over time (stays fixed, see HiPPO section below)

> **Key Insight**: $A$ stays fixed (initialized for long-range memory), but $B$, $C$, and $\Delta$ are **input-dependent** — the model learns to selectively focus on tokens that matter. This is Mamba's version of "attention."

### 4. Gating and Output Projection

$$\text{out} = (y \odot \text{SiLU}(z))$$
$$\text{output} = \text{Linear}_{\text{out}}(\text{out}) + x \quad \text{(residual connection)}$$

---

## Parameter Summary

For a Mamba block with model dimension $D$, expansion $E$, SSM state dimension $N$, and conv kernel $d$:

| Component | Shape | Role |
|---|---|---|
| `in_proj` | $(D, 2 \cdot ED)$ | Dual projection (SSM path + gate) |
| `conv1d` | $(ED, 1, d)$ | Depthwise causal convolution |
| `x_proj` | $(ED, \Delta_{\text{rank}} + 2N)$ | Projects to $\Delta$, $B$, $C$ jointly |
| `dt_proj` | $(\Delta_{\text{rank}}, ED)$ | Expands low-rank $\Delta$ |
| `A_log` | $(ED, N)$ | Log-space state matrix (fixed init) |
| `D` | $(ED,)$ | Skip/direct connection |
| `out_proj` | $(ED, D)$ | Project back to model dimension |

---

## WTF is Delta (Δ)?

Delta is the most important parameter to understand. It's the **"how much do I care about this token"** knob.

The SSM recurrence is:

$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t$$

Where $\bar{A} = \exp(\Delta \cdot A)$ and $\bar{B} = \Delta \cdot B$. Watch what happens at the extremes:

| $\Delta$ value | $\bar{A} = \exp(\Delta \cdot A)$ | $\bar{B} = \Delta \cdot B$ | What happens |
|---|---|---|---|
| **Large** | → 0 (kills old state) | Large (amplifies input) | **"Forget everything, this token is important"** |
| **Small** | → 1 (preserves old state) | ≈ 0 (ignores input) | **"Skip this token, keep coasting on memory"** |

**Real example**: The model sees the sentence *"The cat sat on the quantum"*:
- "The", "on", "the" → tiny $\Delta$ → model barely updates its state (filler words)
- "cat", "quantum" → large $\Delta$ → model wipes/overwrites parts of its state (content words)

> $\Delta$ is **input-dependent**: it's computed from the current token via a learned linear projection + softplus. The model LEARNS which tokens deserve attention. This is **Mamba's equivalent of attention weights** — but it runs in O(1) instead of O(L).

---

## WTF is HiPPO?

**Short answer**: HiPPO is just a fancy initialization for matrix $A$. In practice it boils down to:

```python
A = [1, 2, 3, 4, ..., N]  # that's it. that's HiPPO in Mamba.
```

**Why these numbers?** Remember, $A$ controls how old memories decay. Different dimensions of the hidden state $h$ use different entries of $A$. So:

- Dimension 1 uses $A_1 = -1$ → **slow decay** → remembers far back
- Dimension 8 uses $A_8 = -8$ → **medium decay** → remembers recent paragraphs
- Dimension 16 uses $A_{16} = -16$ → **fast decay** → only remembers last few tokens

> Think of it like having 16 note-takers in a lecture. One writes down the overall topic (slow decay). One tracks the current paragraph (medium). One tracks the exact last word (fast). Together they form a **multi-resolution memory** — the model can recall information at any timescale.

The full HiPPO framework (High-order Polynomial Projection Operators) derives these values from approximation theory — it proves that these specific decay rates optimally compress a continuous signal into a fixed-size polynomial basis. But **you don't need to understand the theory**. The practical implementation is just `A = [1, 2, 3, ..., N]` stored in log-space.

$$A_n = -(n + 1) \quad \text{for } n = 1, 2, \dots, N$$

Stored as `A_log = log(A)` so that after `exp(A_log)`, the values remain positive (and the negative sign is applied separately), ensuring stable dynamics.

**Is HiPPO required?** No. You could init $A$ randomly and the model would still train. HiPPO just makes training converge faster and gives better long-range memory out of the box.

---

## Why No Attention?

| Feature | Transformer | Mamba |
|---|---|---|
| **Mixing mechanism** | Self-Attention (global pairwise) | Selective SSM (compressed state) |
| **Training complexity** | $O(L^2 D)$ | $O(L \cdot D \cdot N)$ |
| **Inference per step** | $O(L)$ — reads full KV cache | $O(1)$ — fixed-size state |
| **Memory at inference** | KV cache grows with $L$ | State is always $(ED \times N)$ |
| **Parallelizable training** | ✅ (attention is batched matmul) | ✅ (parallel associative scan) |
| **Content-based selection** | ✅ (attention weights) | ✅ (input-dependent $\Delta, B, C$) |

> Mamba trades the Transformer's **exact pairwise token comparison** for a **compressed running summary** — losing some fine-grained retrieval ability but gaining constant-time inference and linear-time training.

---

## Mamba-2: Simplified and Faster

Mamba-2 (Dao & Gu, 2024) further refines the architecture:

1. **State Space Duality (SSD)**: Shows that the selective SSM is mathematically equivalent to a restricted form of linear attention with a specific masking structure. This unifies SSMs and attention under one framework.
2. **Larger state dimension**: Increases $N$ from 16 to 64–256 by leveraging the SSD connection for more efficient computation.
3. **Multi-head SSM**: Analogous to multi-head attention — multiple independent SSM heads operating in parallel.
4. **~2× faster on hardware**: By expressing the scan as matrix multiplies on tensor cores rather than a custom CUDA kernel.

---

## When to Use Mamba Over Transformers

| Scenario | Recommended |
|---|---|
| Long sequences (>8K tokens) | ✅ Mamba — linear scaling |
| Real-time / streaming inference | ✅ Mamba — O(1) per step |
| Tasks requiring exact retrieval from context | ⚠️ Transformer — attention excels here |
| Memory-constrained deployment | ✅ Mamba — no KV cache |
| Tasks with clear sequential structure (audio, DNA) | ✅ Mamba |
| General-purpose LLM (chatbots, reasoning) | Hybrid approaches work best |

---

## Training Code

For a complete, runnable example of training a Mamba-based sequence model from scratch, see:

📄 **`./mamba_train.py`** — A self-contained script that builds a minimal Mamba model and trains it on a character-level text generation task.

---

## One-Line Intuition

> Mamba is a Transformer without attention — it replaces pairwise token comparisons with a learned, input-dependent running memory that compresses the entire history into a fixed-size state.