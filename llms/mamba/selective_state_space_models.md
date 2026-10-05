# Selective State Space Models (Mamba / S4)

## The Core Problem with Standard SSMs

Standard SSMs are **Linear Time-Invariant (LTI)** — meaning the matrices **(A, B, C)** and step size **Δ** are frozen constants. Every single token gets processed the exact same way.

> Think of it like reading a book where you pay equal attention to every word — "the", "cat", "quantum", "entanglement" — all treated identically. That's an LTI SSM.

This is why early SSMs couldn't compete with Transformers. Attention dynamically weighs every token against every other. LTI systems simply can't do that.

---

## The Mamba Fix: Make Everything Input-Dependent

Mamba breaks the LTI constraint by making **B, C, and Δ functions of the current input xₜ**:

| Parameter | LTI (Old) | Mamba (New) | What it controls |
|---|---|---|---|
| **A** | constant | constant | state decay (kept fixed for stability) |
| **B(xₜ)** | constant | ← input xₜ | what enters the hidden state ("write gate") |
| **C(xₜ)** | constant | ← input xₜ | what is read from hidden state to output ("read gate") |
| **Δ(xₜ)** | constant | ← input xₜ | current token vs. memory tradeoff |

### Δ is the most intuitive one to understand:

- **Large Δ** → model focuses on the **current token** (forgets the past)
- **Small Δ** → model relies on **hidden state memory** (ignores current token)

So Mamba can look at a token and decide: *"is this important enough to attend to right now, or should I keep coasting on memory?"* — which is essentially what Attention does, but as a recurrence.

---

## The Problem This Creates

Making parameters input-dependent **kills the Convolutional Trick**.

In a standard LTI SSM, because A/B/C are constant, the output is a **convolution** of the input with a fixed kernel. Convolutions are parallelizable and fast to train.

Once the matrices change at every step, you're back to a **sequential RNN** — you must compute h₁ before h₂ before h₃, etc. That's O(L) serial steps. Slow.

---

## Mamba's Solution: Hardware-Aware Algorithm

Mamba uses two tricks to recover speed:

### 1. Parallel Associative Scan

The recurrence `hₜ = A·hₜ₋₁ + B·xₜ` looks sequential but is actually **associative** — meaning you can tree-reduce it in parallel across GPU threads.

```
Sequential:  h1 → h2 → h3 → h4 → h5 → h6 → h7 → h8    (8 steps)
Parallel:   [h1,h2] [h3,h4] [h5,h6] [h7,h8]             (log₂8 = 3 steps)
             [h1..h4]        [h5..h8]
              [h1..h8]
```

Result: **O(log L)** depth instead of O(L). Fully parallelizable during training.

### 2. SRAM vs HBM Memory Management

The expensive part of GPU computation isn't arithmetic — it's **memory traffic**.

| Memory | Speed | Size | Mamba's use |
|---|---|---|---|
| **HBM** (High Bandwidth Memory) | Slow | Large (GBs) | Only final output `(B, L, D)` written here |
| **SRAM** (on-chip cache) | Fast | Tiny (MBs) | Discretization + full associative scan happen here |

The intermediate states have shape `(B, L, D, N)` — much larger than the output `(B, L, D)`. Keeping them in SRAM and never flushing to HBM is the key insight that makes this practical on real hardware.

---

## Efficiency Summary

| | Transformer | Mamba |
|---|---|---|
| **Training** | O(L²) | O(L) |
| **Inference** | O(L) — KV cache grows with context | **O(1)** — fixed state size always |
| **Parallelizable** | Yes (attention) | Yes (associative scan) |
| **Selective focus** | ✅ (via attention weights) | ✅ (via input-dependent Δ, B, C) |

The O(1) inference is the killer feature. A Transformer's KV cache grows linearly with context length — Mamba's hidden state is **always the same size** regardless of sequence length.

---

## One-Line Intuition

> Mamba is an RNN that learned to selectively pay attention — without the quadratic cost of actually computing attention.