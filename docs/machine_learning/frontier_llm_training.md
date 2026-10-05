# 🚀 Frontier LLM Training: A Full-Stack Problem

> Notes from the video: *Frontier LLM Training: A Full-Stack Problem*

---

## Overview

Training a frontier LLM (e.g. a 2 trillion parameter model) is a **full-stack systems problem** across five constraint layers:

1. **Memory** — model state doesn't fit on any single GPU
2. **Compute** — astronomical FLOP counts require massive parallelism
3. **Communication** — inter-GPU networks are 18× slower than intra-server links
4. **Numerical Precision** — fewer bits = faster, but gradients break if too few
5. **Fault Tolerance** — at scale, something fails every few hours

---

## 1. 🧮 Memory: 16 Bytes Per Parameter

For each parameter, training keeps **5 entries in memory**:

| Entry | Dtype | Bytes |
|---|---|---|
| Working copy (weight) | BF16 | 2 |
| Gradient | BF16 | 2 |
| Adam optimizer backup (high-precision weight) | FP32 | 4 |
| Adam running mean of gradient | FP32 | 4 |
| Adam running variance of gradient | FP32 | 4 |
| **Total** | | **16** |

- **2T params × 16 bytes = 32 TB** of model state
- An H100 has **80 GB** → need **400 GPUs** just to hold the model state

**Activations** are additional memory:
- Cached during the forward pass for use in backprop
- Scale with **batch size × sequence length**, not parameter count
- At seq len 4,096, one training example = **642 GB**
- Batch of 32 = **~20 TB** — comparable to the model state itself

**Compute:**
- 2T param model ≈ **4.8 × 10²⁶ FLOPs** for a full training run
- A single H100 at half peak would take **~30,700 years**
- 16,384 GPUs together would take **~2 years** → still not feasible without optimization

---

## 2. 🔄 Data Parallelism & Ring All-Reduce

**Data Parallelism**: Copy the full model onto every GPU. Split the batch into N shards. Each GPU processes its shard independently, then gradients are averaged.

The naive approach (funneling gradients to one machine) creates a bottleneck.

### Ring All-Reduce

Splits the averaging into two phases:

**Phase 1 — Reduce-Scatter:**
- Arrange N GPUs in a logical ring
- Split each gradient tensor into N chunks
- Each GPU sends one chunk right, accumulates from left
- After N−1 steps, each GPU holds **one fully reduced chunk**

**Phase 2 — All-Gather:**
- Circulate the N reduced chunks around the ring
- Every GPU ends up with all N chunks = the **complete averaged gradient**

**Communication cost per GPU:**
```
(2 × (N−1) / N) × gradient_size  →  approaches 2× gradient_size as N grows
```
This is **bandwidth-optimal** and **independent of GPU count**.

> In practice, each layer's all-reduce overlaps with earlier layers' backward pass, hiding comms under compute.

### Critical Batch Size & LAMB

- More GPUs = larger effective batch size
- Beyond a **critical batch size**, doubling batch size saves <50% of gradient steps
- **LAMB** (Layer-wise Adaptive Moments) pushed batch size to **32,768**, compressing 3 days into **76 minutes**

---

## 3. 🧊 ZeRO: Zero Redundancy Optimizer

**Problem:** Standard data parallelism replicates the full 32 TB model state on every GPU.

**Solution:** Microsoft DeepSpeed's **ZeRO** — partition model state across GPUs instead.

| Stage | What's partitioned | Memory reduction | Extra communication |
|---|---|---|---|
| Stage 1 | Optimizer states (12/16 bytes) | ~4× | None |
| Stage 2 | + Gradients | ~8× | None |
| Stage 3 | + Parameters (weights) | Linear with GPU count | 1.5× baseline |

**Stage 3 mechanics:**
- No GPU holds the full model at any moment
- Before each layer runs: all-gather weights → compute → discard weights
- For 2T params on 2,048 GPUs: **16 GB per GPU** (fits inside H100's 80 GB)

> ZeRO solves the model state problem. Activations remain a separate challenge.

---

## 4. 💾 Gradient Checkpointing

**Problem:** At batch size 32, activations reach **~20 TB**.

**Standard gradient checkpointing:**
- During the forward pass, **discard most activations**
- During backprop, **recompute them from saved checkpoints**
- Cost: ~33% extra compute; savings: huge activation memory reduction

### Selective Checkpointing

Not all activations are equal:

| Activation type | Memory footprint | Recomputation cost |
|---|---|---|
| Attention internals (softmax, dropout masks) | Large (scales as seq_len²) | Cheap (small matmul + softmax) |
| Linear layer inputs | Smaller | Expensive (full matmul) |

**Selective checkpointing:** Keep the cheap-to-store (linear inputs), throw away the expensive-to-store but cheap-to-recompute (attention internals).

**Results vs. full recomputation:**
- Activation memory: **5× reduction**
- Extra FLOPs: **+2.7% only**
- MFU: **42.1% → 54.2%**

---

## 5. ⚡ FlashAttention: SRAM Tiling

**Problem:** Standard attention builds an N×N matrix. At 131K tokens, that's **17 billion entries per head**, sitting in slow HBM (~3 TB/s).

**On-chip SRAM is ~100× faster, but only tens of MBs.**

### How FlashAttention Works

1. Tile the attention computation into blocks small enough to live in SRAM
2. As each block is processed, track a **running maximum** and **running denominator**
3. When a new block arrives with a larger maximum, apply a correction factor:

```
correction = exp(old_max - new_max)
new_partial_sum = correction × old_partial_sum + new_terms
```

4. The full attention matrix is **never materialized**

**Results:**
- Memory: **O(N²) → O(N)**
- Wall-clock time: **2–4× faster**
- FLOPs: **unchanged** (I/O bound, not compute bound)

> FlashAttention is now the default for long-context training and enables context parallelism.

---

## 6. 🔀 Tensor & Sequence Parallelism

### Tensor Parallelism

**Problem:** Even with ZeRO, each individual matrix multiply runs on one GPU.

**Solution:** Split the matrix multiply itself across GPUs.

For an MLP with two consecutive big matmuls and a per-element op in between:
- Slice the **first matrix vertically**
- Slice the **second matrix horizontally**
- The seams align → each GPU's partial result of the first multiply feeds its slice of the second multiply with **no mid-computation communication**
- One all-reduce at the very end

For attention: it's even more natural — heads are already independent, split heads across GPUs.

**Cost:** 4 blocking all-reduces per layer (2 forward, 2 backward). Every GPU must wait.

**This is why Llama 3 pins tensor parallelism at exactly 8** — one server's worth of GPUs connected by NVLink (900 GB/s). Crossing servers means Infiniband (50 GB/s), an **18× cliff**.

### Sequence Parallelism

After the attention/MLP stitching, every GPU holds the same full output. Per-token operations (LayerNorm, Dropout, residuals) don't mix tokens → **pure duplication**.

- Split along the **token dimension** instead
- Each GPU handles its own slice of tokens
- Same total bytes moved, just reorganized as a scatter + gather around the per-token ops
- Activation memory drops by factor of **T** (tensor parallel degree)

---

## 7. 🚂 Pipeline Parallelism

**Setup:** 128 layers across 16 GPUs = 8 layers per GPU. Forward pass flows sequentially → at any moment, 1 GPU works and 15 wait. This idle time is the **pipeline bubble**.

### Bubble Fraction Formula

```
bubble = (P - 1) / (M + P - 1)
```
where P = stages, M = microbatches in flight.

### Scheduling Strategies

| Strategy | Description | Trade-off |
|---|---|---|
| Naive | One microbatch at a time | Large bubble |
| Microbatch splitting | Split batch into M smaller microbatches; GPUs pipeline | Bubble shrinks, but needs M activations in memory |
| **1F1B** (one-forward-one-backward) | After warm-up, alternate fwd/bwd; activations freed immediately | Same bubble ratio, only P activations in flight |
| **Interleaved** | Each GPU owns non-contiguous layers (e.g., 1, 33, 65, 97) | Bubble ÷4, communication ×4 |
| **Backward split** | Separate weight gradients (no downstream dep) from input gradients; slot into bubble | **+23% throughput** |

### Llama 3's 126-Layer Trick

To make 126 layers divide evenly across 16 GPUs:
- First GPU (embedding) and last GPU (output projection) have **7 layers each**
- Middle 14 GPUs have **8 layers each** → 7 + (14×8) + 7 = **126**
- Every GPU finishes at the same instant → **+6.5% MFU, -5 GB peak memory**

---

## 8. 🌐 Ring Attention: Million-Token Context

**Problem:** FlashAttention handles 131K tokens on one GPU. 1M tokens exceeds a single device.

**Context Parallelism** splits the sequence across GPUs using **Ring Attention**:

1. Arrange GPUs in a logical ring, each holding a **chunk of query tokens**
2. Key-value blocks rotate around the ring, one hop at a time
3. Each GPU computes **partial attention** against its current KV block, then passes it on
4. Transfer **overlaps with computation** (next block arrives while current is being consumed)
5. After one full rotation, every query has attended to every key
6. The **causal mask** zeroes out future tokens → exact attention

**Llama 3 result:** 1M-token prefill with 405B model in **77 seconds** across 128 H100s at **93% parallelization efficiency**.

---

## 9. 🧩 Mixture of Experts (MoE)

**Motivation:** No dense 2T-parameter model has ever been trained. MoE gives massive capacity with only fractional compute per token.

- A **router** selects each token's top-K experts from a large pool
- Only K experts activate per token → sparse compute

**Two problems with MoE across GPUs:**

### Problem 1: All-to-All Communication

Every MoE layer fires **two all-to-all collectives**:
- **Dispatch**: Send tokens to their assigned expert's GPU
- **Combine**: Return the outputs

(Unlike all-reduce which sums into one result, all-to-all exchanges data between every GPU pair.)

### Problem 2: Load Imbalance

If a few experts are overloaded, those GPUs bottleneck.

| Approach | Mechanism | Downside |
|---|---|---|
| Auxiliary loss (Switch Transformer, α=1e-2) | Penalizes uneven routing | Fights the router's learned preferences; can hurt quality |
| **DeepSeek-V3 bias routing** | Add bias to each expert's routing score (raised for underloaded, lowered for overloaded) | Bias only shifts *selection*, never enters gating weights → gradients flow cleanly |

> DeepSeek-V3: no auxiliary loss, no dropped tokens, no loss spikes across 14.8T token run.

---

## 10. 🔢 Mixed Precision Training

Every weight is a floating point number: `sign | mantissa | exponent`

| Format | Total bits | Exponent bits | Mantissa bits | Max value | Training notes |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | ~3.4 × 10³⁸ | Full precision baseline |
| FP16 | 16 | 5 | 10 | 65,504 | Gradients overflow; needs loss scaling |
| **BF16** | 16 | **8** | 7 | ~3.4 × 10³⁸ | Same range as FP32; **default for training** |
| FP8 | 8 | — | — | — | 2× faster matmuls on H100; needs tile-wise scaling |
| FP4 | 4 | — | — | — | Blackwell; still an open research problem |

### BF16

- Reallocates FP16's bits: 8 exponent (same as FP32) + 7 mantissa
- Restores full FP32 range → no gradient overflow, no loss scaling needed
- Loses 3 mantissa bits, but **range >> precision** for training stability

### FP8 (DeepSeek-V3)

**Tile-wise scaling to make FP8 work:**
- Weights split into **128×128 blocks**, each with its own scale factor
- Activations split into **1×128 tiles** (one per token), refreshed every step
- Partial sums accumulated in **FP32** so rounding doesn't compound
- Loss stayed within **0.25% of BF16 baseline**
- Results: **39% less memory, 75% faster training**

---

## 11. 🏗️ Parallelism Strategies: Llama 3 vs DeepSeek-V3

Total GPUs = Tensor × Pipeline × Context × Expert × Data parallelism

| Dimension | Llama 3 405B | DeepSeek-V3 671B |
|---|---|---|
| GPUs | 16,384 H100 | 2,048 H800 |
| Tensor parallelism | **8** (NVLink, 900 GB/s) | **None** (H800 NVLink = 400 GB/s, too slow) |
| Pipeline parallelism | 16 | 16 |
| Data parallelism | 128 (ZeRO-sharded) | — |
| Expert parallelism | None (dense) | 64 |
| Context parallelism | 16 (for 131K context) | — |
| Active params/token | 405B | 37B (of 671B total, 256 experts) |
| GPU hours | ~30M | **2.788M** |
| Cost | ~$750M (at 2T scale) | **~$5.6M** |

**DeepSeek-V3's DualPipe schedule** overlaps expert routing communication with forward/backward compute, hiding latency under useful work.

> ~11× fewer GPU hours than Llama 3 at comparable benchmark quality, shaped entirely by hardware constraints.

---

## 12. 📐 Chinchilla Scaling Law: The 6ND Rule

```
C = 6 × N × D
```
- **C** = total FLOPs
- **N** = number of parameters
- **D** = number of training tokens
- **6** = flop multiplier (2 for forward + 4 for backward)

**Compute-optimal training (Chinchilla):** ~20 tokens per parameter

For 2T parameters:
- Tokens: 2T × 20 = **40 trillion**
- FLOPs: 6 × 2T × 40T = **4.8 × 10²⁶**
- Cost at 50% utilization on $2/hr H100s: **~$750 million**

> No team has paid the full bill for a 2T dense model. MoE is what makes trillion-parameter training financially viable.

---

## 13. 🛡️ Fault Tolerance at Scale

During Llama 3's **54-day training run** on 16,384 GPUs:

| Metric | Value |
|---|---|
| Total unexpected failures | **419** (~1 every 3 hours) |
| GPU / NVLink failures | 30% |
| HBM memory failures | 17% |
| Network + software | ~53% |
| Failures needing human intervention | **3 out of 419** |
| Effective training time | **>90%** |

### Checkpointing Hierarchy

| Frequency | Location | Checkpoint type |
|---|---|---|
| Every ~5 min | Host RAM | In-memory snapshot |
| Every 30 min | Local NVMe SSD | Fast local checkpoint |
| Every few hours | Distributed storage | Durable remote checkpoint |

A full Llama 3 405B checkpoint is **~6.5 TB**.

### Recovery Pipeline

1. Anomaly detection pauses affected tasks
2. Diagnostics identify the fault
3. Orchestrator swaps in a **hot spare**
4. Training resumes from latest checkpoint

**Google PaLM 540B** saw ~20 loss spikes during training. Fix: restart from a checkpoint 100 steps before each spike and skip the offending data batches.

---

## 14. 🗂️ End-to-End System Architecture

### Data Pipeline

- **~1 PB** of raw text → cleaned, deduplicated, tokenized → **~160 TB of integer tokens**
- Corpus mix: ~50% web, ~25% math/reasoning, ~17% code, remainder multilingual + curated
- Each GPU pulls a **deterministic slice** (indexed by step + rank); no two GPUs see the same token
- CPU-side **prefetch queue** keeps next batches ready; tail latency (not bandwidth) is the real enemy

### Training Context Schedule

1. Main training at **8,000 token** context
2. Long-context ramp to **128,000 tokens** in 6 stages
3. Final cooldown: decay learning rate → zero; upsample highest-quality sources

### Control Plane

- **Cluster scheduler** → launches all processes
- **Parallelism mesh** → assigns each GPU its slot in each dimension
- **Async checkpoint daemon** → memory → SSD → long-term storage
- **Health daemon** → watches for stragglers and faults; triggers hot-spare swap

---

## 15. 🔭 What's Next

| Development | Description | Impact |
|---|---|---|
| **Hardware-aware co-design** | Tailor parallelism strategy to specific hardware (DeepSeek-V3 style) | Frontier results at a fraction of cost |
| **GB200 NVL72** | 72 Blackwell GPUs per rack, NVLink 5 at 1,800 GB/s/GPU, 13.5 TB GPU memory | Tensor parallelism ceiling jumps 8→72; pipeline parallelism within a rack may become unnecessary |
| **DiLoCo** | Workers run local optimizer for 100s of steps, then sync parameter deltas (500× less comm) | 96% scaling efficiency across data centers 1,000 km apart |
| **FP4 training** | 4-bit weights on Blackwell; 16 values for the whole number line | Still an open problem; early results track FP8 closely |

---

## 📊 Quick Reference: Key Numbers

| Quantity | Value |
|---|---|
| H100 HBM capacity | 80 GB |
| H100 HBM bandwidth | ~3 TB/s |
| H100 SRAM bandwidth | ~100× HBM |
| NVLink (within server) | 900 GB/s |
| Infiniband (across servers) | 50 GB/s |
| Ratio (NVLink / Infiniband) | **18×** |
| Memory per parameter (training) | **16 bytes** |
| 2T model state | **32 TB** |
| GPUs to hold 2T model state | **400 H100s** |
| Chinchilla tokens per parameter | 20 |
| Activation memory (seqlen 4096, bs 32) | ~20 TB |
| FlashAttention memory improvement | O(N²) → O(N) |
| Selective checkpointing memory gain | 5× |
| Selective checkpointing compute cost | +2.7% FLOPs |
| FP8 vs BF16 | 2× faster matmuls, 39% less memory |
