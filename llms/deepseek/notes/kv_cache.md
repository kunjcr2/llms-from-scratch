# Key-Value Cache (KV Cache)

> Foundational concept required before understanding Multi-Head Latent Attention (MLA).
> KV Cache only applies during the **inference** stage of LLMs — not during pre-training.

---

## 1. Pre-training vs Inference

- **Pre-training**: The model learns its parameters (e.g., 175B params for GPT-3). All weights are trained and then fixed.
- **Inference**: The pre-trained model is used to **predict one new token at a time**. This is the stage where KV Cache matters.

When you type a prompt into ChatGPT, you're in the inference stage — the model is already trained and is now generating tokens sequentially.

---

## 2. How Inference Works (The Autoregressive Loop)

LLM inference is a **next-token prediction** loop:

```
Step 1:  "the next day is"      → LLM → predicts "bright"
Step 2:  "the next day is bright"  → LLM → predicts "and"
Step 3:  "the next day is bright and" → LLM → predicts "lovely"
...repeat until max tokens or EOS
```

Each time a new token is predicted, it is **appended back** to the input sequence, and the entire sequence is fed through the LLM architecture again.

---

## 3. Key Insight: Only the Last Token's Context Vector Matters

When the full input sequence passes through the transformer and produces a **logits matrix**, each row corresponds to one token and has a dimension equal to the vocabulary size (e.g., 50,000).

```
Input: "the  next  day  is"

Logits matrix (4 × 50,000):
    the   → [50,000-dim vector]   ← irrelevant
    next  → [50,000-dim vector]   ← irrelevant
    day   → [50,000-dim vector]   ← irrelevant
    is    → [50,000-dim vector]   ← THIS is all we need
```

> **To predict the next token, we ONLY need the logits vector for the last token in the input sequence.**

That logits vector depends on the **context vector** of the last token, which is computed in the attention block. After the attention block, the earlier tokens are no longer needed — only the last token's context vector travels through the remaining layers (feed-forward, layer norm, output projection) to produce the final logits.

> **Insight #1**: We only need the **context vector for the last token** to predict the next token.

---

## 4. The Redundancy Problem

Watch what happens across inference steps:

### Step 1 — Input: `"the next day is"` (4 tokens)

```
Input Embeddings:  4×8
Q, K, V:           4×4  (each)
Attention Scores:  4×4
Attention Weights: 4×4
Context Vectors:   4×4
```

Predict → `"bright"`

### Step 2 — Input: `"the next day is bright"` (5 tokens)

```
Input Embeddings:  5×8
Q, K, V:           5×4  (each)
Attention Scores:  5×5
Attention Weights: 5×5
Context Vectors:   5×4
```

**Look at what's repeated:**

| Matrix | Step 1 Size | Step 2 Size | Repeated Portion |
|--------|------------|------------|-----------------|
| Q, K, V | 4×4 | 5×4 | Top 4×4 rows already computed in Step 1 |
| Attn Scores | 4×4 | 5×5 | Top-left 4×4 block already computed |
| Attn Weights | 4×4 | 5×5 | Top-left 4×4 block already computed |
| Context Vectors | 4×4 | 5×4 | Top 4 rows already computed |

> **We are recomputing the Q, K, V, attention scores, attention weights, and context vectors for ALL previous tokens — even though we only need the context vector for the LAST token.**

---

## 5. What Exactly Do We Need to Cache?

Since we only need the context vector of the last token (`bright`), let's **backtrack** and figure out the minimum computation required:

### To get context vector for `bright`:

```
context_bright = attn_weights_bright × V_matrix
```

### To get attention weights for `bright`:

```
attn_weights_bright = softmax(attn_scores_bright / √d_k)
```

### To get attention scores for `bright`:

```
attn_scores_bright = q_bright × K_matrix.T     (1×d × d×n → 1×n)
```

### What we need:
1. **`q_bright`** → Just compute: `x_bright × W_Q`  (one new computation)
2. **`K_matrix`** → Top rows from **cache** + new `k_bright = x_bright × W_K`
3. **`V_matrix`** → Top rows from **cache** + new `v_bright = x_bright × W_V`

> **We do NOT need to cache queries!** We only ever need the query vector for the current (last) token. But we need the full Keys and Values matrices to compute attention — and we can **cache** the previously computed portions.

---

## 6. The KV Cache Algorithm

When a **new token** arrives during inference:

```
1. Compute q_new = x_new × W_Q       ← query for new token only
2. Compute k_new = x_new × W_K       ← key for new token only
3. Compute v_new = x_new × W_V       ← value for new token only

4. K_matrix = [K_cache ; k_new]       ← append new key to cached keys
5. V_matrix = [V_cache ; v_new]       ← append new value to cached values

6. attn_scores = q_new × K_matrix.T   ← 1×n scores (not n×n!)
7. attn_weights = softmax(scores/√d)  ← apply scaling + causal mask + softmax
8. context_new = attn_weights × V_matrix  ← context vector for new token only

9. Update cache: K_cache ← K_matrix
                 V_cache ← V_matrix

10. context_new → rest of transformer layers → logits → next token prediction
```

**Only 3 new matrix multiplications** per inference step (steps 1-3), instead of recomputing everything for all tokens.

> Since we only cache **Keys** and **Values** (not Queries), this is called the **Key-Value Cache** or **KV Cache**.

---

## 7. Advantages of KV Cache

### Computational Speedup

| Without KV Cache | With KV Cache |
|-------------------|--------------|
| Recompute Q, K, V for ALL tokens every step | Only compute q, k, v for the NEW token |
| Attention computation scales **quadratically** with sequence length | Computation scales **linearly** with sequence length |
| Redundant work grows with every new token | Minimal new work per token |

### Empirical Speedup (GPT-2, 100 new tokens)

```
With KV Cache:    ~2 seconds
Without KV Cache: ~7 seconds
Speedup:          ~3.5x faster
```

Even on small models like GPT-2, KV Cache provides a **~1.4–3.5x speedup**. On larger models, the speedup is even more significant.

---

## 8. The Dark Side: KV Cache Memory Cost

KV Cache speeds up inference but **occupies memory**. Every cached key and value vector takes up space, and we pay for every byte stored.

### KV Cache Size Formula

```
KV Cache Size = l × b × n × h × s × 2 × 2

where:
    l = number of transformer blocks (layers)
    b = batch size
    n = number of attention heads
    h = attention head dimension
    s = context length (sequence length)
    2 = one for Keys, one for Values
    2 = bytes per parameter (float16 = 2 bytes)
```

### Real-World Examples

| Model | Layers (l) | Heads (n) | Head Dim (h) | Context (s) | Batch (b) | KV Cache Size |
|-------|-----------|-----------|-------------|-------------|-----------|--------------|
| 30B param model | 48 | — | 7168 (n×h) | 1024 | 128 | **~180 GB** |
| DeepSeek V3/R1 | 61 | 128 | 128 | 100,000 | 1 | **~400 GB** |

> **Key observation**: KV Cache size scales **linearly** with context length `s`. Doubling the context length doubles the cache size. This is why OpenAI charges **more for larger context windows** — GPT-4 (8K context) is $30/M tokens while GPT-4-32K is $60/M tokens.

### Why This Matters

- The cache consumes GPU memory (VRAM) that could be used for other computations.
- As models get larger (more layers, more heads) and context windows grow (100K+ tokens), the KV cache size grows dramatically.
- For DeepSeek R1, a naive KV cache would require **400 GB** — far exceeding the memory of even the most powerful individual GPUs.
- The plot of GPT-3 variants (Small → Medium → Large → XL) shows KV cache size growing in a **near-quadratic** manner as model size increases.

---

## 9. Solutions to the KV Cache Memory Problem

The dark side of KV Cache motivated several innovations, each progressively reducing cache size:

| Technique | Approach | Cache Reduction |
|-----------|----------|----------------|
| **Multi-Query Attention (MQA)** | All attention heads share the SAME K and V | ~130x smaller (400GB → 3GB) |
| **Grouped Query Attention (GQA)** | K and V shared within groups of heads | Between MHA and MQA |
| **Multi-Head Latent Attention (MLA)** | Cache a single low-rank latent `C_kv` instead of K and V separately | Massive reduction (DeepSeek: ~6GB) |

> DeepSeek did **not** use MQA or GQA — they invented **MLA** which achieves both low memory AND high performance. See `DeepSeek.md` for MLA details.

### Also Important in Serving: PagedAttention (vLLM)

The techniques above reduce the *amount* of KV cache per token/head. PagedAttention solves a different but related production issue: **how KV memory is allocated and reused across many concurrent requests**.

Key distinction:
1. MQA/GQA/MLA: reduce KV representation size.
2. PagedAttention: reduce allocator fragmentation and idle reserved memory.

In interview language:
1. "KV-cache compression lowers memory footprint per token."
2. "PagedAttention improves memory utilization efficiency at runtime."
3. "Both are complementary in high-throughput inference systems."

---

## 10. Summary

```
Why KV Cache?
├── During inference, LLMs predict one token at a time (autoregressive)
├── Each step re-feeds the growing sequence through the full architecture
├── Without caching → redundant recomputation of K, V for all previous tokens
├── Key insight: only the LAST token's context vector matters for prediction
│
├── Solution: Cache the Keys and Values matrices
│   ├── Only compute q, k, v for the NEW token
│   ├── Append new k, v to the cache
│   └── Use cached K, V to compute attention for the new token only
│
├── Advantages
│   ├── Reduces compute from quadratic → linear in sequence length
│   └── Empirically ~1.4–3.5x faster (GPT-2), even more for larger models
│
└── Disadvantages (The Dark Side)
    ├── Cache size = l × b × n × h × s × 2 × 2
    ├── Scales linearly with context length (why longer context costs more)
    ├── DeepSeek R1 naive cache = 400 GB
    └── Motivated innovations:
        - representation-side: MQA → GQA → MLA (DeepSeek's solution)
        - serving-side allocation: PagedAttention (vLLM)
```

---

## References

- [Build DeepSeek from Scratch — KV Cache Lecture (Dr. Raj Dandkar)](https://youtu.be/2TT384U4vQg)
- Points 4–6 in [`deepseek.md`](../deepseek.md) for brief KV Cache, MQA, and GQA notes
- Points 7+ in [`deepseek.md`](../deepseek.md) for Multi-Head Latent Attention (MLA)
