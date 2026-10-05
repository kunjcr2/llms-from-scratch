# Multi-Head Latent Attention (MLA)

> DeepSeek's key innovation that **rewrote the transformer architecture**. Introduced in the DeepSeek V2 paper (June 2024).
> MLA replaces the multi-head attention block to achieve **low KV cache size** AND **high performance** — the best of both worlds.

---

## 1. The Problem: KV Cache Dark Side (Recap)

During inference, we cache Keys and Values to avoid redundant computation (see [kv_cache.md](./kv_cache.md)). But the cache size is massive:

```
KV Cache Size = l × b × n × h × s × 2 × 2

where:
    l = transformer layers       n = attention heads
    b = batch size               h = head dimension
    s = context length           2 = K and V caches × 2 bytes (float16)
```

For **DeepSeek R1/V3**: `61 × 1 × 128 × 128 × 100,000 × 2 × 2 = ~400 GB` — completely impractical.

> **Core question**: Can we reduce the KV cache while maintaining performance?

---

## 2. Failed Attempts: MQA and GQA

### Multi-Query Attention (MQA)

**Idea**: Make ALL attention heads share the **same** K and V values.

```
Normal MHA:  K1 ≠ K2 ≠ K3 ≠ K4    (different per head → cache all)
MQA:         K1 = K2 = K3 = K4    (identical → cache only one)
```

- W_K1 = W_K2 = W_K3 = W_K4 → so K1 = K2 = K3 = K4
- W_V1 = W_V2 = W_V3 = W_V4 → so V1 = V2 = V3 = V4
- Queries remain distinct (Q1 ≠ Q2 ≠ Q3 ≠ Q4)

**Result**: Cache size reduced by factor of `n` (number of heads). For GPT-3: **4.5 GB → 48 MB**.

**Problem**: Sharing K/V across all heads **destroys diversity** → worst performance.

---

### Grouped Query Attention (GQA)

**Idea**: Instead of all heads sharing, group heads together. Within each group, K and V are shared.

```
Example: 4 heads, 2 groups
Group 1: K1 = K2 (same color)
Group 2: K3 = K4 (different color from Group 1)
```

- Cache size uses `g` (number of groups) instead of `n` (number of heads)
- Formula: replaces `n` with `g` → `l × b × g × h × s × 2 × 2`

**Result**: For GPT-3 with 8 groups: **384 MB** (between MHA and MQA).

**Problem**: Still loses some performance compared to full MHA.

---

### The Scoreboard

| Mechanism | KV Cache (GPT-3) | Performance | Problem |
|-----------|-----------------|-------------|---------|
| **MHA** | 4.5 GB | ★★★ Best | Cache too large |
| **GQA** | 384 MB | ★★ Medium | Still loses performance |
| **MQA** | 48 MB | ★ Worst | Shared heads kill diversity |

> **The goal**: Can we get MQA-level cache size with MHA-level performance? → **MLA answers yes.**

---

## 3. Multi-Head Latent Attention: The Key Idea

DeepSeek reframed the problem:

> Instead of caching K and V **separately**, what if we cache **one single matrix** with **fewer dimensions** than `n × h`?

### The Two Controllable Factors

In the KV cache formula, the factor `2 × n × h` is what we can target:
- `2` → because we cache K and V separately
- `n × h` → dimension of Keys and Values (e.g., 128 × 128 = 16,384 for DeepSeek)

**MLA eliminates both**: one cache, lower dimension.

---

## 4. How MLA Works: The Latent Projection

### Step 1: Project Input into a Latent Space

Instead of directly computing K and V from input embeddings:

```
Traditional MHA:
    K = X × W_K        (input → keys directly)
    V = X × W_V        (input → values directly)

MLA:
    C_kv = X × W_dkv   (input → latent space, DOWN projection)
    K = C_kv × W_uk    (latent → keys, UP projection)
    V = C_kv × W_uv    (latent → values, UP projection)
```

Where:
- `W_dkv` = **down**-projection matrix (compresses input into latent space)
- `W_uk` = **up**-projection for Keys
- `W_uv` = **up**-projection for Values
- `C_kv` = the **latent matrix** — this is the ONLY thing we cache

```
Input X (4×8) → × W_dkv (8×4) → C_kv (4×4) → × W_uk → K
                                              → × W_uv → V
```

The latent dimension (`d_l`) can be much smaller than `n × h`. DeepSeek chose `d_l = 576` vs. original `n×h = 128×128 = 32,768`.

### Step 2: Queries Stay the Same (Simple Variant)

```
Q = X × W_Q    (unchanged from standard attention)
```

> In the more advanced MLA variant, queries are also projected to a different space — but the simple version keeps queries as-is.

### Step 3: Rest of Attention is Identical

```
Attention Scores = Q × K^T
Attention Weights = softmax(Scores / √d_k) with causal mask
Context Vector   = Attention Weights × V
```

---

## 5. The Absorption Trick (Why This Actually Works)

You might think: "We added an extra matrix multiplication step — how does this help?"

The answer is the **absorption trick**, which is the mathematical insight that makes MLA possible.

### Computing Attention Scores

```
Attn_Scores = Q × K^T
            = (X × W_Q) × (C_kv × W_uk)^T
            = (X × W_Q) × (W_uk^T × C_kv^T)
```

Now rearrange by **absorbing** `W_Q` and `W_uk^T` together:

```
            = X × (W_Q × W_uk^T) × C_kv^T
                   ╰─────────────╯
                   "Absorbed Query"
                   FIXED at pre-training
```

> **Key insight**: `W_Q × W_uk^T` is just a product of two fixed weight matrices — computed once during pre-training, costs nothing at inference.

So the attention score computation becomes:

```
Absorbed_Query = x_new × (W_Q × W_uk^T)     ← only for new token
Attn_Scores    = Absorbed_Query × C_kv^T     ← C_kv is from CACHE
```

**We don't need a separate Keys cache at all** — just `C_kv`.

### Computing Context Vector

```
Context = Attn_Weights × V
        = Attn_Weights × (C_kv × W_uv)
```

And for the final logits:

```
Logits = Context × W_o
       = Attn_Weights × C_kv × (W_uv × W_o)
                                ╰───────────╯
                                Absorbed together
                                FIXED at pre-training
```

> `W_uv × W_o` can also be absorbed — fixed at pre-training, no caching needed.

**We don't need a separate Values cache either** — the same `C_kv` is reused.

---

## 6. What Happens When a New Token Arrives

Step-by-step inference with MLA:

```
New token "bright" comes in:

1. Compute absorbed query:
   q_absorbed = x_bright × (W_Q × W_uk^T)
   ↳ Both W matrices are FIXED (pre-trained), no caching needed

2. Compute new latent vector:
   c_new = x_bright × W_dkv
   ↳ W_dkv is FIXED, just one matrix multiply

3. Update latent cache:
   C_kv_updated = [C_kv_cached ; c_new]
   ↳ Append new vector to cached latent matrix

4. Compute attention scores:
   attn_scores = q_absorbed × C_kv_updated^T
   ↳ Uses the SAME cache as values will

5. Get attention weights:
   attn_weights = softmax(attn_scores / √d) with causal mask

6. Compute values from SAME cache:
   V = C_kv_updated × W_uv
   ↳ Reuses the SAME C_kv_updated, no separate V cache!

7. Get context vector:
   context_bright = attn_weights × V

8. Predict next token:
   context_bright → remaining layers → logits → next token
```

### Contrast with Standard KV Cache

| | Standard KV Cache | MLA |
|---|---|---|
| **Caches needed** | 2 (Keys + Values) | 1 (latent C_kv) |
| **Cache dimension** | n × h per cache | d_l (much smaller) |
| **New token computation** | x × W_Q, x × W_K, x × W_V | x × (W_Q·W_uk^T), x × W_dkv |
| **Head diversity** | ★★★ All distinct | ★★★ All distinct (W_uk, W_uv differ per head) |

---

## 7. Why Performance is Preserved

The critical question: why doesn't MLA suffer the same performance loss as MQA?

In MQA, all heads literally share the same K and V values:
```
K1 = K2 = K3 = K4    ← same content, no diversity
```

In MLA, the latent matrix `C_kv` is shared, but `W_uk` and `W_uv` are **different for every attention head**:

```
C_kv is shared (cached once), BUT:
    K_head1 = C_kv × W_uk_1    ← unique
    K_head2 = C_kv × W_uk_2    ← unique
    K_head3 = C_kv × W_uk_3    ← unique
    ...

    V_head1 = C_kv × W_uv_1    ← unique
    V_head2 = C_kv × W_uv_2    ← unique
    V_head3 = C_kv × W_uv_3    ← unique
    ...
```

> Each head still produces **distinct** K and V values because the up-projection matrices (`W_uk`, `W_uv`) differ per head. These are fixed at pre-training — no caching cost.

**Result**: Full head diversity is maintained → performance matches MHA.

---

## 8. Memory Savings

### New Cache Size Formula

```
MLA Cache Size = l × b × d_l × s × 2

where:
    l   = transformer layers
    b   = batch size
    d_l = latent dimension (chosen by design)
    s   = context length
    2   = 2 bytes per float16 parameter
```

Note: the `× 2` for separate K/V caches is **gone** (only one cache now).

### DeepSeek V3/R1 Reduction

```
Original (MHA):  2 × n × h = 2 × 128 × 128 = 32,768
MLA:             d_l = 576

Reduction factor = 32,768 / 576 ≈ 57×

Cache size:  400 GB → ~6 GB
```

### Full Comparison

| Method | Cache Factor | DeepSeek Cache | Performance |
|--------|-------------|---------------|-------------|
| **MHA** | 2 × n × h = 32,768 | ~400 GB | ★★★ Best |
| **MQA** | 2 × h = 256 | ~3 GB | ★ Worst |
| **GQA** (8 groups) | 2 × g × h | ~50 GB | ★★ Medium |
| **MLA** (d_l=576) | d_l = 576 | **~6 GB** | **★★★ Best** |

> MLA achieves cache sizes **comparable to MQA** while maintaining performance **comparable to MHA**. Best of both worlds.

---

## 9. Summary

```
The MLA Journey:
├── Problem: KV Cache is huge (400 GB for DeepSeek)
│
├── Failed solutions:
│   ├── MQA: Share K/V across all heads → small cache but terrible performance
│   └── GQA: Share K/V within groups → medium cache, medium performance
│
├── MLA Innovation:
│   ├── Project input into a LATENT space: C_kv = X × W_dkv
│   ├── Derive K and V from latent: K = C_kv × W_uk, V = C_kv × W_uv
│   ├── Cache only C_kv (ONE matrix, LOWER dimension)
│   │
│   ├── Absorption trick:
│   │   ├── W_Q and W_uk^T absorbed → "absorbed query" (fixed, no cache)
│   │   └── W_uv and W_o absorbed → (fixed, no cache)
│   │
│   ├── Result: Only C_kv needs caching
│   │   ├── No separate K cache needed
│   │   └── No separate V cache needed
│   │
│   └── Head diversity preserved:
│       └── W_uk, W_uv differ per head → distinct K, V per head
│
└── Impact:
    ├── 57× reduction in cache size (400 GB → ~6 GB)
    ├── Performance matches full MHA
    └── This is why DeepSeek inference is so cheap
```

---

## 10. What's Next

- **Decoupled Rotary Position Embedding (RoPE + MLA)**: The more advanced MLA variant where positional encoding is applied. Requires splitting query and keys into two parts. See `DeepSeek.md` point 9.
- **Coding MLA from scratch**: Implementation of the full mechanism.

---

## References

- [DeepSeek V2 Paper (June 2024)](https://arxiv.org/abs/2405.04434) — Section on Low-Rank KV Joint Compression
- [Build DeepSeek from Scratch — MLA Lecture (Dr. Raj Dandkar)](https://www.youtube.com/watch?v=2TT384U4vQg)
- [kv_cache.md](./kv_cache.md) — Prerequisite: understanding KV Cache
- [deepseek.md](../deepseek.md) — Points 5-9 for brief MQA, GQA, MLA, and RoPE notes
