# Gated DeltaNet — Personal Reference

Yang et al. (NVIDIA / MIT), 2024. Used in Qwen3-Next (2025) as a hybrid with full attention.

---

## Prerequisites: Transformer Attention in 60 Seconds

Three learned projections turn each token embedding `x` into:

```
q = x W_q    (query  — "what am I looking for?")
k = x W_k    (key    — "what label do I file myself under?")
v = x W_v    (value  — "what content do I carry?")
```

Attention computes, for each query at position t:

```
o_t = Σ_i  softmax(q_t · k_i / √d) · v_i
```

Every past value `v_i` comes back, weighted by how well its key `k_i` matches the current query `q_t`. The softmax normalizes the weights so they sum to 1.

**KV cache**: during generation, `k_i` and `v_i` for every past token must be kept in memory, because the softmax needs all of them at read time. This cache grows by `2 × d_head` floats per token, per head, per layer. At 100k tokens with 128-dim heads, that's 25.6M floats per head per layer. It's the main memory bottleneck of inference.

---

## Step 1: Drop the Softmax

Start with the attention formula and remove softmax:

```
o_t = Σ_i (q_t · k_i) · v_i
```

This looks like a downgrade, but watch what happens when you rearrange the parentheses:

```
o_t = Σ_i  v_i · (k_i · q_t)
    = Σ_i  v_i · (k_iᵀ q_t)
    = (Σ_i  v_i k_iᵀ) q_t
       └────────────┘
             S_t
```

The sum `Σ v_i k_iᵀ` does not depend on `q_t`. You can compute it once, store it as a single matrix `S`, and reuse it for any query. `S` is `d_v × d_k` — fixed size, independent of how many tokens you've seen.

**This is the entire idea behind linear attention.** Softmax attention keeps the ingredients (`k_i`, `v_i`) and mixes them at read time. Linear attention pre-mixes at write time and keeps only the mixture.

The KV cache is gone. In its place: one matrix `S` that you update incrementally:

```
S_t = S_{t-1} + v_t k_tᵀ       (add the new token's outer product)
o_t = S_t q_t                   (read)
```

Cost per token is O(d²) regardless of sequence length. No growing buffer.

---

## Step 2: Why Retrieval Works — The Outer Product Trick

This is the mechanical core. One fact stored: `v = [2, 5]` at key `k = [0.8, 0.6]`.

The outer product `v kᵀ` gives:

```
       ┌ 1.6  1.2 ┐
v kᵀ = │          │
       └ 4.0  3.0 ┘
```

Now hit this matrix with any query vector `q`:

```
(v kᵀ) q  =  v (kᵀ q)  =  v · (k · q)
                             ↑       ↑
                          the value   a scalar: how similar q is to k
```

The matrix hands back `v`, scaled by `k · q`. That's a dot product similarity, same as attention. If `q = k` (perfect match), you get `v` back exactly. If `q ⊥ k`, you get zeros.

**One outer product = one retrievable fact.** The matrix is just a container that holds a value and releases it when you query with the matching key.

### Two facts superimposed

Store `v₁=[2,5]` at `k₁=[0.8,0.6]` and `v₂=[9,1]` at `k₂=[0,1]`:

```
S = v₁k₁ᵀ + v₂k₂ᵀ
```

Query with `k₁`:

```
S k₁ = v₁(k₁·k₁) + v₂(k₂·k₁)
     = [2,5]·1.0  + [9,1]·0.6
     = [2,5]      + [5.4, 0.6]    ← fact #2 leaking in
     = [7.4, 5.6]                  ← wanted [2,5]
```

**Every stored fact answers every query.** They just answer at volume `k · q`. Retrieval is all facts shouting at once; you hope the addressed one shouts loudest.

Here `k₂·k₁ = 0.6`, so the interference is massive. In `d = 128`, random unit vectors have dot product ≈ 0 with std `1/√128 ≈ 0.09`, so in practice the intended fact does dominate — but the noise is always there, and it grows with the number of stored facts.

---

## Step 3: Where Plain Linear Attention Breaks

Two failure modes:

### Failure 1: Capacity

Noise from interference scales as `√(n/d)` relative to signal, where n is the number of stored facts and d is the head dimension. Around `n ≈ d`, noise overwhelms signal and recall collapses. For `d = 128`, you get maybe a few dozen clean retrievals per head.

### Failure 2: Staleness

If the same key gets written twice with different values:

```
S = v_old k₁ᵀ + v_new k₁ᵀ = (v_old + v_new) k₁ᵀ
```

Querying `k₁` returns `v_old + v_new`. Both values are stuck in there, summed. There is no way to update a fact — every write is permanent and additive.

Real language constantly updates state: "x = 3" then later "x = 7", or a character moves from the kitchen to the garden. Summation cannot handle this.

---

## Step 4: The Delta Rule — Targeted Editing

DeltaNet (Schlag et al., Yang et al.) fixes the staleness problem. Before writing `v_new` at key `k_t`, first read what's currently stored there, and subtract it out:

```
v_old = S k_t                              # what k_t currently returns
S     = S + β (v_new - v_old) k_tᵀ         # correct the error
```

`β ∈ (0,1)` controls write strength: β=1 is full replacement, β=0.5 is halfway between old and new.

### Why this works — one SGD step

Rewrite the update:

```
S ← S - β (S k_t - v_new) k_tᵀ
```

This is literally one step of gradient descent on the loss `L = ½‖S k_t - v_t‖²`:

```
∂L/∂S = (S k - v) kᵀ
S ← S - β · ∂L/∂S
```

The layer is doing online learning at inference time: for each token, it takes one gradient step to make the memory more accurate at the current key. β is the learning rate.

### What it actually buys — measured

With 16 keys being overwritten repeatedly (tested in the code below):

```
 writes/key      sum    delta
          1    0.911    0.919      ← one write each: roughly the same
          2    0.656    0.886      ← two writes: summation degrades, delta holds
          4    0.486    0.889
          8    0.336    0.861      ← eight rewrites: summation is useless, delta still fine
```

Delta rule does NOT increase raw storage capacity (distinct keys). It prevents stale values from contaminating current ones. That's a different, and more useful, fix.

### Honest caveat

The "converges to the pseudoinverse" claim you'll see in papers assumes multiple passes over the data. Gated DeltaNet makes one online pass, one gradient step per token, with β < 1. It doesn't reach that optimum. Recent writes are close to exact; older ones degrade as later writes with correlated keys eat into them.

---

## Step 5: The Gate — Bulk Erasure

The delta rule edits one slot at a time. But sometimes you need to clear the whole board — a document boundary, a topic switch, a context reset.

The gate `α ∈ (0,1)` multiplies the entire state:

```
S_t = α_t · S_{t-1}
```

- α = 1.0 → keep everything
- α = 0.9 → everything fades by 10%
- α = 0.1 → near-total wipe

Applied every step, this creates exponential decay. Content from 300 tokens ago at α = 0.98 is at `0.98³⁰⁰ ≈ 0.2%` amplitude. This bounds the effective number of things stored at any time, keeping you under the capacity limit even on long sequences.

In the SGD analogy: α is weight decay on the inner learner. Old parameters (stored facts) get regularized toward zero, freeing capacity for new ones.

---

## Step 6: Gated DeltaNet = Both Together

Full update rule, combining gate + delta:

```
S_t = α_t · S_{t-1} + β_t · (v_t - α_t · S_{t-1} @ k_t) · k_tᵀ
o_t = S_t @ q_t
```

Or equivalently (expanding the delta step):

```
S_t = α_t · S_{t-1} · (I - β_t · k_t k_tᵀ) + β_t · v_t k_tᵀ
```

Both `α_t` and `β_t` are **data-dependent** — predicted from the current token via learned linear projections + sigmoid:

```
α_t = sigmoid(x_t @ W_α)      per head
β_t = sigmoid(x_t @ W_β)      per head
```

The model learns when to forget (α low), when to overwrite hard (β high), and when to leave things alone (β low).

### Why each mechanism alone fails

| mechanism | can edit one fact | can clear everything | failure mode |
|---|---|---|---|
| plain sum | no | no | stale values pile up, capacity saturates |
| gate only (Mamba2) | no | yes | to erase one fact, must fade all of them |
| delta only (DeltaNet) | yes | no | can't free capacity after a context switch |
| gate + delta | yes | yes | — |

---

## Step 7: The Generation Cache — "S IS the Cache"

During autoregressive generation:

```
carry forward: S_{t-1}         (d_v × d_k per head — fixed)
new token:     compute q_t, k_t, v_t, α_t, β_t
update:        S_t = α · S_{t-1} + β · (v_t - α · S_{t-1} k_t) k_tᵀ
output:        o_t = S_t q_t
discard:       k_t, v_t          (never needed again)
```

There is no array of past keys or values. `S_t` is a sufficient statistic — it encodes everything the layer needs from the past. After the update, `k_t` and `v_t` are gone.

Concrete sizes per head at d_head = 128:

| | size at T = 100k |
|---|---|
| KV cache (full attention) | 2 × 128 × 100,000 = 25.6M floats |
| S (linear attention) | 128 × 128 = 16,384 floats |

S is 1,562× smaller. And it stays that size forever.

The tradeoff: a KV cache is **lossless** — every k and v sits there individually, retrievable exactly. S is **lossy** — once outer products are summed, you cannot recover individual entries. Information is genuinely destroyed, not compressed-and-recoverable.

---

## Step 8: Practical Reality — Hybrids

No production model uses Gated DeltaNet alone. The lossy recall means hard retrieval tasks (needle-in-haystack, exact copying) degrade compared to full attention.

The solution: **hybrid architectures**. Qwen3-Next uses roughly 3 linear layers per 1 full-attention layer.

- The linear layers handle the bulk of the sequence cheaply (constant-size state).
- The few full-attention layers have a real KV cache, but only in 1/4 of the layers, so it's ~4× smaller and grows 4× slower.

The linear layers carry running context; the attention layers provide exact recall when it matters.

---

## Step 9: Efficiency — Why the Loop Isn't Real

The sequential loop in the code below is O(T) but processes one token at a time — fine for understanding, terrible on a GPU.

The key insight: `(I - β k kᵀ)` is a rank-1 matrix (generalized Householder). Products of these across a chunk of ~64 tokens can be collapsed into a compact matmul using the WY representation. This gives a **chunkwise parallel** form:

- Within a chunk: one big matmul (uses tensor cores, fast)
- Between chunks: pass the state S forward (sequential, but only L/64 steps)

`flash-linear-attention` has the Triton kernels for this. Training cost is O(T) total, with hardware utilization close to full attention's.

---

## Code: Complete Implementation

```python
"""
Gated DeltaNet — minimal reference implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gated_delta_rule(q, k, v, alpha, beta):
    """
    q, k  : (B, H, T, Dk)  — k must be L2-normalized
    v     : (B, H, T, Dv)
    alpha : (B, H, T)      — decay gate
    beta  : (B, H, T)      — write gate
    returns o : (B, H, T, Dv)
    """
    B, H, T, Dk = q.shape
    S = q.new_zeros(B, H, v.shape[-1], Dk)
    out = []
    for t in range(T):
        k_t, v_t = k[:, :, t], v[:, :, t]

        S = alpha[:, :, t, None, None] * S                          # decay
        v_old = torch.einsum('bhvk,bhk->bhv', S, k_t)               # current content at k_t
        delta = beta[:, :, t, None] * (v_t - v_old)                 # error, scaled
        S = S + torch.einsum('bhv,bhk->bhvk', delta, k_t)           # correct

        out.append(torch.einsum('bhvk,bhk->bhv', S, q[:, :, t]))    # read
    return torch.stack(out, dim=2)


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.ab = nn.Linear(d_model, 2 * n_heads)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = F.normalize(F.silu(q), dim=-1), F.normalize(F.silu(k), dim=-1)
        alpha, beta = torch.sigmoid(self.ab(x)).transpose(1, 2).chunk(2, dim=1)
        o = gated_delta_rule(q, k, v, alpha, beta)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))
```

### Test: Delta rule vs plain summation on repeated writes

```python
torch.manual_seed(0)
Dk = Dv = 64

def run(n_keys, writes_per_key, rule):
    k = F.normalize(torch.randn(n_keys, Dk), dim=-1)
    S, latest = torch.zeros(Dv, Dk), {}
    for _ in range(writes_per_key):
        for i in torch.randperm(n_keys):
            v = F.normalize(torch.randn(Dv), dim=-1)
            latest[int(i)] = v
            if rule == "sum":
                S = S + torch.outer(v, k[i])
            else:
                S = S + torch.outer(v - S @ k[i], k[i])
    V = torch.stack([latest[i] for i in range(n_keys)])
    return F.cosine_similarity((S @ k.T).T, V, dim=-1).mean().item()

for w in (1, 2, 4, 8):
    print(f"writes/key={w}  sum={run(16,w,'sum'):.3f}  delta={run(16,w,'delta'):.3f}")
```

### Test: β and α controls

```python
k1 = F.normalize(torch.randn(Dk), dim=0)
S = torch.outer(torch.ones(Dv), k1)             # stores v_old = [1,1,...,1] at k1
v_new = torch.full((Dv,), 5.0)

for b in (0.0, 0.5, 1.0):
    S_new = S + b * torch.outer(v_new - S @ k1, k1)
    print(f"beta={b}: read -> {(S_new @ k1)[0]:.1f}")
    # beta=0.0 -> 1.0 (no write)
    # beta=0.5 -> 3.0 (halfway)
    # beta=1.0 -> 5.0 (full replace)

for a in (1.0, 0.9, 0.1):
    print(f"alpha={a}: read -> {((a * S) @ k1)[0]:.1f}")
    # alpha=1.0 -> 1.0 (no decay)
    # alpha=0.9 -> 0.9 (10% fade)
    # alpha=0.1 -> 0.1 (90% fade)
```

---

## Key Takeaways (for quick re-reading)

1. Remove softmax → associativity lets you pre-sum `v_i k_iᵀ` into a fixed matrix S.
2. S replaces the KV cache. Fixed size (d² per head), no growth.
3. Retrieval = `S q`. Each stored fact comes back weighted by its key-query similarity. Same as attention, different execution order.
4. Interference is the core problem: non-orthogonal keys leak into each other's reads.
5. Delta rule = one gradient step per token on the memory, fixing staleness (repeated writes).
6. Gate = exponential decay, fixing capacity (clearing old junk).
7. Neither fixes the fundamental lossiness. Hybrid with full attention is the practical answer.`