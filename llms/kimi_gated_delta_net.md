# Kimi K3 — Kimi Delta Attention (KDA) Notes

From Kimi K3 Technical Report (2025). KDA is K3's core attention layer — essentially Gated DeltaNet with two upgrades: **channel-wise decay** and a **lower-bounded decay parameterization** that unlocks faster GPU kernels.

---

## Section 1: Prerequisites (from the GDN notes)

If you've read the Gated DeltaNet notes, skip to Section 2. Quick recap:

**Linear attention** replaces the KV cache with a fixed-size matrix `S ∈ ℝ^{d_k × d_v}`:
- Write: `S_t = S_{t-1} + k_t v_tᵀ` (outer product, adds one fact)
- Read: `o_t = Sᵀ q_t` (matrix-vector multiply, retrieves by similarity)

**The delta rule** subtracts the old content before writing, so repeated writes to the same key don't pile up stale values.

**Gating** multiplies all of S by a decay factor each step, so old content fades and capacity doesn't saturate.

**Gated DeltaNet** combines both. KDA is the next iteration.

---

## Section 2: The KDA Recurrence (Equation 1)

```
S_t = (I - β_t k_t k_tᵀ) Diag(α_t) S_{t-1}  +  β_t k_t v_tᵀ

o_t = S_tᵀ q_t
```

Three operations, left to right:

### Step 1: Decay — `Diag(α_t) S_{t-1}`

`α_t ∈ (0,1)^{d_k}` is a **vector**, not a scalar. `Diag(α_t)` is a diagonal matrix:

```
            ┌ α₁  0  0 ┐
Diag(α_t) = │ 0  α₂  0 │     (for d_k = 3)
            └ 0   0  α₃┘
```

Each row of S gets its own decay rate. Row 1 might decay at 0.99 (long memory), row 2 at 0.8 (short memory). This is the upgrade over Gated DeltaNet where one scalar α decayed everything uniformly.

**Why it matters:** the model can assign "fast-changing" state (character location, variable value) to dimensions with low α, and "stable" state (character name, data type) to dimensions with high α. One scalar α forces everything onto the same timescale.

### Step 2: Delta correction — `(I - β_t k_t k_tᵀ) · [decayed S]`

After decaying, subtract what's currently stored at key `k_t`:

```
(I - β k kᵀ) X  =  X - β k (kᵀ X)
                         └──────┘
                   what X currently returns for key k
```

This clears out the old content at address `k_t`, making room for the new value. β controls how aggressively: β=1 full erase, β=0 no erase.

### Step 3: Write — `+ β_t k_t v_tᵀ`

Add the new fact. `k_t v_tᵀ` is the outer product (the fact), scaled by β.

### Step 4: Read — `o_t = Sᵀ q_t`

Standard linear attention readout. Every stored fact contributes `v_i (k_i · q_t)`.

Note: their S is `d_k × d_v` (transposed from our GDN notes where it was `d_v × d_k`), so the read is `Sᵀ q` instead of `S q`. Same thing, just a convention flip.

---

## Section 3: Parameterization (Equation 2)

How each quantity is computed from the input token `x_t ∈ ℝ^d`:

### Query and Key

```
q, k = L2Norm( Swish( ShortConv( W_{q/k} x ) ) )
```

Three stages:
1. **Linear projection** `W_q x` — standard, maps d → d_k
2. **ShortConv** — a small 1D convolution (kernel size ~3-4) over the last few tokens. This gives each token a tiny window of local context before the recurrence even starts. During generation, this needs a buffer of the last ~3 token embeddings (not a KV cache — a tiny fixed FIFO buffer).
3. **Swish activation** — `x · sigmoid(x)`, adds non-linearity
4. **L2Norm** — normalize to unit length. This is what makes `k · k = 1`, which the delta rule needs for clean erasure.

### Value

```
v = Swish( ShortConv( W_v x ) )
```

Same pipeline but no L2Norm — values don't need to be unit vectors.

### Beta (write strength)

```
β = sigmoid( W_β x ) ∈ (0, 1)
```

One scalar per head. Simple: a linear projection + sigmoid.

### Alpha (decay) — the interesting one

```
z = W_↑ W_↓ x + b_α ∈ ℝ^{d_k}          (one logit per key dimension)
```

Two things to notice:

**Low-rank projection.** `W_↓` maps d → r (some small rank), `W_↑` maps r → d_k. This is like LoRA — fewer parameters than a full d → d_k projection, but enough expressivity. The intuition: you don't need d_k independent decay decisions; a few basis "decay patterns" combined by the input are enough.

**Per-head bias `b_α`.** Each head gets its own bias vector, initialized so that different heads default to different decay timescales. Some heads are born "long memory", others "short memory", and the data-dependent part `W_↑ W_↓ x` adjusts from there.

The mapping from logit z to actual decay α is covered in Section 5 (lower-bounded decay).

---

## Section 4: Chunkwise Parallel Form (Equations 3-4)

This is the section that makes it fast. The sequential loop processes one token at a time — O(T) steps, each waiting for the previous one. On a GPU that's terrible: you're using one thread while thousands sit idle.

The fix: split the sequence into **chunks** of C tokens (typically C=64). Within each chunk, compute all outputs with matrix multiplications (parallel, uses tensor cores). Between chunks, pass the state S forward (sequential, but only T/C steps instead of T).

### 4.1: Deriving the chunkwise form — gated linear attention (no delta first)

Start with the simpler case to build intuition. No delta rule, just decay + write:

```
S_t = Diag(α_t) S_{t-1} + k_t v_tᵀ
o_t = Sᵀ_t q_t
```

#### Unrolling within a chunk

Say chunk 2 covers tokens 3 and 4, and enters with state S[2] (the state after chunk 1). What is S at token 4?

```
S₃ = Diag(α₃) S[2] + k₃ v₃ᵀ
S₄ = Diag(α₄) S₃ + k₄ v₄ᵀ
   = Diag(α₄) Diag(α₃) S[2] + Diag(α₄) k₃ v₃ᵀ + k₄ v₄ᵀ
   = Diag(α₄α₃) S[2] + Diag(α₄) k₃ v₃ᵀ + k₄ v₄ᵀ
```

Pattern: each token j's contribution `k_j v_jᵀ` gets decayed by the product of all α's from j+1 to the current position i.

#### Cumulative decay (Equation 3)

Define the cumulative decay from position 1 (start of chunk) to position i:

```
γ_i = α₁ · α₂ · ... · α_i          (element-wise products, each is a vector in ℝ^{d_k})
```

The decay from position j+1 to position i is then:

```
γ^{j+1→i} = γ_i / γ_j              (element-wise division)
```

Stack all γ_i as rows of a matrix `Γ ∈ ℝ^{C × d_k}`:

```
     ┌ γ₁ ┐     ┌ α₁                ┐
Γ =  │ γ₂ │  =  │ α₁·α₂             │      (each row is d_k-dimensional)
     └    ┘     └                    ┘
```

**Worked example** (C=2, d_k=2, α₃=[0.9,0.9], α₄=[0.8,0.7]):

```
γ₁ = [0.9, 0.9]
γ₂ = [0.9·0.8, 0.9·0.7] = [0.72, 0.63]

Γ = ┌ 0.9   0.9  ┐
    └ 0.72  0.63 ┘
```

#### The output formula

The output at position i in the chunk has two parts:

**Inter-chunk** — what S[t] contributes (information from all previous chunks):
```
o_i^{inter} = S[t]ᵀ (γ_i ⊙ q_i)
```

The query is element-wise scaled by the cumulative decay: content from S[t] has been decaying through all positions up to i.

In matrix form for all positions in the chunk:
```
O^{inter} = (Γ ⊙ Q) S[t]          (C×d_k) × (d_k×d_v) = (C×d_v)
```

**Intra-chunk** — what tokens within this chunk contribute to each other:
```
o_i^{intra} = Σ_{j=1}^{i} a_{i,j} · v_j
```

where the attention weight between positions i and j (within the chunk) is:

```
a_{i,j} = q_iᵀ Diag(γ^{j+1→i}) k_j
```

This is just: query dot key, but the key is scaled by how much it's decayed since it was written.

Using the identity `γ^{j+1→i} = γ_i / γ_j`:

```
a_{i,j} = (γ_i ⊙ q_i)ᵀ (k_j / γ_j)
```

In matrix form:
```
A = Tril( (Γ ⊙ Q)(K / Γ)ᵀ )       (C×C) causal attention matrix
```

The `Tril` zeros out the upper triangle — position i can't attend to position j > i.

The intra-chunk output:
```
O^{intra} = A V                     (C×C) × (C×d_v) = (C×d_v)
```

**Full output:**
```
O = (Γ ⊙ Q) S[t]  +  Tril( (Γ⊙Q)(K/Γ)ᵀ ) V
    ─────────────     ────────────────────────
     inter-chunk            intra-chunk
```

#### Worked example (verified numerically)

Data (chunk 2, tokens 3-4, d_k = d_v = 2):

```
k₃ = [0, 1]      v₃ = [2, 4]      q₃ ≈ [0.707, 0.707]     α₃ = [0.9, 0.9]
k₄ = [0.8, 0.6]  v₄ = [7, 0]      q₄ = [1, 0]             α₄ = [0.8, 0.7]
```

State entering chunk 2 (from sequential computation):
```
S[2] = ┌ 1.889   3.150 ┐
       └-0.142   3.313 ┘
```

Cumulative decay:
```
Γ = ┌ 0.9   0.9  ┐
    └ 0.72  0.63 ┘
```

Attention matrix:
```
Γ⊙Q = ┌ 0.636  0.636 ┐      K/Γ = ┌ 0.000  1.111 ┐
      └ 0.720  0.000 ┘            └ 1.111  0.952 ┘

A_full = (Γ⊙Q)(K/Γ)ᵀ = ┌ 0.707  1.313 ┐
                        └ 0.000  0.800 ┘

A = Tril(A_full) = ┌ 0.707  0.000 ┐      ← upper triangle zeroed
                   └ 0.000  0.800 ┘
```

Outputs:
```
Inter = (Γ⊙Q) S[2] = ┌ 1.111  4.110 ┐     (reading from previous chunks)
                      └ 1.360  2.268 ┘

Intra = A V = ┌ 1.414  2.828 ┐              (reading within this chunk)
              └ 5.600  0.000 ┘

O = Inter + Intra = ┌ 2.525  6.939 ┐        (should be ≈ but won't match sequential
                    └ 6.960  2.268 ┘         exactly because this ignores the delta rule)
```

This matches the sequential no-delta computation (verified by code: match = True).

### 4.2: Adding the delta rule — pseudo-values Ṽ

The delta rule modifies what gets written. Instead of writing `v_t` directly, you write the *error*: `β_t(v_t - current content at k_t)`.

In the chunkwise form, this correction splits into two parts:

**Correction against S[t]** (inter-chunk state): what does S[t] currently return at key k_t? This can be computed upfront since S[t] is known when the chunk starts.

**Correction against intra-chunk writes**: token 4's delta correction should also account for token 3's write within the same chunk. This creates within-chunk dependencies.

The **UT transform** (from Kimi Linear, 2025) handles both. It produces two matrices:
- `U ∈ ℝ^{C × d_v}` — the scaled values to write (includes intra-chunk interactions)
- `W ∈ ℝ^{C × d_k}` — the scaled keys for reading the inter-chunk state

The **pseudo-values** combine them:

```
Ṽ = U - W S[t]                      (C × d_v)
```

Row i of Ṽ is: "what I want to write **minus** what's already stored in S[t] at my key." This is the delta — the correction — folded into a modified "value" that the standard chunkwise attention can use.

The final chunkwise formula (Equation 4):

```
O = (Γ ⊙ Q) S[t]  +  A Ṽ
    ─────────────     ────
     inter-chunk    intra-chunk (with delta correction baked in)
```

Same structure as before. The only change: V is replaced by Ṽ. The triangular attention matrix A handles causal masking within the chunk, and because Ṽ already encodes the delta corrections, the outputs account for both the outer-product writes and the subtractive edits.

### 4.3: Diagonal retention in A

The paper says: *"the diagonal is retained because each output reads the state after the current-token update."*

In standard causal attention, position i can attend to positions 1..i (including itself). Same here: the output at position i reflects the state *after* token i's own write. So the diagonal of A is non-zero — `a_{i,i}` accounts for token i attending to its own contribution.

### 4.4: State update between chunks

After processing all outputs within a chunk, update S for the next chunk:

```
S[t+1] = Diag(γ^{1→C}) S[t] + [accumulated writes within this chunk]
```

The accumulated writes come from the UT transform. This is one matmul, done once per chunk.

### 4.5: Computational cost

| operation | size | hardware |
|---|---|---|
| inter-chunk: (Γ⊙Q) S[t] | (C × d_k) × (d_k × d_v) | matmul, tensor cores |
| intra-chunk: A Ṽ | (C × C) × (C × d_v) | matmul, tensor cores |
| A itself: (Γ⊙Q)(K/Γ)ᵀ | (C × d_k) × (d_k × C) | matmul, tensor cores |
| state update | (d_k × d_v) | matmul, tensor cores |

Everything is a matrix multiply. The sequential bottleneck (T steps) is reduced to T/C state updates (~T/64), while all within-chunk work is parallel.

---

## Section 5: Lower-Bounded Decay (Equation 5)

This section solves a numerical problem in the chunkwise form.

### The problem

Look at the intra-chunk attention matrix:

```
A = Tril( (Γ ⊙ Q)(K / Γ)ᵀ )
                   ^^^^^
                   division by Γ
```

Γ is a cumulative product of α values, all in (0,1). Over many positions, this product gets very small. Dividing K by a tiny number gives a huge number. In BF16 (which has limited range), this overflows.

Concrete: if α_min = e^{-20} (an unrestricted negative-softplus can produce this), then over 16 positions:
```
γ = (e^{-20})^{16} = e^{-320} ≈ 10^{-139}
1/γ ≈ 10^{139}   →   overflow in BF16 (max ≈ 3.4 × 10^{38})
```

### Kimi Linear's workaround

Compute everything in log space. Split each chunk into secondary 16-token **tiles**. Off-diagonal tiles (tile i attends to tile j where i ≠ j) can be computed with standard matmuls because the relative decay between tiles is manageable. But **diagonal tiles** (tile attending to itself) need special position-pair computation — slower, can't use tensor cores fully.

### K3's fix: bound α from below

Instead of mapping logit z to decay via unbounded negative-softplus:

```
Kimi Linear:   g = -exp(A) · softplus(z)  ∈  (-∞, 0)     ← log-decay can be arbitrarily negative
```

K3 uses a bounded sigmoid:

```
Kimi K3:       g = g_min · sigmoid(exp(A) · z)  ∈  (g_min, 0)    where g_min = -5

               α = exp(g)  ∈  (exp(g_min), 1)  =  (0.0067, 1)
```

With g_min = -5, the worst case over a 16-token tile:
```
γ = (e^{-5})^{16} = e^{-80} ≈ 1.8 × 10^{-35}
1/γ = e^{80} ≈ 5.5 × 10^{34}   →   fits in BF16 ✓
```

**Consequence:** the reciprocal never overflows, so ALL tiles (diagonal and off-diagonal) can use dense tensor-core matmuls. The special-case diagonal path is eliminated entirely. Same math, faster kernels.

### The learnable scale A

`exp(A)` in the sigmoid scales how sensitive the decay is to z. Initialized at A=0 (so exp(A)=1, sigmoid is standard). Each head learns its own A, allowing some heads to have sharp decay transitions and others to have smooth ones.

### Is g_min = -5 limiting?

α_min = e^{-5} ≈ 0.0067. That means the fastest possible decay per step is a 99.3% reduction. Over 16 tokens the state can drop to 10^{-35} of its original. In practice, you rarely want faster decay than this — and you get tensor-core speed on every tile in exchange.

---

## Section 6: Output Gate (Equation 6)

After the recurrence produces output õ_t, a gating layer cleans it up:

```
y_t = W_o [ sigmoid(W_g x_t) ⊙ RMSNorm(õ_t) ]
```

Three sub-steps:

### RMSNorm

```
RMSNorm(x) = x / √(mean(x²))
```

Normalizes the scale of the recurrent output. Head-wise: applied independently per head. This stabilizes training because the recurrent state S can accumulate scale drift over long sequences.

### Sigmoid gate

```
gate = sigmoid(W_g x_t)     ∈ (0,1)^{d_model}
```

**Input-dependent** and **full-rank** — a full d_model × d_model matrix, not low-rank. Each output dimension gets its own gate value, controlled by the current input. This lets the model suppress dimensions that aren't relevant to the current token.

Kimi Linear used a low-rank gate here. K3 upgraded to full-rank, meaning more parameters but more expressive gating.

### Element-wise multiply + output projection

```
y_t = W_o (gate ⊙ normed_output)
```

The gated, normalized output is mixed across heads by the output projection W_o.

**Why gate the output at all?** The recurrent output õ is a noisy sum of many stored facts (the interference we discussed in the GDN notes). The gate learns to suppress the noise dimensions and pass through the signal dimensions, conditioned on what the current token actually needs.

---

## Section 7: Summary — What KDA Changes Over Gated DeltaNet

| aspect | Gated DeltaNet | KDA (Kimi K3) |
|---|---|---|
| decay α | one scalar per head | one value per key dimension per head |
| α parameterization | negative softplus (unbounded) | bounded sigmoid, g_min = -5 |
| diagonal-tile computation | position-pair (scalar) | dense tensor-core matmul |
| output gate | low-rank | full-rank input-dependent |
| delta rule | same | same |
| β (write strength) | same | same |
| ShortConv on q,k,v | same | same |

The delta rule itself is unchanged. The upgrades are about making the decay more expressive (channel-wise), making the GPU kernels faster (bounded decay), and making the output cleaner (full-rank gate).

---

## Appendix: Notation Quick Reference

| symbol | shape | meaning |
|---|---|---|
| x_t | d | input token embedding |
| q_t, k_t | d_k | query and key (L2-normalized) |
| v_t | d_v | value |
| S_t | d_k × d_v | recurrent state (the "compressed KV cache") |
| α_t | d_k | channel-wise decay (per key dimension) |
| β_t | scalar | write strength for delta rule |
| o_t / õ_t | d_v | raw output before gating |
| γ_i | d_k | cumulative decay from chunk start to position i |
| Γ | C × d_k | stacked γ vectors for all positions in a chunk |
| A | C × C | intra-chunk causal attention matrix |
| Ṽ | C × d_v | pseudo-values (delta-corrected) |
| g_t | d_k | log-decay: g = g_min · sigmoid(eᴬz), α = eᵍ |
| C | scalar | chunk size (typically 64) |