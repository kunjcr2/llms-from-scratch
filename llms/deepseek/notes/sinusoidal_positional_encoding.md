# Sinusoidal Positional Encoding

> Building block for understanding **Rotary Positional Encoding (RoPE)**, which DeepSeek couples with Multi-Head Latent Attention.
> Introduced in the original **"Attention Is All You Need"** paper (Vaswani et al., 2017).

---

## 1. Why Positional Encoding?

Transformers have no inherent notion of token order — unlike RNNs which process tokens sequentially. We need to **inject position information** so the model can distinguish "the dog bit the man" from "the man bit the dog."

From the paper:
> *"We must inject some information about the relative or absolute position of tokens in the sequence."*

---

## 2. Previous Attempts (Recap)

### Integer Positional Encoding

Encode each token's position as a repeated integer:

```
Token "dog" at position 200:
    Token embedding:      [0.12, -0.34, 0.56, 0.01, ...]   ← values near 0
    Position embedding:   [200,   200,  200,  200,  ...]    ← values huge!
```

**Problem**: Position values are orders of magnitude larger than token embeddings → **dilutes/pollutes semantic information**. The meaning of "dog" gets drowned out by the position number.

### Binary Positional Encoding

Convert position to binary and use as the encoding vector:

```
Position 200 (8-bit): [1, 1, 0, 0, 1, 0, 0, 0]
```

**Key observation**: When plotting bit values across positions:
- **Lower indexes** (least significant bits) → oscillate **fast** between positions
- **Higher indexes** (most significant bits) → oscillate **slow** between positions

```
Position:  0  1  2  3  4  5  6  7
Bit 0:     0  1  0  1  0  1  0  1    ← changes every 1 position (fastest)
Bit 1:     0  0  1  1  0  0  1  1    ← changes every 2 positions
Bit 2:     0  0  0  0  1  1  1  1    ← changes every 4 positions (slowest)
```

**Advantage**: Values constrained to {0, 1} — no magnitude explosion.

**Problem**: Values are **discrete jumps** (0 or 1, nothing in between). These discontinuities make LLM optimization unstable during backpropagation — gradients can't flow smoothly through hard jumps.

---

## 3. Sinusoidal Positional Encoding: The Formula

> **Goal**: Keep the binary encoding intuition (lower indexes = fast oscillation, higher indexes = slow oscillation) but make it **continuous and differentiable**.

### The Formula

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))     ← even indexes
PE(pos, 2i+1)   = cos(pos / 10000^(2i/d_model))     ← odd indexes
```

Where:
- `pos` = position of the token in the sequence (range: 0 to context_size - 1)
- `i` = index within the embedding vector (range: 0 to d_model/2 - 1)
- `d_model` = model embedding dimension (e.g., 768 for GPT-2)

### Reading the Formula

For a given position, the positional embedding is a vector of size `d_model`:

```
GPT-2 example: d_model = 768, context_size = 1024

Position 2 → 768-dimensional vector:
    Index 0 (even):  sin(2 / 10000^(0/768))
    Index 1 (odd):   cos(2 / 10000^(0/768))
    Index 2 (even):  sin(2 / 10000^(2/768))
    Index 3 (odd):   cos(2 / 10000^(2/768))
    ...
    Index 767 (odd): cos(2 / 10000^(766/768))
```

### Frequency Interpretation

The formula can be rewritten as:

```
PE(pos, 2i) = sin(ω · pos)

where ω = 1 / 10000^(2i/d_model)    ← frequency of oscillation
```

**Why `i` is in the denominator (exponent)**:
- Low `i` → large `ω` → **high frequency** → fast oscillation
- High `i` → small `ω` → **low frequency** → slow oscillation

This preserves the exact same property as binary encoding!

**Why `10,000`?**
- Experimental choice from the paper
- With `d_model ≈ 1000`, using 10,000 ensures frequencies die down gradually across indexes
- Too small → frequencies die too fast; too large → frequencies die too slow
- 10,000 gives an optimal range for typical model dimensions

---

## 4. Visualizing the Formula

For GPT-2 (`d_model = 768`, context = 1024), plotting PE values across positions for fixed indexes:

```
Index i=1:    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿    ← extremely rapid oscillation
Index i=50:   ∿  ∿  ∿  ∿  ∿  ∿       ← moderate oscillation  
Index i=150:  ∿      ∿      ∿         ← slow, gentle curves
```

### What This Confirms

| Property | Binary Encoding | Sinusoidal Encoding |
|----------|----------------|-------------------|
| Lower indexes oscillate faster | ✅ | ✅ |
| Higher indexes oscillate slower | ✅ | ✅ |
| Values bounded | {0, 1} | [-1, +1] |
| Continuous | ❌ (discrete jumps) | ✅ (smooth curves) |
| Differentiable | ❌ | ✅ |

> Sinusoidal encoding captures the **same intuition** as binary encoding but on a **continuous, differentiable** spectrum — enabling stable gradient-based optimization.

---

## 5. Why Sine AND Cosine? — The Rotation Property

This is the most important insight and the **bridge to Rotary Positional Encodings (RoPE)**.

### The Desired Property

> When constructing positional encodings, we want a **linear relation between encoded positions**: given the encoding at position `p`, it should be straightforward to compute the encoding at position `p + k`.

This way, the transformer can learn that positions are **related** to each other in a predictable, mathematical way.

### How Sin/Cos Enable Rotation

Consider a pair of consecutive indexes `(2i, 2i+1)` at position `p`:

```
PE(p, 2i)   = sin(θ)     where θ = p / 10000^(2i/d_model) = ω·p
PE(p, 2i+1) = cos(θ)
```

This forms a 2D vector `v₁ = (cos θ, sin θ)` — a point on the unit circle.

Now, to find the encoding at position `p + k`:

```
PE(p+k, 2i)   = sin(θ + θ₁)     where θ₁ = ω·k
PE(p+k, 2i+1) = cos(θ + θ₁)
```

This is just `v₂ = (cos(θ + θ₁), sin(θ + θ₁))` — **a rotation of v₁ by angle θ₁**.

```
        y (sin)
        ↑
        |    · v₂ = rotation of v₁ by θ₁
        |   /
        |  /  θ₁ (rotation angle = ω·k)
        | /
        |/ · v₁ at position p
        +————————→ x (cos)
```

### The Key Insight

> **Relative positional encodings are just rotations of each other.**

- To go from position `p` to position `p + k`: rotate by angle `ω·k`
- The rotation angle depends only on the **offset** `k`, not on the absolute position
- This is only possible because we use **both sine and cosine** — you need both components to define a rotation in 2D

### Rotation Matrix Form

```
[PE(p+k, 2i)  ]   [cos(θ₁)  -sin(θ₁)] [PE(p, 2i)  ]
[PE(p+k, 2i+1)] = [sin(θ₁)   cos(θ₁)] [PE(p, 2i+1)]
```

This is a standard 2D rotation matrix applied to pairs of dimensions.

---

## 6. Advantages of Sinusoidal Encoding

| Advantage | Explanation |
|-----------|-------------|
| **Continuous & differentiable** | Smooth sin/cos curves → stable backpropagation |
| **Bounded values** | Always in [-1, +1] → doesn't overwhelm token embeddings |
| **Same intuition as binary** | Lower indexes = fast oscillation, higher = slow |
| **Relative positions are rotations** | Linear relationship between positions → transformer can learn patterns |
| **No learned parameters** | Deterministic formula, no training needed for the encoding itself |

---

## 7. The Problem: Why We Still Need RoPE

Despite solving the discontinuity problem, sinusoidal encoding has a fundamental flaw:

> **Positional embeddings are directly ADDED to token embeddings, which pollutes semantic information.**

```
Input to transformer = Token_embedding + Position_embedding
                       ↑ semantic info     ↑ position info
                       └── these get mixed/contaminated ──┘
```

Even though PE values are small (bounded to [-1, +1]), the addition still modifies the token embedding vector, changing its magnitude and direction. The semantic meaning of "dog" is slightly different at position 5 vs position 500.

### Two Key Realizations That Lead to RoPE

**Realization 1**: Positional information matters most in the **attention mechanism** (Q × K^T). That's where the influence of one token on another is computed. So instead of encoding position in the token embeddings, why not encode it **directly in Q and K vectors**?

**Realization 2**: Instead of **adding** a position vector (which changes magnitude), why not **rotate** the Q and K vectors? Rotation changes the direction but **preserves magnitude** — the original vector's "strength" is untouched.

```
Sinusoidal:  Q = (X + PE) × W_Q     ← PE contaminates X before Q is computed
RoPE:        Q = X × W_Q, then ROTATE Q based on position
             ↑ semantic info preserved, position encoded via rotation
```

### Summary of the Progression

```
Integer Encoding
├── Problem: Values too large, pollute semantics
│
├── Binary Encoding
│   ├── Fixed: Values bounded to {0, 1}
│   ├── Key insight: Lower bits oscillate faster
│   └── Problem: Discrete jumps, non-differentiable
│
├── Sinusoidal Encoding
│   ├── Fixed: Continuous, differentiable
│   ├── Key insight: sin/cos enable ROTATION between positions
│   └── Problem: Still ADDED to token embeddings → pollutes semantics
│
└── Rotary Positional Encoding (RoPE)    ← NEXT
    ├── Fixed: Applied to Q and K directly, not token embeddings
    ├── Rotation preserves vector magnitude
    └── Used by DeepSeek, LLaMA, and most modern LLMs
```

---

## 8. Summary

```
Sinusoidal Positional Encoding:
│
├── Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))
│            PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
│
├── Two variables:
│   ├── pos → position of token (0 to context_size - 1)
│   └── i   → index in embedding vector (0 to d_model/2 - 1)
│
├── Frequency: ω = 1/10000^(2i/d)
│   ├── Low i → high ω → fast oscillation (fine-grained encoding)
│   └── High i → low ω → slow oscillation (coarse encoding)
│
├── Why sin AND cos?
│   └── Enables ROTATION between position encodings
│       └── PE(p+k) = Rotation_matrix(ω·k) × PE(p)
│       └── Relative positions → fixed angular differences → learnable patterns
│
├── Why 10,000?
│   └── Experimental choice for optimal frequency decay across typical d_model (~768-1024)
│
├── Advantages over binary:
│   ├── Continuous and differentiable
│   └── Stable LLM optimization
│
└── Limitation:
    └── Added to token embeddings → pollutes semantic information
    └── Solved by RoPE (next lecture)
```

---

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Section 3.5: Positional Encoding
- [Build DeepSeek from Scratch — Sinusoidal Positional Encoding Lecture (Dr. Raj Dandkar)]()
- [kv_cache.md](./kv_cache.md) — Prerequisite on KV Cache
- [multi_head_latent_attention.md](./multi_head_latent_attention.md) — MLA (uses RoPE)
- [`deepseek.md`](../deepseek.md) — Point 8 for encoding overview, Point 9 for MLA + RoPE integration
