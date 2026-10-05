# Rotary Positional Encoding (RoPE)

> The positional encoding used by **DeepSeek V3/R1**, **LLaMA**, and most modern LLMs.
> RoPE injects position information by **rotating** query and key vectors — preserving magnitude and avoiding semantic pollution.
> Introduced in the **RoFormer** paper (Su et al., 2023), building on sinusoidal encoding from "Attention Is All You Need" (2017).

---

## 1. Why Not Sinusoidal? (Motivation)

Sinusoidal positional encoding (see [sinusoidal_positional_encoding.md](./sinusoidal_positional_encoding.md)) solves the continuity problem but has a fundamental flaw:

```
Traditional approach:
    Input to transformer = Token_embedding + Position_embedding
                           ↑ semantics        ↑ position info
                           └── ADDED together → pollutes semantics ──┘
```

**Two key realizations** that lead to RoPE:

### Realization 1: Encode Position in Q and K, Not Token Embeddings

The attention mechanism (`Q × K^T`) is where the influence of one token on another is computed — that's where position **actually matters**. So instead of adding position info to token embeddings in the preprocessing step, inject it **directly into the query and key vectors** inside the attention block.

```
Sinusoidal:  Token_emb + PE → Transformer → Q, K, V → Attention
RoPE:        Token_emb → Transformer → Q, K → ROTATE by position → Attention
```

### Realization 2: Rotate Instead of Add

Adding a vector to Q or K **changes its magnitude**. But if we **rotate** the vector, individual values change but the magnitude stays the same. Position is encoded in the **direction** of the vector, not its length.

```
Adding:    |Q + PE| ≠ |Q|     ← magnitude changes
Rotating:  |Rotate(Q, θ)| = |Q|  ← magnitude preserved ✅
```

---

## 2. How RoPE Works: The Visual Intuition

### Step 1: Split into Pairs

Take a query (or key) vector and split it into **groups of 2** consecutive elements:

```
Query vector for "the" (4-dim): [x₁, x₂, x₃, x₄]

Group 1: (x₁, x₂)    ← index i = 0
Group 2: (x₃, x₄)    ← index i = 1
```

### Step 2: Treat Each Pair as a 2D Vector

Each pair `(xₐ, xᵦ)` is a point in 2D space:

```
        y (x₂)
        ↑
        |    · (x₁, x₂)
        |   /
        |  /  ← original vector
        | /
        +————————→ x (x₁)
```

### Step 3: Rotate by Angle θ

Rotate each 2D vector by an angle that encodes position:

```
θ = ω_i × p = p / 10000^(2i/d)

where:
    p = position of the token (0 to context_size - 1)
    i = pair index (0, 1, 2, ...)
    d = embedding dimension
```

```
        y
        ↑
        |  · (x₁', x₂') ← rotated vector
        | /
        |/ θ (rotation angle)
        ·————→ · (x₁, x₂) ← original vector
        +————————→ x

|rotated vector| = |original vector|  ✅ magnitude preserved
```

The rotated values become:

```
[x₁']   [cos θ  -sin θ] [x₁]
[x₂'] = [sin θ   cos θ] [x₂]
```

This is the standard 2D rotation matrix — the same one from sinusoidal encoding.

### Step 4: Repeat for All Pairs

Each pair uses a **different index `i`**, so each pair rotates by a different angle:

```
Original:  [x₁,  x₂,  x₃,  x₄]
            ╰──────╯   ╰──────╯
            Group 1     Group 2
            i = 0       i = 1
            θ₀ = p/10000^(0/d)    θ₁ = p/10000^(2/d)

Rotated:   [x₁', x₂', x₃', x₄']
```

### Full Picture

```
Token "the" at position p=1, dimension d=4:

Query vector: [x₁, x₂, x₃, x₄]
                 ↓         ↓
              Pair 1    Pair 2
              (x₁,x₂)  (x₃,x₄)
                 ↓         ↓
            Rotate by   Rotate by
            θ₀=ω₀·1    θ₁=ω₁·1
                 ↓         ↓
              (x₁',x₂') (x₃',x₄')
                 ↓         ↓
RoPE'd vector: [x₁', x₂', x₃', x₄']

Key properties:
    |(x₁', x₂')| = |(x₁, x₂)|   ✅ pair magnitude preserved
    |(x₃', x₄')| = |(x₃, x₄)|   ✅ pair magnitude preserved
```

> The same operation is applied to **both** query vectors AND key vectors. The values matrix V is left untouched.

---

## 3. The Rotation Angle: Two Variables

```
θ = p / 10000^(2i/d)
    ↑              ↑
    position       index (via frequency)
```

### Effect of Position (p)

For a **fixed index**, as position increases → θ increases → **larger rotation**.

```
Position 1:  small rotation  ·→
Position 2:  medium rotation ·→→
Position 3:  larger rotation ·→→→
Position 5:  even larger     ·→→→→→
```

**Intuition**: Tokens that are **close together in position** get similar rotations → similar positional encodings. Tokens **far apart** get very different rotations.

> From the RoFormer paper: *"The inner product will decay when relative position increases. This property coincides with the intuition that a pair of tokens with a long relative distance should have less connection."*

### Effect of Index (i)

For a **fixed position**, as index increases → frequency ω decreases → **smaller rotation angle**.

```
Index i=0:   large ω → fast change across positions  (high frequency)
Index i=1:   medium ω → moderate change              (medium frequency)
Index i=2:   small ω → slow change across positions   (low frequency)
```

This is the **same property** as binary and sinusoidal encoding: lower indexes oscillate faster, higher indexes oscillate slower.

---

## 4. What Do Lower vs Higher Indexes Capture?

### Lower Indexes → Fast Oscillation → Fine-Grained / Local Patterns

Lower indexes change rapidly between adjacent positions. This captures **small positional shifts** that change meaning:

```
"I just told her the truth"     ← "just" modifies timing (recently)
"I told just her the truth"     ← "just" modifies target (only her)
```

The word "told" is at position 3 vs position 2 — a tiny shift. Lower index frequencies oscillate fast enough to distinguish these nearby positions.

### Higher Indexes → Slow Oscillation → Long-Range Dependencies

Higher indexes barely change even across large position differences. This preserves relationships between **distant tokens**:

```
"Einstein developed the theory of relativity. This breakthrough reshaped physics."
 ↑ position 1                                  ↑ position ~10
 └──────────── "breakthrough" refers to "theory of relativity" ──────────────┘
```

Higher indexes have similar values at position 1 and position 10, so they can encode that these distant tokens are related. Lower indexes would have oscillated too wildly between these positions to preserve the relationship.

| Index Level | Oscillation | Captures |
|-------------|------------|----------|
| **Low i** | Fast (high ω) | Small positional shifts, local word order |
| **High i** | Slow (low ω) | Long-range dependencies, distant token relationships |

---

## 5. RoPE vs Sinusoidal Encoding

| Property | Sinusoidal | RoPE |
|----------|-----------|------|
| **Where applied** | Added to token embeddings (preprocessing) | Applied to Q and K vectors (inside attention) |
| **How applied** | Vector addition | Vector rotation |
| **Magnitude change** | Yes (addition changes magnitude) | No (rotation preserves magnitude) |
| **Semantic pollution** | Yes (modifies token embeddings) | No (token embeddings enter transformer unmodified) |
| **Same frequency formula** | `ω = 1/10000^(2i/d)` | `ω = 1/10000^(2i/d)` ✅ same |
| **Lower indexes = fast oscillation** | ✅ | ✅ |
| **Rotation insight** | Inspired the idea (positions are rotations of each other) | Directly implements rotations |

> RoPE builds on sinusoidal encoding's insight that relative positions are rotations — but applies this rotation **directly to Q and K** instead of adding PE to token embeddings.

---

## 6. The Rotation Matrix

For each pair of dimensions, the rotation is:

```
[q₂ᵢ'  ]   [cos(mθᵢ)  -sin(mθᵢ)] [q₂ᵢ  ]
[q₂ᵢ₊₁'] = [sin(mθᵢ)   cos(mθᵢ)] [q₂ᵢ₊₁]

where:
    m = position of the token
    θᵢ = 1/10000^(2i/d)
```

For a full d-dimensional vector, this is a **block-diagonal** rotation matrix:

```
[cos θ₀  -sin θ₀   0        0       0        0    ] [q₀  ]
[sin θ₀   cos θ₀   0        0       0        0    ] [q₁  ]
[0        0       cos θ₁  -sin θ₁   0        0    ] [q₂  ]
[0        0       sin θ₁   cos θ₁   0        0    ] [q₃  ]
[0        0       0        0       cos θ₂  -sin θ₂] [q₄  ]
[0        0       0        0       sin θ₂   cos θ₂] [q₅  ]

Each 2×2 block is an independent rotation for one pair of dimensions.
```

---

## 7. Summary

```
Rotary Positional Encoding (RoPE):
│
├── Motivation:
│   ├── Sinusoidal PE pollutes token embeddings (addition changes magnitude)
│   └── Position matters most in attention (Q × K^T), not in preprocessing
│
├── Core Idea:
│   ├── Split Q and K vectors into pairs of consecutive dimensions
│   ├── Treat each pair as a 2D vector
│   ├── ROTATE each pair by angle θ = p / 10000^(2i/d)
│   └── Magnitude is preserved (rotation ≠ addition)
│
├── Rotation angle θ depends on:
│   ├── Position p: higher position → larger rotation
│   │   └── Nearby tokens get similar encodings, distant tokens differ
│   └── Index i: higher index → smaller rotation (lower frequency)
│       ├── Low i: fast oscillation → captures local word order shifts
│       └── High i: slow oscillation → captures long-range dependencies
│
├── Applied to:
│   ├── Query vectors ✅
│   ├── Key vectors ✅
│   └── Value vectors ❌ (untouched)
│
├── Key advantages over sinusoidal:
│   ├── No semantic pollution (token embeddings unmodified)
│   ├── Magnitude preserved (rotation, not addition)
│   └── Position encoded where it matters (Q and K in attention)
│
└── Used by: DeepSeek V3/R1, LLaMA, and most modern LLMs
    └── Next: How RoPE integrates with Multi-Head Latent Attention (MLA + RoPE)
```

---

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2023)](https://arxiv.org/abs/2104.09864)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — sinusoidal encoding origin
- [sinusoidal_positional_encoding.md](./sinusoidal_positional_encoding.md) — prerequisite
- [multi_head_latent_attention.md](./multi_head_latent_attention.md) — MLA (next: MLA + RoPE integration)
- [`deepseek.md`](../deepseek.md) — Point 8 (encoding overview), Point 9 (MLA + RoPE)
