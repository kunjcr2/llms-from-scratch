# 🧭 ALiBi (Attention with Linear Biases)

## What is ALiBi?

ALiBi is a method for encoding positional information in Transformers.  
Instead of giving tokens explicit positional embeddings (like sinusoidal or RoPE),  
it adds a **linear bias** to the attention scores based on how far apart two tokens are.

👉 This makes models trained on shorter sequences (e.g., 2k tokens) **generalize naturally** to much longer ones (100k+).

---

## Intuition

- Nearby tokens are usually more relevant (e.g., “today” relates to “yesterday” more than “10 years ago”).
- ALiBi enforces this by **penalizing distant tokens linearly** in the attention mechanism.
- Embeddings stay the same — only the **attention score matrix** is adjusted.

---

## Example

### Step 1: Normal Attention Scores (Q·Kᵀ)

Let’s say we have 3 tokens:

- T1 (pos=1), T2 (pos=2), T3 (pos=3)

Attention scores before ALiBi:

```
    T1  T2  T3
T1 1.0 2.0 3.0
T2 2.0 1.5 2.5
T3 3.0 2.5 1.2

```

---

### Step 2: Compute ALiBi Penalties

Let slope = **-0.2**.
Penalty = slope × distance.

```
    T1  T2  T3
T1 0.0 -0.2 -0.4
T2 -0.2 0.0 -0.2
T3 -0.4 -0.2 0.0

```

---

### Step 3: Add Penalties to Scores

```
    T1  T2  T3
T1 1.0 1.8 2.6
T2 1.8 1.5 2.3
T3 2.6 2.3 1.2

```

---

## ✅ Key Takeaways

- **No embeddings needed** for position, just a bias on attention.
- **Linear penalty** means far tokens aren’t ignored, but are less important.
- **Scales gracefully** → models can handle much longer contexts than they were trained on.

👉 Used in models like **LLaMA** and many long-context LLMs.

---

# CODE FOR ALIBI ENCODINGS

```python
import torch
import torch.nn as nn

class AlibiBias(nn.Module):
    """
    Builds ALiBi bias matrices for attention.
    Returns a tensor of shape [1, n_heads, q_len, k_len].
    """
    def __init__(self, n_heads: int, causal: bool = True, slopes: torch.Tensor | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.causal = causal

        # If slopes not given, create linearly spaced negative slopes for heads
        if slopes is None:
            slopes = torch.linspace(-0.05, -0.5, steps=n_heads)

        # Ensure slopes shape matches number of heads
        assert slopes.shape == (n_heads,), "slopes must be shape [n_heads]"

        # Register slopes as non-trainable buffer
        self.register_buffer("slopes", slopes, persistent=False)

        # Cache to avoid rebuilding if lengths repeat
        self._cached_len = None
        self._cached_bias = None

    @torch.no_grad()
    def build(self, q_len: int, k_len: int, device=None, dtype=None) -> torch.Tensor:
        """
        Build ALiBi bias for a given query/key length.
        Returns: [1, n_heads, q_len, k_len]
        """
        # Default device and dtype
        device = device or self.slopes.device
        dtype = dtype or self.slopes.dtype

        # If cached bias exists for same lengths, return it
        if self._cached_len == (q_len, k_len) and self._cached_bias is not None:
            return self._cached_bias

        # Query and key positions
        q_pos = torch.arange(q_len, device=device)
        k_pos = torch.arange(k_len, device=device)

        if self.causal:
            # Distance = (q - k), but clamp negatives to 0
            # clamp_min(0) sets any value < 0 → 0
            dist = (q_pos[:, None] - k_pos[None, :]).clamp_min(0)
        else:
            # Non-causal = absolute distance
            dist = (q_pos[:, None] - k_pos[None, :]).abs()

        # Reshape slopes to [n_heads, 1, 1] for broadcasting
        slopes = self.slopes[:, None, None].to(dtype)

        # Reshape distance to [1, 1, q_len, k_len] for broadcasting
        dist = dist.to(dtype)[None, None, :, :]

        # Final bias: [1, n_heads, q_len, k_len]
        bias = slopes[None, ...] * dist

        # Cache and return
        self._cached_len = (q_len, k_len)
        self._cached_bias = bias
        return bias
```
