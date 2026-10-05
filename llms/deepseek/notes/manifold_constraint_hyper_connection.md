# Residual Connections, Hyper-Connections, and mHC

A complete walkthrough — every concept explained before it is used, every step justified, math shown, code included, and how DeepSeek uses all of this in V4.

---

# PART 1: RESIDUAL CONNECTIONS (RC)

## What was before residual connections

Before 2016, deep neural networks could not be trained past about 20 layers. People tried — they kept stacking more layers thinking it would make the model more powerful — but training got worse, not better. A 50-layer network performed worse than a 20-layer one.

The reason is the **vanishing gradient problem**.

When you train a neural network, you compute the loss at the output and propagate gradients backward through every layer (this is backpropagation). At each layer, the gradient gets multiplied by the layer weights. If those weights are small (less than 1 on average), the gradient shrinks at every step. After 50 layers of multiplication, the gradient is essentially zero.

What this means in practice: the early layers stop receiving any meaningful signal. They cannot learn anything. The network is broken.

## What residual connections do

In 2016, the ResNet paper proposed a fix that is almost embarrassingly simple. Instead of:

```
x_out = f(x)
```

They wrote:

```
x_out = x + f(x)
```

The `+ x` part is called a **skip connection** or **residual connection**. The idea is that you take the input `x`, run it through the layer to get `f(x)`, and then **add the original `x` back in**.

### Why does this fix vanishing gradients

The magic is in what happens during backpropagation. When you compute the gradient of `x_out` with respect to `x`:

```
d(x_out)/dx = d(x + f(x))/dx = 1 + d(f(x))/dx
```

That `1` is the key. Even if `d(f(x))/dx` is tiny (like 0.001), the total gradient is still `1.001`. The gradient passes through the `+ x` essentially unchanged. It is a highway for gradients to flow back to early layers without shrinking.

Stack 50 of these and the gradient still flows freely. The network can train.

### What residual connections enable

After ResNet, networks went from 20 layers to 1000+ layers without training collapsing. Every transformer you have ever heard of (BERT, GPT, LLaMA, DeepSeek, Claude) uses residual connections. They are everywhere.

## The limitation of residual connections

The `+ x` is **completely fixed**. The model has zero control over it. Every layer just blindly adds whatever came before to its own output. There is no flexibility.

This becomes a problem when you want the model to do more sophisticated routing — for example:
- "Layer 5 should pull more from layer 2 than layer 4"
- "Stream A should skip this layer entirely"
- "Combine information from three different depths"

None of this is possible with a fixed `+ x`. The model is locked into one rigid pattern of information flow.

## Code for residual connections

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # f is the actual layer — could be attention, FFN, etc.
        self.f = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        # x: [B, T, D] — batch, sequence length, hidden dim
        return x + self.f(x)    # the entire residual connection
```

That is it. One line. Fixed addition. No learning, no routing, no flexibility.

---

# PART 2: HYPER-CONNECTIONS (HC)

## What problem does HC solve

The fixed `+ x` is a one-size-fits-all routing scheme. It works, but it is not optimal. ByteDance researchers asked a simple question: what if the model could **learn** how information flows between layers, instead of being forced into a fixed pattern?

That is the entire motivation for Hyper-Connections.

## The core idea

Instead of one stream of hidden states flowing through the network, run **multiple parallel streams**.

Imagine the hidden state is a river. Standard residual networks have one river — the river enters each layer, gets modified, comes out, and continues. Hyper-Connections have **N parallel rivers** flowing simultaneously. Typically N is 4 or 8.

At every layer, before the layer does any computation, the N rivers get **mixed together** using a learned matrix. That mixing is the key innovation. The model learns how to route information between streams at each depth.

## The two learned things — A and B

### Matrix A — the input mixer

`A` is a learned matrix of shape `[N, N]`. It controls how the N streams mix together before the layer's computation `f` runs.

`A[i][j]` answers the question: "How much should stream `i` pull from stream `j`?"

Concrete example with N = 4:

```
A = [[0.7, 0.2, 0.1, 0.0],
     [0.1, 0.6, 0.2, 0.1],
     [0.0, 0.1, 0.8, 0.1],
     [0.2, 0.1, 0.1, 0.6]]
```

Reading the first row: stream 0's mixed input is 70% of stream 0, 20% of stream 1, 10% of stream 2, 0% of stream 3.

If `A` is the identity matrix:

```
A = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
```

Then no mixing happens — each stream just stays itself. This is essentially how a standard residual network behaves.

If `A` has large off-diagonal values, streams pull heavily from each other. Information jumps across streams. The model learns more interesting routing patterns.

### Vector B — the output distributor

After `f` (the actual layer) runs and produces output, that output needs to go somewhere. `B` is a learned vector of shape `[N]` that controls how `f`'s output gets distributed back to each stream.

`B[k] = 0` means stream `k` is not updated by this layer at all. The layer's output is ignored for that stream.

`B[k] = 2.0` means stream `k` receives a strong update from the layer.

This gives the model another lever: it can decide which streams should be affected by which layers.

## How HC actually works step by step

```
Input: 4 streams, each of shape [B, T, D]
       streams = [stream_0, stream_1, stream_2, stream_3]

Step 1: Mix the streams using A
        mixed_streams = A @ streams       # routing happens here

Step 2: Run f on the first mixed stream (or any chosen stream)
        f_output = f(mixed_streams[0])

Step 3: Distribute f_output back to each stream, scaled by B
        update[k] = B[k] * f_output

Step 4: Add the update to each mixed stream
        new_streams[k] = mixed_streams[k] + update[k]

Output: new 4 streams, ready for the next layer
```

Notice that this is essentially a generalization of residual connections. If `A = identity` and `B = [1, 1, 1, 1]`, you get back exactly `x + f(x)` for each stream independently. So HC strictly contains residual connections as a special case.

## What HC enables that residual cannot

- **Layer skipping for some streams** — set `B[k] = 0` and stream k bypasses the layer
- **Cross-depth information flow** — large off-diagonal `A` values mix information from different streams (which carry different layer histories)
- **Per-stream specialization** — different streams can develop different "personalities" that the model uses for different purposes
- **Faster convergence** — empirically converges 1.8x faster than residual
- **Better benchmarks** — +6 points on ARC-Challenge over residual baseline at same training cost

## Why HC breaks at scale

Here is the catastrophic problem.

At each layer, `A` is applied to the streams. Stack 32 layers and you get this product:

```
final_streams = A_32 @ A_31 @ A_30 @ ... @ A_2 @ A_1 @ initial_streams
```

You are multiplying 32 matrices together.

If even **one** of these matrices has values slightly above 1 in some direction, that direction gets amplified. The amplification compounds across all 32 layers. Even tiny imbalances explode exponentially.

Mathematically, if a matrix's largest singular value is `s`, then 32 of them multiplied together has largest singular value approximately `s^32`. If `s = 1.1`, then `s^32 ≈ 17.4`. If `s = 1.2`, then `s^32 ≈ 341`. If `s = 1.3`, then `s^32 ≈ 4920`.

DeepSeek tried HC on a 27 billion parameter model and measured **3000x signal growth**. Activations became so large that gradients exploded. Loss spiked, training diverged, model crashed.

This is why despite HC's promise, it could not be used at scale. Until mHC fixed it.

## Code for HC

```python
import torch
import torch.nn as nn

class HyperConnection(nn.Module):
    def __init__(self, d_model, n_streams=4):
        super().__init__()
        self.n_streams = n_streams

        # The actual layer function — same as residual block
        self.f = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Learned mixing matrix A — controls stream-to-stream routing
        # Initialized near identity so training starts like standard residual
        self.A = nn.Parameter(
            torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams)
        )

        # Learned distribution vector B — controls per-stream update strength
        # Initialized to ones so all streams receive equal updates initially
        self.B = nn.Parameter(torch.ones(n_streams))

    def forward(self, x_streams):
        # x_streams: [B, T, n_streams, D]

        # Step 1: Mix streams using A
        # einsum is just batched matrix multiplication along the stream dimension
        x_mixed = torch.einsum('ij, btjd -> btid', self.A, x_streams)

        # Step 2: Run f on the first mixed stream
        f_output = self.f(x_mixed[:, :, 0, :])     # [B, T, D]

        # Step 3 + 4: Distribute f_output to each stream, scaled by B
        update = f_output.unsqueeze(2) * self.B.view(1, 1, -1, 1)
        x_out = x_mixed + update

        return x_out
```

The key thing to notice: **`A` is unconstrained**. It is just a regular learnable parameter. Nothing prevents it from having large values. That is the bug that makes HC explode at scale.

---

# PART 3: mHC — MANIFOLD-CONSTRAINED HYPER-CONNECTIONS

## The fix in plain English

DeepSeek's insight: HC is great, but `A` needs a constraint. Specifically, `A` must be a matrix that can only **route** information, never **amplify** it.

The mathematical object that has this property is called a **doubly stochastic matrix** — a matrix where every row sums to 1 and every column sums to 1.

## What is a doubly stochastic matrix

A matrix where:
1. Every row sums to exactly 1
2. Every column sums to exactly 1
3. All values are between 0 and 1 (non-negative)

Example with N = 3:

```
0.5   0.3   0.2     row sum: 0.5 + 0.3 + 0.2 = 1.0
0.2   0.5   0.3     row sum: 0.2 + 0.5 + 0.3 = 1.0
0.3   0.2   0.5     row sum: 0.3 + 0.2 + 0.5 = 1.0

column sums:
col 0: 0.5 + 0.2 + 0.3 = 1.0
col 1: 0.3 + 0.5 + 0.2 = 1.0
col 2: 0.2 + 0.3 + 0.5 = 1.0
```

All rows sum to 1. All columns sum to 1. All values are non-negative. This is doubly stochastic.

## Why this constraint prevents explosion

This is the most important part. The math is genuinely beautiful.

### Row sum = 1 means weighted average

If a row of `A` sums to 1 with all non-negative values, then `A @ x` for that row is a **weighted average** of the input streams.

A weighted average has a critical property: **its result cannot be larger than the maximum input**. If the inputs are between -10 and 10, the weighted average is also between -10 and 10. It is impossible for the output to be 100. Amplification is mathematically forbidden.

So with row-sum = 1, applying `A` to streams cannot amplify the signal magnitude. Period.

### Column sum = 1 means total contribution is conserved

Each input stream contributes some amount to each output stream. If the column for stream `j` sums to 1, then stream `j` contributes a total of 1 unit of "weight" across all output streams combined.

This means no stream gets ignored (which would lose information) and no stream gets duplicated (which would create extra information from nothing). Information is **conserved**.

### Combined effect

Together, the two constraints mean `A` is a pure routing matrix. It can rearrange information across streams in any way the model learns, but it cannot create new information or amplify what is there. Signal magnitude stays bounded across any number of layers.

This is exactly what was missing in HC. With `A` constrained to be doubly stochastic, the 3000x explosion that DeepSeek saw becomes mathematically impossible. The signal is bounded by the input magnitude, no matter how many layers stack.

## What is the Birkhoff Polytope

The set of all doubly stochastic matrices forms a geometric shape called the **Birkhoff Polytope**. It is named after George Birkhoff, who studied it in 1946.

You do not need to deeply understand the geometry. The only thing that matters is: **the Birkhoff Polytope is the set of all valid doubly stochastic matrices**, and we want `A` to live inside this set.

So when DeepSeek says "constrain `A` to the Birkhoff Polytope," it just means "force `A` to be doubly stochastic."

## How do you actually enforce the constraint — Sinkhorn-Knopp algorithm

You cannot just hope `A` stays doubly stochastic during training. Gradient updates will push `A` away from this property at every step. You need an algorithm that takes any matrix and **projects** it back onto the Birkhoff Polytope.

That algorithm is called **Sinkhorn-Knopp**. It was published in 1967 (yes, this is decades-old math being used in a 2026 LLM). It is shockingly simple:

```
Step 1: Make all values positive (apply exp to everything)
Step 2: Repeat ~20 times:
    a. Divide each row by its row sum  (now rows sum to 1)
    b. Divide each column by its column sum  (now columns sum to 1)
```

That is the whole algorithm.

### Why does this converge

After step 2a, all rows sum to 1 — but step 2b might break this (because column normalization changes row sums slightly).
After step 2b, all columns sum to 1 — but this might slightly break the row-sum property.

Each iteration, the breakage gets smaller. Mathematically, the matrix gets closer and closer to satisfying both constraints simultaneously. After about 20 iterations, both constraints hold to numerical precision. The matrix is now in the Birkhoff Polytope.

This is the kind of algorithm that looks too simple to work, but the math guarantees it does.

### Code for Sinkhorn-Knopp

```python
def sinkhorn_knopp(M, n_iters=20):
    # Step 1: make all values positive
    # exp guarantees positivity and works well with the alternating normalization
    M = torch.exp(M)

    # Step 2: alternate row and column normalization
    for _ in range(n_iters):
        M = M / M.sum(dim=1, keepdim=True)   # rows sum to 1
        M = M / M.sum(dim=0, keepdim=True)   # columns sum to 1

    return M
```

8 lines of code. That is the entire mathematical engine that enables stable training of HC at scale.

## How mHC works — the one line difference from HC

mHC is identical to HC except for ONE line in the forward pass.

In HC:
```python
x_mixed = torch.einsum('ij, btjd -> btid', self.A, x_streams)
```

In mHC:
```python
A_constrained = sinkhorn_knopp(self.A)     # ← project A onto Birkhoff Polytope
x_mixed = torch.einsum('ij, btjd -> btid', A_constrained, x_streams)
```

That is the entire difference. The raw parameter `self.A` can drift anywhere during training (gradient updates do not respect the constraint). But every forward pass, before `A` actually gets used to route information, Sinkhorn-Knopp snaps it back to a valid doubly stochastic form. The data only ever sees a constrained `A`.

### Properties of mHC

- **Zero extra parameters** over HC. The constraint is enforced algorithmically, not by adding weights.
- **~6-7% training overhead** because Sinkhorn-Knopp runs every forward pass. Negligible for the stability gain.
- **Signal stays bounded at any scale.** Mathematically guaranteed by the doubly stochastic property.
- **Training is stable at 27B+ parameters** where unconstrained HC diverges.
- **Same expressiveness as HC**, just with the explosion bug fixed.

## Code for mHC

```python
import torch
import torch.nn as nn

def sinkhorn_knopp(M, n_iters=20):
    M = torch.exp(M)
    for _ in range(n_iters):
        M = M / M.sum(dim=1, keepdim=True)
        M = M / M.sum(dim=0, keepdim=True)
    return M


class ManifoldHyperConnection(nn.Module):
    def __init__(self, d_model, n_streams=4, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        self.f = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Same A as HC — but we will constrain it before use
        self.A = nn.Parameter(
            torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams)
        )
        self.B = nn.Parameter(torch.ones(n_streams))

    def forward(self, x_streams):
        # ── THE KEY DIFFERENCE: project A onto Birkhoff Polytope ──
        A_constrained = sinkhorn_knopp(self.A, self.sinkhorn_iters)

        # Everything else is identical to HC
        x_mixed = torch.einsum('ij, btjd -> btid', A_constrained, x_streams)
        f_output = self.f(x_mixed[:, :, 0, :])
        update = f_output.unsqueeze(2) * self.B.view(1, 1, -1, 1)
        return x_mixed + update
```

The forward pass is 4 lines. The constraint is enforced by 1 function call. That is mHC.

---

# PART 4: HOW DEEPSEEK USES THIS IN V4

DeepSeek V4 uses mHC throughout the entire model. Specifically:

- Every transformer block uses mHC instead of standard residual connections
- Combined with CSA and HCA attention mechanisms in the hybrid architecture
- The first 2 layers and middle layers use HCA attention with mHC connections
- CSA layers in the middle use mHC connections
- The final full-attention layer also uses mHC connections

The result is a model that:
- Trains stably at scale (no explosion)
- Has more flexible information routing than standard residual networks
- Achieves 27% inference FLOPs and 10% KV cache vs DeepSeek V3.2

mHC is one of the three architectural innovations (along with CSA and HCA) that make DeepSeek V4 dramatically more efficient than V3.2 at long contexts.

---

# PART 5: SUMMARY

## Quick comparison

**Residual Connection (RC):**
- `x_out = x + f(x)`
- Fixed skip connection, no learning
- Solves vanishing gradients
- Used in every transformer since 2016
- Limitation: no flexibility in routing

**Hyper-Connection (HC):**
- Multiple parallel streams (N = 4 typical)
- Learned mixing matrix `A` controls stream-to-stream routing
- Learned vector `B` controls per-stream layer updates
- More expressive than residual
- 1.8x faster convergence, +6 ARC-Challenge points
- Limitation: `A` is unconstrained, signal explodes at scale (3000x at 27B params)

**Manifold-Constrained Hyper-Connection (mHC):**
- Same as HC, but `A` is constrained to be doubly stochastic
- Doubly stochastic = every row sums to 1, every column sums to 1
- Enforced via Sinkhorn-Knopp algorithm (alternate row + column normalization)
- Information is routed but never amplified
- Signal magnitude stays bounded across any number of layers
- Zero extra parameters over HC
- ~6-7% training overhead
- Used in DeepSeek V4

## The unifying pattern

All three follow the same structure:

| Technique | Innovation | Constraint |
|---|---|---|
| RC | Skip connection enables deep training | None — fixed addition |
| HC | Learned routing across N streams | None — `A` is free |
| mHC | Learned routing that cannot explode | `A` must be doubly stochastic |

Each step adds expressiveness, then adds the right constraint to keep it stable. The same pattern shows up in LoRA (constrain to low rank) and CSA (constrain to local + compressed). Find what is unconstrained and breaking, add the right mathematical constraint, get the benefit for free.