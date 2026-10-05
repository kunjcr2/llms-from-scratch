# Compressed Sparse Attention (CSA), Heavily Compressed Attention (HCA), and the DeepSeek V4 Hybrid Architecture

---

## Part 1: The Problem — Why Standard Attention Breaks

You have a sequence of tokens. Each token is a vector. During attention, every token looks at all previous tokens to understand context.

At 1M tokens, token number 999,999 has to look at 999,998 previous tokens. Every single layer. Every single forward pass.

That is:
- Insane amount of compute
- Insane amount of memory (you have to store K and V vectors for every past token — this is the KV cache)

People tried fixing this by reducing the number of attention heads (MQA, GQA, MLA). Those work on the head dimension. DeepSeek V4 does something different — it compresses along the **sequence dimension** instead. Fewer past tokens to attend to, not fewer heads.

---

## Part 2: The Intuition Behind CSA — Your Own Memory

Think about how your memory works.

- Ask you what happened **yesterday** → you remember it clearly, in detail
- Ask you what happened **6 months ago** → you remember a rough summary, not every moment

CSA does the same thing:

```
Recent tokens  →  remember exactly       →  local window (exact attention)
Old tokens     →  remember as a summary  →  compression (approximate attention)
```

That is the entire idea of CSA. Everything else is just DeepSeek's specific way of implementing it.

---

## Part 3: The Local Window — Easy Part

For any token at position `t`, the most recent `w` tokens get **full exact attention**. Nothing special here. Same as standard attention, just on a small window.

```
Sequence:  [tok_0, tok_1, ..., tok_990, tok_991, tok_992, tok_993, tok_994, tok_995, tok_996, tok_997, tok_998, tok_999]
                                                                                                                  ^
                                                                                                              current token (t=999)

Local window (w=8):  [tok_992, tok_993, tok_994, tok_995, tok_996, tok_997, tok_998, tok_999]
                      ← these 8 tokens get full exact attention, nothing compressed
```

Everything before `tok_992` is "old context" and needs to be compressed.

---

## Part 4: The Compression — Where It Gets Interesting

You have 992 old tokens. You do not want to attend to all 992. So you compress them into a smaller number of **summary tokens**.

The simplest version: group every 4 tokens together, squash them into 1.

```
992 old tokens, compress ratio r=4:

[tok_0,  tok_1,  tok_2,  tok_3]   →  summary_0
[tok_4,  tok_5,  tok_6,  tok_7]   →  summary_1
[tok_8,  tok_9,  tok_10, tok_11]  →  summary_2
...
992 tokens → 248 summary tokens
```

Now instead of attending to 992 tokens, you attend to 248 summaries. 4x cheaper on the old context.

The summary tokens are NOT the original tokens. They are compressed representations that try to capture the most important information from each group.

---

## Part 5: How DeepSeek Actually Does the Compression

This is where the video explained something more specific than the simple "linear projection" idea. DeepSeek does it in 3 steps.

### Step 1: First project each token down to a smaller vector

Before any compression, each token's hidden state (dimension D = 7168 in DeepSeek V4) gets projected down to a smaller KV vector of dimension C = 512:

```
token hidden state: [7168 dims]
        ↓   W_KV matrix
KV entry:           [512 dims]
```

So every token becomes a 512-dim KV entry. This is the thing that actually gets compressed, not the full 7168-dim hidden state. Cheaper to work with.

### Step 2: Score each token by importance — data-dependent weighting

Simple averaging would work but it is bad. If one token in a group of 4 is really important and the other 3 are filler, averaging dilutes the important one.

Instead, DeepSeek scores each token by how important it is, then does a **weighted sum** — important tokens contribute more to the summary.

**Scalar version (simpler idea first):**

```
Group: [kv_0, kv_1, kv_2, kv_3]   each is a 512-dim vector

score_i = dot(hidden_state_i, learned_vector)   ← one number per token
[s_0, s_1, s_2, s_3] = softmax([score_0, score_1, score_2, score_3])
                        ← scores sum to 1, like attention weights

summary = s_0*kv_0 + s_1*kv_1 + s_2*kv_2 + s_3*kv_3
```

If `tok_1` is the most important, `s_1` is large, and the summary is mostly `kv_1`.

**Per-dimension version (what DeepSeek actually does):**

The scalar version gives one importance score per token. That means all 512 dimensions of that token's KV vector get scaled by the same number.

But what if `tok_0` has useful information in dimensions 0-100, and `tok_2` has useful information in dimensions 300-400? A single scalar cannot capture that.

So instead of one score per token, DeepSeek computes **one score per token per dimension** — 512 scores per token:

```
Group: [kv_0, kv_1, kv_2, kv_3]   each is 512-dim

scores per token per dim:
  tok_0: [s_0_dim0, s_0_dim1, ..., s_0_dim511]   ← 512 numbers
  tok_1: [s_1_dim0, s_1_dim1, ..., s_1_dim511]   ← 512 numbers
  tok_2: [s_2_dim0, s_2_dim1, ..., s_2_dim511]   ← 512 numbers
  tok_3: [s_3_dim0, s_3_dim1, ..., s_3_dim511]   ← 512 numbers

softmax across the 4 tokens, per dimension:
  dim 0:  softmax([s_0_dim0, s_1_dim0, s_2_dim0, s_3_dim0])  → which token owns dim 0
  dim 1:  softmax([s_0_dim1, s_1_dim1, s_2_dim1, s_3_dim1])  → which token owns dim 1
  ...
  dim 511: softmax over 4 tokens → which token owns dim 511

summary = weighted sum   → each dimension independently picks its best token
```

Result: a single 512-dim summary vector where different dimensions can come from different tokens. Much more informative than scalar weighting.

### Step 3: Overlapping groups — no hard boundaries

Non-overlapping groups create a problem. Imagine the group boundary falls right between two highly related tokens:

```
Group 1: [tok_0, tok_1, tok_2, tok_3]  →  summary_0
Group 2: [tok_4, tok_5, tok_6, tok_7]  →  summary_1
```

If `tok_3` and `tok_4` are closely related (say, two parts of the same sentence), they end up in completely separate summaries and never influence each other. Information gets cut off at the boundary.

DeepSeek fixes this by using **overlapping windows** — each summary draws from multiple adjacent groups. `tok_3` contributes to both `summary_0` and `summary_1`. No hard cuts.

---

## Part 6: DSA — Sparse Selection on Top of Compression

Even after compressing 4:1, at 1M tokens you still have 250,000 summary tokens. Full attention over 250,000 entries is still expensive.

So DeepSeek adds one more step: a fast **scorer** (called the lightning indexer) that quickly evaluates all 250,000 summaries and picks only the top-k most relevant ones for the actual attention computation.

```
250,000 compressed summaries
        ↓  lightning indexer scores each one cheaply
top-k selected  (say k=1000)
        ↓
actual attention computed only on those 1000
```

This is called **DSA (DeepSeek Sparse Attention)**. CSA = compression + DSA together.

---

## Part 7: The Full CSA Pipeline, Step by Step

```
At token t=999, local window=8, compress ratio=4:

STEP 1: Project all past tokens to 512-dim KV entries
        [tok_0...tok_998] → [kv_0...kv_998]   each 512-dim

STEP 2: Local window
        kv_992 to kv_998 → exact attention, no compression

STEP 3: Compress old tokens (kv_0 to kv_991)
        Group into 4s with overlapping windows
        Score each token per dimension
        Weighted sum → 248 summary entries (from 992 old tokens)

STEP 4: DSA sparse selection
        Score 248 summaries → pick top-k (say k=64)

STEP 5: Attend to: 8 local (exact) + 64 selected summaries = 72 total
        Instead of 999 tokens in standard attention
```

---

## Part 8: Why it is called Sparse

The full attention matrix would be `T x T` — every cell filled. CSA leaves most cells empty (zero) because most past tokens are either compressed away or not selected by DSA. Only the local window and selected summaries get non-zero attention weights. Sparse = mostly zeros.

---

## Part 9: HCA — Same Idea, More Aggressive

HCA does the same compression but:
- **No local window** — everything gets compressed, no exceptions
- **Compress ratio = 128** instead of 4 — every 128 tokens becomes 1 summary

At 1M tokens with ratio 128:
```
1,000,000 tokens → 7,812 summary tokens
```

Dense attention over 7,812 entries is already cheap — no DSA needed. The compression alone is enough.

HCA is not trying to be precise. It gives the model a rough global picture of the whole sequence cheaply. Good for early layers where you just need to know "what is this sequence generally about" before doing any detailed processing.

### CSA vs HCA side by side

```
CSA:
  local window  →  exact attention on recent tokens
  old tokens    →  compress 4:1, then sparse select top-k
  purpose       →  local detail + compressed history
  cost          →  medium

HCA:
  no local window
  all tokens    →  compress 128:1, dense attention on summaries
  purpose       →  cheap broad global context
  cost          →  very low
```

---

## Part 10: The Hybrid Architecture — How DeepSeek Stacks Them

Different layers in a transformer need different things. DeepSeek matches the right attention type to what each layer actually needs.

### Early layers (Layer 1-2): HCA only

The model has not built any useful representations yet. Doing precise local attention this early is wasted effort. What you need first is a rough sense of the whole sequence — what is the general topic, what are the main ideas.

HCA is perfect here. Cheap, global, approximate. Think of it as the model skimming the whole document before reading carefully.

### Middle layers (alternating HCA + CSA): Both

Once the model has a global picture, it needs to refine its representations. It alternates:

- **CSA layer** — zoom in, do precise local work, look carefully at recent context and selected history
- **HCA layer** — zoom out, refresh the global picture
- **CSA layer** — zoom in again with updated representations
- ...

Each pass makes the representations sharper. The model builds understanding incrementally without ever paying quadratic cost.

### Last layer: Full attention

One final layer of exact, unrestricted attention over all tokens. Whatever detail the compression blurred gets recovered here. This is the expensive layer, but only one layer pays this cost. The output from this layer directly produces the logits.

### Visual

```
Input tokens
     |
  [Layer 1]  ← HCA         cheap global summary
  [Layer 2]  ← HCA         cheap global summary
  [Layer 3]  ← CSA         zoom in, local detail
  [Layer 4]  ← HCA         zoom out, global refresh
  [Layer 5]  ← CSA         zoom in, local detail
  [Layer 6]  ← HCA         zoom out, global refresh
     ...
  [Layer N-1]← CSA
  [Layer N]  ← Full Attention   one precise final readout
     |
Output logits
```

### The numbers this achieves

Compared to DeepSeek V3.2 at 1M token context:

| Metric | Before (V3.2) | After (V4 Pro) |
|---|---|---|
| Inference compute | 100% | 27% |
| KV cache memory | 100% | 10% |

---

## Part 11: Summary — The One Table You Need

| | Full Attention | CSA | HCA |
|---|---|---|---|
| Local precision | Exact | Exact (window) | None |
| Global coverage | Exact | Approximate | Very approximate |
| KV cache | O(T) | O(k + w) | O(T/128) |
| Compute | O(T²) | Much less | Very little |
| When used in V4 | Last layer only | Middle layers | First 2 + middle layers |