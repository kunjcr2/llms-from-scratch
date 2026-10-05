# Per-Layer Embeddings (PLE)

**Source:** Gemma 4 architecture (E2B, E4B models)  
**Type:** Architectural technique — capacity add-on for compact transformers

---

## Core Idea

Each transformer block receives its own dedicated embedding slice derived from the input token. This slice is injected as an extra residual signal at each layer, giving the model additional representational capacity without widening the attention and FFN stack.

The hidden state dimension `d` stays constant throughout the entire forward pass. PLE never changes the shape of the main computation path.

---

## Mental Model

The token is looked up in a large PLE embedding table that produces a vector of size `num_layers × ple_dim`. This big vector is sliced once at the start — each layer claims its own chunk.

At every layer:

```
hidden (d) --> Attention --> FFN --> hidden' (d)
                                         |
                              + PLE slice for this layer
                                (gated, projected to d, normalized)
                                         |
                                    hidden'' (d)        <- still d-dimensional
```

The PLE slice for layer `i` is a table lookup result, not a computed feature. It is token-specific and layer-specific but does not depend on prior hidden states.

---

## Forward Pass (Step by Step)

1. Token IDs are looked up in a dedicated PLE embedding table, producing a packed vector of size `num_layers × ple_dim`.
2. Normal token embeddings are projected into the same PLE space and combined with the lookup result.
3. The combined vector is reshaped — each layer gets its own slice of size `ple_dim`.
4. The transformer runs normally: attention, then FFN.
5. After FFN, the current layer's PLE slice is:
   - Gated by the hidden state
   - Projected back to model width `d`
   - Normalized
   - Added as a residual to the hidden state

---

## Key Distinction: What Does NOT Change

| Property | Behavior with PLE |
|---|---|
| Hidden state dimension | Stays `d` throughout |
| Attention computation | Unchanged |
| FFN computation | Unchanged |
| KV cache size | Unchanged |
| Routing / sparsity | None — fully dense |

The PLE path is purely additive. It does not replace or reroute any existing computation.

---

## Why It Is Cheap

Embedding parameters are cheap relative to attention/MLP parameters:

- Embedding lookup = table index + small projection (no matrix multiply scaling with sequence length)
- Extra parameters live in embedding tables, not in every repeated attention block
- FLOPs added per layer are small compared to the attention and FFN cost

This is why Gemma 4 reports two parameter counts:

| Model | Effective (compute-heavy) | Total (including PLE embeddings) |
|---|---:|---:|
| Gemma 4 E2B | 2.3B | 5.1B |
| Gemma 4 E4B | 4.5B | 8B |

The **E** prefix stands for **effective** — the compute behavior resembles the smaller number, not the total.

---

## Comparison to Similar Ideas

**KV Sharing**  
Reduces KV cache memory during inference. PLE does not interact with the cache at all.

**Mixture of Experts (MoE)**  
Sparsely routes tokens to different FFN experts. PLE keeps the fully dense path and adds a separate cheap residual on top.

**Wider backbone**  
Increasing `d` scales compute at every attention and FFN layer uniformly. PLE concentrates extra parameters in embedding tables and small projections, leaving the backbone unchanged.

---

## When It Makes Sense

- Compact models (sub-10B) where compute budget is tight but more capacity is needed
- Edge deployment under latency or memory constraints
- Less clearly useful for large models — at larger scale, the backbone already has sufficient capacity and MoE is typically a better fit for adding conditional compute

---

## Open Questions / Follow-Up

- What is the actual `ple_dim` used in E2B vs E4B?
- How is the gating implemented — sigmoid, tanh, or learned scalar?
- Does PLE interact with quantization or affects activation outliers?
- Is the PLE table shared across layers or fully independent per layer?