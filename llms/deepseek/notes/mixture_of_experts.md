# Mixture of Experts (MoE)

> The second foundational innovation in DeepSeek (alongside MLA). Instead of a dense FFNN, MoE breaks it into smaller **specialized experts** — only a few activate per token, massively reducing compute.
> Covers: MoE Introduction, Routing, Balancing Techniques, and DeepSeek's 3 MoE Innovations.

---

## 1. What is Mixture of Experts?

In a standard transformer, after the attention block, every token passes through a **dense feed-forward neural network (FFNN)** — all neurons fire for every single token. That's expensive.

**MoE replaces the dense FFNN with multiple smaller "expert" networks.** For each token, only a few experts (e.g., 2 out of 64) activate. The rest stay idle.

```
Standard Transformer:
    Token → Attention → Dense FFNN (ALL neurons) → Output
                        ↑ entire network runs for every token

MoE Transformer:
    Token → Attention → Router → Expert 3  ← only these 2 run
                               → Expert 17 ←
                        (other 62 experts idle)
```

**Intuition**: Think of it like a verb token going to the "verb expert" and a noun token going to the "noun expert." Each expert specializes in something different. Instead of the entire NN running, just 1-2 experts run → reduces compute cost and increases speed for both training and inference.

> This is called **sparsity** — breaking down a dense layer into smaller trainable experts where only a subset activates per token.

---

## 2. How Routing Works

The key question: when a token comes in, **which experts should handle it?** This is decided by the **Router**.

### The Routing Matrix

There's a trainable `Routing Matrix` of shape `(embed_dim, num_experts)`:

```
inputs(n_tokens, emb_dim) × Routing_matrix(emb_dim, n_experts) = Expert_Selector_matrix(n_tokens, n_experts)
```

The Expert Selector (ES) matrix tells us how much each expert "wants" each token.

### Example: 4 tokens, 3 experts, choosing top-2

**Step 1**: Multiply inputs with routing matrix → raw scores:

```
ES matrix:
[1, 2, 3],     ← token 1
[5, 2, 4],     ← token 2
[9, 0, 4],     ← token 3
[8, 2, 0]      ← token 4
```

**Step 2**: Keep only top-2 values per token, set the rest to `-inf`:

```
[-inf, 2, 3],     ← token 1 routes to E2 and E3
[5, -inf, 4],     ← token 2 routes to E1 and E3
[9, -inf, 4],     ← token 3 routes to E1 and E3
[8, 2, -inf]      ← token 4 routes to E1 and E2
```

**Step 3**: Apply softmax to get routing weights:

```
[0,    0.36, 0.64],     ← token 1: 36% E2, 64% E3
[0.55, 0,    0.45],     ← token 2: 55% E1, 45% E3
[0.78, 0,    0.22],     ← token 3: 78% E1, 22% E3
[0.9,  0.1,  0   ]      ← token 4: 90% E1, 10% E2
```

**Step 4**: For each token, run the selected experts, **multiply** their outputs by the weight factors, and **add** them together:

```
Token 1 output = 0.36 × Expert2(token1) + 0.64 × Expert3(token1)
Token 2 output = 0.55 × Expert1(token2) + 0.45 × Expert3(token2)
...etc
```

> Two decisions happen here:
> 1. **Sparsity decision**: WHICH experts to choose (top-k selection)
> 2. **Routing**: HOW MUCH attention to give each chosen expert (softmax weights)

---

## 3. The Balancing Problem

### Why Balance Matters

If the router keeps sending most tokens to the same few experts, we get problems:
- **Overloaded experts**: Some experts see way too many tokens → bottleneck
- **Underloaded experts**: Some experts barely train → wasted parameters
- **Expert collapse**: In the worst case, some experts effectively "shut off" and never learn anything useful

### Auxiliary Loss

To penalize imbalanced expert selection, we add an **Auxiliary Loss** to the training loss. It pushes the routing function towards a more uniform distribution.

**Step 1**: Calculate **Expert Importance** — sum each column in the ES weight matrix:

```
Expert Importance:
    E1 importance = sum of E1 weights across all tokens
    E2 importance = sum of E2 weights across all tokens
    E3 importance = sum of E3 weights across all tokens
```

**Step 2**: If there's high variation between expert importances, loss should be high:

```
Coefficient of Variation (CV) = Standard Deviation / Mean
```

**Step 3**: Auxiliary Loss:

```
Auxiliary Loss = λ × (CV)²
```

This gets added to the LLM training loss. As training progresses, expert importances converge → SD decreases → mean gets centered → CV drops → auxiliary loss decreases.

### Load Balancing Loss

Expert importance alone isn't enough. The issue: **expert importance ≠ equal token distribution.** We could have an expert with 1 token routed at very high confidence vs another expert with 4 tokens at low confidence — both might have similar "importance" but wildly different loads.

We need two quantities:

**Expert Probability** (P_i) — probability of being selected:

```
P1 = E1_importance / sum(all_importances)
P2 = E2_importance / sum(all_importances)
P3 = E3_importance / sum(all_importances)
```

**Fraction of Tokens Routed** (f_i):

```
f1 = tokens_routed_to_E1 / n_tokens
f2 = tokens_routed_to_E2 / n_tokens
f3 = tokens_routed_to_E3 / n_tokens
```

**Load Balancing Loss**:

```
Load Balance Loss = scaling_factor × n_experts × Σ(fi × Pi)
```

This ensures experts with more importance handle proportionally more tokens, reducing the mismatch between importance and actual token load.

### Capacity Factor

To prevent any single expert from completely shutting off or being overwhelmed:

```
Expert Capacity = (Tokens_per_batch / n_experts) × Capacity_factor

where:
    Tokens_per_batch = batch_size × context_length × top_k
    top_k = number of experts chosen per token
```

This caps the maximum number of tokens any single expert can handle per batch.

---

## 4. DeepSeek's 3 MoE Innovations

DeepSeek didn't just use standard MoE — they introduced **3 key innovations** that made their MoE architecture significantly better.

### Innovation 1: Auxiliary-Loss-Free Load Balancing

**The problem with auxiliary loss**: The scaling factor λ is a nightmare to tune:
- Too low → negligible effect, experts stay imbalanced
- Too high → dominates the training loss → inefficient backpropagation, hurts model quality

**DeepSeek's solution**: Get rid of the loss entirely. Instead, use a **bias-based** approach:

**Step 1**: Find average token load per expert:

```
Average load = total_tokens_routed / num_experts
```

**Step 2**: For each expert, check if it's **Underloaded** or **Overloaded** compared to the average. Calculate the `load violation`.

**Step 3**: Maintain a bias `b_i = 0` for each expert, updated as:

```
b_i = b_i + u × sign(load_violation_error)

where:
    u = predefined small constant
    sign(load_violation_error) = +1 or -1
```

**Step 4**: Add biases to the Expert Selector Matrix:

```
Underloaded expert → add bias    → increases probability of being chosen
Overloaded expert  → reduce bias → decreases probability of being chosen
```

> No loss function needed. The biases are adjusted dynamically during training, nudging the router towards balance without contaminating the training loss.

---

### Innovation 2: Shared Experts

**Two problems** identified in the DeepSeekMoE paper:
1. **Knowledge Redundancy**: Multiple experts end up learning the same common knowledge (e.g., basic grammar), wasting capacity
2. **Knowledge Hybridity**: Experts try to learn too many things at once instead of specializing

**Solution for redundancy**: Split experts into two types:

```
MoE Layer:
├── Shared Experts (always activated for EVERY token)
│   └── Handle common/redundant knowledge (grammar, syntax, etc.)
│
└── Routed Experts (selected via Expert Selector Matrix)
    └── Handle specialized tasks (domain knowledge, reasoning, etc.)

Final output = Shared_expert_output + Routed_expert_output
```

By offloading redundant knowledge to shared experts, the routed experts are **freed to specialize** — they don't waste capacity re-learning basic patterns that every token needs.

---

### Innovation 3: Fine-Grained Expert Segmentation

**Solution for knowledge hybridity**: Make experts **smaller and more numerous** so each one can truly specialize on a single thing.

```
Standard MoE:
    Dense FFNN hidden layer: 4096 neurons
    Split into: 4 experts × 1024 neurons each
    ↑ each expert is still large, tries to learn multiple things

DeepSeek Fine-Grained:
    Dense FFNN hidden layer: 4096 neurons
    Split into: 64 experts × 64 neurons each
    ↑ each expert is tiny, focuses on ONE specialized pattern
```

Smaller experts = more specialization = better performance. Each expert becomes a **super-specialized** unit that focuses on a single linguistic or reasoning pattern.

---

## 5. DeepSeek's Full MoE Architecture

Putting it all together:

```
DeepSeek MoE Layer:
│
├── Shared Experts (always ON)
│   └── Common knowledge all tokens need
│
├── Router (with bias-based load balancing, no aux loss)
│   └── Selects top-k from fine-grained routed experts
│
├── Fine-Grained Routed Experts (64 tiny experts)
│   └── Only top-k activate per token
│   └── Each is super-specialized
│
└── Output = Shared_output + Σ(weight_i × Routed_expert_i_output)
```

---

## 6. Summary

```
Mixture of Experts (MoE):
│
├── Core Idea:
│   ├── Replace dense FFNN with multiple smaller expert networks
│   ├── Only top-k experts activate per token (sparsity)
│   └── Massively reduces compute while maintaining model capacity
│
├── Routing:
│   ├── Routing matrix: inputs × routing_weights = expert scores
│   ├── Keep top-k, set rest to -inf
│   ├── Softmax → routing weights
│   └── Output = weighted sum of selected experts' outputs
│
├── Balancing Problem:
│   ├── Auxiliary Loss: λ × (CV)² added to training loss
│   ├── Load Balancing Loss: scaling × n_experts × Σ(fi × Pi)
│   └── Capacity Factor: caps max tokens per expert
│
└── DeepSeek's 3 Innovations:
    ├── 1. Auxiliary-Loss-Free Load Balancing
    │   └── Bias-based approach, no loss contamination
    ├── 2. Shared Experts
    │   └── Always-on experts handle redundant knowledge → frees routed experts to specialize
    └── 3. Fine-Grained Expert Segmentation
        └── Many tiny experts (64×64) instead of few large ones (4×1024) → super specialization
```

---

## References

- [Build DeepSeek from Scratch — MoE Lectures 18-21 (Dr. Raj Dandkar / Vizuara)](https://www.youtube.com/@vizuara)
- [DeepSeekMoE Paper](https://arxiv.org/abs/2401.06066) — Knowledge redundancy, hybridity, shared experts
- [DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437) — Auxiliary-loss-free balancing, fine-grained segmentation
- Points 10–12 in [`deepseek.md`](../deepseek.md) — Brief MoE notes
- Code: [`../code/moe.py`](../code/moe.py)
