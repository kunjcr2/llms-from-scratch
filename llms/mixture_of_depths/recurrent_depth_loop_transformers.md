# Recurrent Depth and Looped Transformers

> These notes summarize Sebastian Raschka's discussion of the OpenAI Astra rumor, recurrent depth, and looped transformers. The claim that Astra uses this architecture was not confirmed by OpenAI at the time of the discussion.

## 1. Core idea

A **looped transformer** reuses a transformer stack one or more times. Instead of giving every effective layer its own parameters, the model sends the hidden representation back through the same stack.

```text
input → 22-layer stack → 22-layer stack (same weights) → output
```

This produces an effective depth of 44 layers while storing only one set of 22 transformer blocks. The input text is not literally re-tokenized and processed repeatedly; more precisely, the model repeatedly processes the **intermediate hidden representations**.

## 2. Why reuse layers?

Layer reuse increases effective computation and depth without proportionally increasing the parameter count.

| Architecture | Unique layers | Effective layers | Parameter storage | Compute |
| --- | ---: | ---: | --- | --- |
| Regular 44-layer transformer | 44 | 44 | High | High |
| Looped 22-layer transformer, 2 passes | 22 | 44 | Lower | Similar to 44 layers |

Benefits include fewer weights to store and load, potentially more computation per parameter, and scaling depth without duplicating every block. However, training and inference still execute every effective pass, so recurrence does not automatically reduce FLOPs or latency.

## 3. Nanbeige4.2-3B example

Nanbeige4.2-3B is presented as an example of a fixed looped transformer:

1. Token embeddings enter a 22-layer transformer stack.
2. The resulting hidden states go through the same 22 layers again.
3. The output is passed to the remaining output components.

Conceptually, this resembles a 44-layer transformer, except corresponding layers share weights.

## 4. Training from scratch vs. retrofitting

A recurrent architecture can be created by:

- **Training from scratch:** define the recurrence before training so the model learns representations compatible with repeated passes.
- **Retrofitting:** train a normal transformer first, then repeat its layer stack.

Training from scratch is expected to work better. A pretrained model depends on the activation distributions it encountered during training; repeating its layers changes those distributions. The discussion reports that the from-scratch approach performed significantly better than retrofitting, though a detailed ablation study was not provided.

## 5. Why two passes?

Two passes may provide a favorable compute/capacity trade-off. The cited experiments retained approximately 75% of token efficiency while gaining capacity. Additional passes produced only marginal improvements, suggesting diminishing returns. The best number of passes remains dependent on the model, data, and training setup.

## 6. KV-cache implications

Weight sharing does **not** imply KV-cache sharing. Even when the same block is reused, its input and output hidden states differ on each pass. Therefore, a 44-effective-layer, two-pass model should keep separate KV-cache entries for each effective layer for best performance. Sharing or halving the cache saves memory but can reduce quality.

## 7. Mixture-of-Recursions and dynamic depth

**Mixture-of-Recursions** allows different tokens to receive different numbers of recurrent passes:

- some tokens pass through the recursion block once;
- some pass through it twice;
- difficult or important tokens may pass through it three or more times.

A learned router decides whether a token continues to the next loop. This is analogous to Mixture-of-Experts routing, but the decision is about **how many times a token uses shared depth**, not which expert processes it.

Dynamic depth can improve efficiency because easy tokens stop earlier while harder tokens receive more computation. It also complicates attention and caching because tokens at the same sequence position may exist at different effective depths.

## 8. Attention with dynamic depth

At a given recurrent layer, the model must decide which previous tokens are available for attention. It can attend to every previous token, maximizing context but increasing memory and compute, or attend only to tokens that reached the same layer, reducing cost but potentially losing information. This is similar to the trade-off in sliding-window and sparse attention.

## 9. Relation to chain-of-thought

Recurrent depth should not automatically be interpreted as hidden or obscured chain-of-thought. Chain-of-thought usually refers to additional generated tokens used for inference-time reasoning; looped transformers perform additional computation on hidden states inside the network. More internal depth may make reasoning more compact, but it does not prove that the model generates fewer reasoning tokens or that its reasoning is inherently uninterpretable.

## 10. Relation to recursive self-improvement

These are different concepts:

- **Looped transformer:** repeatedly applies the same learned transformer block.
- **Recursive self-improvement:** a system changes or improves itself, such as modifying its own capabilities or training process.

Looped recurrence is an architectural scaling technique, not a model improving itself.

## 11. Universal Transformers

The idea is closely related to **Universal Transformers** (2018), which apply shared transformer layers repeatedly for a variable number of recurrent steps (T).

- Universal Transformer: repeated shared layers, potentially with variable steps;
- Nanbeige-style loop: a fixed 22-layer stack repeated a fixed number of times;
- Mixture-of-Recursions: a router selects recurrent steps per token.

## 12. Latent reasoning with recurrent depth

A 2025 latent-reasoning approach extends fixed recurrence by feeding the original input back into intermediate hidden states. This can be viewed as adding repeated residual-style connections between the input and recurrent states. It remains part of the same broad family: repeated computation over latent representations rather than reasoning expressed entirely as generated text.

## 13. Main conclusions

1. Astra's alleged use of recurrent depth was a rumor, not a confirmed architectural disclosure.
2. A looped transformer reuses weights while increasing effective depth and computation.
3. Reusing weights saves parameter storage, but not the cost of executing repeated layers.
4. Training the recurrent architecture from scratch is generally preferable to adding recurrence after pretraining.
5. Two passes may offer a strong practical trade-off; more passes can have diminishing returns.
6. KV caches usually need separate entries for each effective layer, even when weights are shared.
7. Mixture-of-Recursions adds token-level dynamic depth through routing.
8. Recurrent depth is related to Universal Transformers, but is not recursive self-improvement and does not necessarily hide chain-of-thought.
