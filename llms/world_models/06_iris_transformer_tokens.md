# World Models From Scratch - Lecture 6

## The Journey of a Token Through Iris

Lecture 6 follows a token through the Iris Transformer used to simulate CoinRun. Lecture 5 established the overall model: a VQ-VAE turns each frame into discrete visual tokens, and a causal Transformer predicts future tokens conditioned on actions. This lecture opens that Transformer up and explains how its token sequence is embedded, masked, transformed, and decoded into the next visual token.

## 1. The Input Sequence

In the lecture configuration, a frame is represented by 16 VQ-VAE codebook indices and has one associated action token. With an eight-frame history, the dynamics model receives

$$
8 \times (16 + 1) = 136
$$

tokens:

- 128 visual tokens, representing eight frames; and
- 8 action tokens, one for each frame/action step.

The Transformer uses this history to predict the next visual token. Repeating this autoregressively produces the 16 tokens of the next frame, which the VQ-VAE decoder converts back to an image.

## 2. Token and Position Embeddings

A token ID is only a discrete label. It must be converted to a vector before the Transformer can process it. The model therefore looks up a learned **token embedding** for every visual-token or action-token ID. In the lecture example, the incoming token representation has 64 values and is projected into the Transformer's 256-dimensional residual stream.

Order matters: the same collection of tokens means something different if the frames or patches are rearranged. Because attention alone has no inherent notion of sequence order, Iris adds a learned **position embedding** to each token embedding:

$$
x_i^{(0)} = E_{\text{token}}(z_i) + E_{\text{pos}}(i).
$$

The resulting vector says both *what* the token is and *where* it occurs in the 136-token sequence.

## 3. Causal Masking Prevents Looking Ahead

At position $i$, the model may use only positions at or before $i$. A token cannot attend to a token from the future, because that future token will not be available during an imagined rollout.

This restriction is encoded by a lower-triangular **causal mask**. In scaled dot-product attention, disallowed future positions receive $-\infty$ before the softmax:

$$
\operatorname{Attn}(Q,K,V) =
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V,
$$

where $M_{ij}=0$ for $j\leq i$ and $M_{ij}=-\infty$ for $j>i$.

The mask makes training match inference. Without it, a model could use the known answer from a later token during training and would fail when asked to generate that token at rollout time.

## 4. What Happens in a Transformer Layer

The embedded, position-aware tokens enter a stack of Transformer layers. Each layer has two main transformations, both wrapped in residual connections and normalization:

1. **Masked multi-head self-attention.** Each token exchanges information with permitted earlier tokens.
2. **MLP / feed-forward block.** Each token independently applies a nonlinear transformation to its newly updated representation.

For one attention head, the layer forms queries, keys, and values from its current residual-stream vectors:

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V.
$$

The query at a position asks what information it needs; the keys describe what earlier positions offer; and the values contain the information to retrieve. The attention-weighted values are added back to the residual stream. Multiple heads perform this retrieval in parallel, enabling different heads to specialize in different patterns of frame, patch, or action dependence.

## 5. The Residual Stream Is a Contextual Representation

The lecture describes the hidden vectors as flowing through a **residual stream**. There is not one global context vector: all 136 positions maintain their own vector and update it at every layer.

At early layers, a token's representation is mostly its identity and position. Attention lets it collect relevant information from preceding tokens; the MLP then transforms that combined information. Repeating this process gradually enriches each vector with context.

The lecture's visualizations show two useful qualitative observations:

- the norm of a token vector can grow substantially across layers as information is accumulated; and
- its cosine similarity to its initial embedding can fall, showing that the final vector represents contextual meaning rather than merely the original token identity.

These are diagnostic observations, not a requirement that every well-trained Transformer must show exactly the same numerical pattern.

## 6. Information Moves Toward Recent Positions

Early attention layers may directly consult many earlier tokens to collect context. After several layers, that context has already been passed into more recent positions through the residual stream. Later layers can consequently attend heavily to the latest relevant positions.

This explains why the final hidden state at the end of the visible sequence is sufficient for next-token prediction: it has repeatedly received information from the tokens before it. Older tokens are still part of the causal context, but their contribution may have been compressed into newer contextual vectors rather than read directly by the last layer.

## 7. Predicting the Next Codebook Token

The final hidden state at the current sequence end is mapped through an output head to a logit for every codebook entry. Since the Iris visual vocabulary has 512 entries,

$$
\ell \in \mathbb{R}^{512},
\qquad
p(z_{\text{next}}=k \mid \text{history})
= \operatorname{softmax}(\ell)_k.
$$

The model therefore predicts a categorical distribution, not an averaged continuous image feature. It can assign mass to several plausible tokens and then select or sample one. After a token is generated, it is appended to the sequence, and the process continues until all 16 tokens for the next frame have been produced.

## 8. Iris and RSSM: Experimental Observations

The lecture compares Iris with the previous RSSM on a dataset of 600 CoinRun episodes.

| Property | Iris | RSSM |
| --- | --- | --- |
| Sequence model | Causal Transformer | GRU-style recurrence |
| Latent representation | 16 discrete tokens per frame | Continuous Gaussian latent |
| Visual prediction | Categorical codebook-token prediction | Continuous latent/image prediction |
| Qualitative rollouts | Sharper | More prone to blur |

The learned VQ-VAE codebook is actively used rather than collapsing to only a few entries. Consecutive frames typically differ in only some of their 16 token IDs; the count of changed tokens evolves in a jagged, discrete pattern rather than along a smooth continuous path.

The reported sharpness comparison illustrates the intended effect: token-model dreams score about 0.67, close to the real game's 0.68 and roughly 1.3 times sharper than the RSSM rollout in the shown experiment. This is an empirical result for the lecture setup, not a universal guarantee of discrete models.

## 9. How Much History Is Useful?

Although Iris can attend over eight frames in this configuration, the lecture's context-length experiment finds little additional prediction improvement beyond roughly three frames for this CoinRun task. This means the appropriate history length is a property of the environment and task, not simply a larger-is-always-better setting.

Longer context increases the number of input tokens and computation. It is worth measuring the accuracy--cost trade-off instead of assuming that the full available history is necessary.

## 10. End-to-End Rollout

For one imagined frame, the process is:

1. Tokenize the observed history into 16 discrete tokens per frame and interleave the action information.
2. Add token and position embeddings, then apply the causal Transformer.
3. Use the final state to predict one of 512 codebook tokens.
4. Append that prediction and repeat until 16 next-frame tokens have been generated.
5. Decode the 16 generated tokens with the VQ-VAE to obtain the next imagined CoinRun frame.
6. Add the predicted frame and supplied next action to the rolling context, then continue simulation.

## 11. Key Takeaways

1. Eight frames with 16 visual tokens and one action token each form a 136-token Iris input sequence.
2. Learned token embeddings encode token identity, and position embeddings encode order.
3. Causal masking prevents future-token leakage and makes autoregressive rollout possible.
4. Attention mixes information between allowed token positions; MLPs refine each position independently; residual connections carry the evolving context through the layer stack.
5. Each token position has its own contextual residual-stream vector, but the final position is used to predict the next token.
6. Iris outputs a probability distribution over 512 discrete visual tokens and generates a frame one token at a time.
7. In the CoinRun experiment, discrete token prediction yields noticeably sharper dreams than the continuous-latent RSSM baseline.
