# 3. Architectures and Hyperparameters

## 1. How to think about architecture design

The lecture's goal is not to identify one universally best Transformer. It is to survey the choices made by many modern language models and separate robust defaults from details that remain open to experimentation.

Architecture is a bundle of interacting trade-offs:

- **Expressiveness:** Can the model represent the patterns required by the data?
- **Optimization:** Do signals and gradients remain well behaved as depth increases?
- **Systems efficiency:** Do the operations map well to GPU matrix multiplication, memory bandwidth, and parallelism?
- **Inference cost:** How much computation and memory must be used for every generated token?

These objectives do not always agree. A theoretically expressive operation may be slow because it requires extra memory movement. A stability intervention may constrain the model slightly but make a large training run much less likely to fail. The best way to build intuition is to combine broad comparisons across technical reports with small controlled experiments.

The current default is broadly Llama-like: a decoder-only Transformer with pre-normalization, RMSNorm, a gated MLP, rotary position embeddings, and increasingly grouped-query attention. These defaults are a sensible starting point, not immutable laws.

## 2. Normalization and the residual stream

Consider a residual stream `x` passing through a Transformer block. A post-norm block is conceptually:

```text
x <- LayerNorm(x + Attention(x))
x <- LayerNorm(x + MLP(x))
```

In a pre-norm block, normalization happens before each computation:

```text
x <- x + Attention(LayerNorm(x))
x <- x + MLP(LayerNorm(x))
```

The important difference is whether the normalization operation is placed directly on the residual path. With pre-norm, the residual stream has a relatively clean identity route through the network. During backpropagation, gradients can travel through this route without being repeatedly transformed by LayerNorm. This generally improves signal propagation, convergence, and stability, especially in deep models and when warmup is reduced.

The original Transformer used post-norm. Most modern large language models moved to pre-norm, although the design space is not exhausted:

- A norm can be placed after a sublayer while remaining outside the residual stream.
- A block can use both a pre- and post-sublayer norm (sometimes called double norm).
- Extra norms can be inserted inside attention or the MLP when a particular training run is unstable.

The practical lesson is that normalization placement is both an architectural and an optimization decision. If a model is unstable, adding normalization in a carefully chosen location is often an effective first intervention.

### RMSNorm and systems considerations

LayerNorm centers and scales each feature vector. RMSNorm only rescales it:

```text
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * weight
```

It does not subtract the mean and normally has no bias term. LayerNorm is more expressive in principle, but the extra expressiveness rarely produces a meaningful quality advantage in modern language models. RMSNorm is attractive because it performs less work and moves less data.

This illustrates the lecture's systems point: FLOPs are not the same as runtime. A small normalization operation may account for very few floating-point operations but still require separate reads and writes to memory. Fusing operations, eliminating unnecessary statistics, and keeping the GPU busy can matter more than minimizing the arithmetic count alone.

## 3. MLPs, nonlinearities, and block ordering

The feed-forward sublayer is usually an MLP applied independently at every sequence position. A simple version is:

```text
MLP(x) = W2 activation(W1 x)
```

Older Transformers commonly used ReLU. Modern models tend to use smoother activations such as GELU or SiLU/Swish, and especially gated linear units (GLUs). A generic gated form is:

```text
GLU(x) = W2 (activation(W1 x) * Vx)
```

The activation branch controls the information from the second branch. GeGLU and SwiGLU are common variants. Gating adds a second input projection, so a gated MLP with the same hidden dimension has more parameters than a plain MLP. Implementations therefore reduce the hidden dimension when comparing parameter budgets.

Transformer blocks normally execute serially:

```text
x -> attention -> MLP -> next block
```

A parallel block computes attention and the MLP from the same normalized input and adds both updates to the residual stream. This removes a dependency and can improve utilization. The quality evidence, however, is mixed: early reports found useful systems gains, while later model families have not consistently kept the parallel design. It is a reasonable experiment, not a universal default.

## 4. Position embeddings

Self-attention by itself is permutation-equivariant: without additional information, it cannot tell whether one token came before another. Position information can be supplied by:

- learned absolute embeddings;
- fixed sinusoidal embeddings;
- relative-position biases added to attention scores;
- rotary position embeddings (RoPE);
- no explicit position embedding in some specially designed local/global hybrids.

### RoPE intuition

RoPE rotates pairs of coordinates in queries and keys by an angle determined by position. If `R_i` is the rotation at position `i`, attention uses vectors such as `R_i q_i` and `R_j k_j`. Their dot product depends on the relative offset between `i` and `j`, because rotations compose predictably:

```text
(R_i q) dot (R_j k) = q dot (R_(j-i) k)   (up to the rotation convention)
```

This gives attention a relative-position effect without constructing a separate position-dependent bias matrix. RoPE is also convenient because it can be applied directly to Q and K before the usual attention calculation.

RoPE is not a complete solution to long-context generalization. Models differ in frequency scaling, how many dimensions they rotate, and whether they use full or partial RoPE. Some architectures alternate full attention with local attention, reducing the cost of long sequences while allowing global information to propagate through the stack.

## 5. Hyperparameter rules of thumb

The search space looks enormous when every parameter is considered independently, but modern models cluster around a relatively small set of useful choices.

### Feed-forward width

For a non-gated MLP, a common rule is:

```text
ff_dim approximately 4 * d_model
```

This ratio controls the capacity of the per-token nonlinear transformation. It is surprisingly forgiving: controlled studies show a broad basin of reasonable values rather than a sharp optimum.

Gated MLPs have an additional projection. If the goal is to preserve roughly the same parameter count, the hidden dimension is reduced by about `2/3`, leading to common gated ratios around `2.5-2.7`. Llama-like models sometimes use a larger value around `3.5`, partly because efficient attention reduces the relative cost of spending more parameters in the MLP.

Very large multipliers can still work. T5 used an unusually large multiplier, motivated partly by making matrix multiplications more efficient. The lesson is not that 4x is mandatory; it is that standard values are usually safe and compute-efficient, while unusual values should have a specific systems or modeling motivation.

### Head dimension

The common multi-head convention is:

```text
d_model = num_heads * head_dim
```

This makes the concatenated heads have the model width. The equality is not mathematically required, but many model families use it and experiments suggest a fairly broad basin around this choice. Head dimensions must also be large enough for efficient matrix multiplications; extremely small heads can make the operation less hardware-friendly.

### Depth versus width

When scaling a family of models, one often fixes an aspect ratio between width and depth. A rough modern rule is:

```text
d_model / num_layers is around 100
```

There is meaningful variation, but most models avoid being extremely deep or extremely wide. Depth improves the number of sequential transformations, while width is generally easier to shard across devices. Very deep models create pipeline-parallelism challenges and expose more communication and scheduling overhead. Wide models are often easier to split using tensor parallelism.

## 6. Vocabulary size and scaling

Vocabulary size creates a three-way trade-off:

1. A larger vocabulary can represent text with fewer tokens, reducing sequence-length-dependent computation.
2. A larger embedding table increases parameter and memory cost.
3. The output softmax becomes more expensive as the number of vocabulary entries grows.

Small monolingual models often use vocabularies in the tens of thousands. Multilingual models may use substantially larger vocabularies because a shared vocabulary must represent many scripts and languages. There is no universally correct vocabulary size; it depends on the data mixture, tokenizer behavior, model size, and hardware.

Scaling is similarly multidimensional. Increasing parameters while holding data, sequence length, or hardware behavior fixed does not guarantee an efficient improvement. Width, depth, vocabulary, sequence length, data, and parallelism interact, and two models with similar FLOP counts can have different wall-clock times and memory requirements.

## 7. Regularization and training stability

Language-model training often uses a huge corpus and may make only one pass over the data. In that setting, classical overfitting is less dominant than in small supervised-learning problems. Dropout is therefore not automatically necessary.

Weight decay is more subtle. It may improve optimization even when it does not visibly reduce the gap between training and validation loss. Its effect depends on the optimizer and learning-rate schedule; with learning-rate decay, stronger weight decay can sometimes lead to a better final solution. Therefore, weight decay should not be interpreted only as a generalization regularizer.

As training runs become more expensive, avoiding instability can matter more than making a small architectural improvement. A loss curve with gradient spikes can ruin a run that has consumed substantial compute. Two dangerous locations are the output softmax and the attention softmax: both contain exponentials and normalization operations that can become numerically extreme.

### Stability interventions

- **Z-loss:** Let `z` be the output softmax normalizer. Add a penalty proportional to `(log z)^2`, encouraging `log z` to remain near zero. Because adding a constant to all logits does not change the softmax probabilities, this penalty controls an otherwise poorly constrained numerical degree of freedom.
- **QK norm:** Normalize queries and keys before their dot product. This controls the scale of attention logits and reduces attention degeneracies. It often improves robustness without changing the intended representational structure.
- **Logit soft-capping:** Apply a bounded function, commonly a scaled `tanh`, to the attention logits before softmax. This prevents extreme logits, but it is a stronger intervention and can reduce quality by disallowing very confident attention patterns.

The general pattern is to stabilize the inputs to dangerous nonlinearities. QK norm is a relatively gentle intervention; soft-capping is safer numerically but can impose a larger quality constraint.

## 8. Attention and inference cost

Training and prompt prefill process many tokens in parallel. Autoregressive decoding is different: the model generates one token, appends it to the context, and generates the next. A KV cache stores past keys and values so the model does not recompute all previous projections at every step.

The cache reduces computation, but decoding can become memory-bound. At every step, the system repeatedly reads model weights and the cached keys and values. This is why an architecture that is efficient during training may still be expensive to serve.

### MQA and GQA

In standard multi-head attention, every query head has its own key and value heads. Multi-query attention (MQA) keeps many query heads but shares one key and one value across them. The KV cache becomes much smaller, reducing memory traffic and improving inference efficiency, but the shared K/V representation reduces expressive power.

Grouped-query attention (GQA) is the compromise. Several query heads share one K/V group, but there are multiple K/V groups rather than only one. The number of K/V heads becomes a tunable knob:

```text
multi-head attention:      num_kv_heads = num_query_heads
multi-query attention:     num_kv_heads = 1
grouped-query attention:   1 < num_kv_heads < num_query_heads
```

GQA usually captures most of the cache-size savings while retaining performance close to full multi-head attention. This is why it is a common modern deployment-oriented default.

### Sliding-window attention

Full causal attention lets every token attend to the entire preceding context, but its sequence interaction is quadratic. Sliding-window attention restricts a layer to a fixed local window. A model can alternate several local-attention layers with a full-attention layer. Local layers cheaply aggregate nearby information; later full-attention layers distribute that information globally.

This local/global mixture is increasingly common for long-context models. Other models use a state-space or other efficient sequence layer in place of local attention. The broader theme is that long-context architecture is moving away from the assumption that every layer must perform full global attention.

## 9. Practical defaults and experimental checklist

For a first implementation, a reasonable baseline is:

- pre-norm residual blocks;
- RMSNorm;
- SwiGLU or another gated MLP;
- `ff_dim` around `2.5-3.5 * d_model` for the gated MLP;
- RoPE on queries and keys;
- `d_model = num_heads * head_dim`;
- serial attention and MLP sublayers;
- GQA if decoding memory is important.

Then test one change at a time. Measure not only validation loss, but also tokens per second, peak memory, kernel utilization, communication, KV-cache size, and decode latency. The right architecture is the one that achieves the desired quality within the available training and serving budget.
