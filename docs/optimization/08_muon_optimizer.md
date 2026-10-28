# Muon Optimizer — Beginner Notes

Muon is a neural-network optimizer designed mainly for **matrix-shaped weights in hidden layers**. It combines:

1. **Momentum**, which smooths noisy gradients;
2. **Orthogonalization**, which reshapes the update so that its important directions are more evenly balanced;
3. **Newton–Schulz iteration**, a fast approximation used to perform that orthogonalization.

The name is commonly expanded as **MomentUm Orthogonalized by Newton-schulz**.

These notes assume very little mathematics. The goal is to build intuition first and introduce formulas only when they clarify what the code is doing.

> Important: Muon is not simply a drop-in replacement for AdamW for every parameter. It is intended primarily for 2-D hidden-layer weight matrices. Biases, embeddings, normalization parameters, and output heads are normally handled by AdamW or another standard optimizer.

## 1. First: what does an optimizer do?

Training a neural network means adjusting many numbers called **parameters** so that the model makes fewer mistakes.

An optimizer decides how to adjust those numbers after each batch of training examples.

The basic loop is:

```text
1. Run the model and calculate a loss.
2. Calculate the gradient: which direction would reduce the loss?
3. Use an optimizer to turn that gradient into an update.
4. Change the parameters.
5. Repeat.
```

### A small analogy

Imagine walking down a foggy mountain:

- the **loss** is your height;
- the **gradient** tells you which way is uphill;
- moving opposite the gradient takes you downhill;
- the **learning rate** controls your step size;
- the **optimizer** decides how to use the direction and how large the step should be.

SGD, AdamW, and Muon are different strategies for choosing those steps.

## 2. Why do we need something beyond plain SGD?

Suppose the gradient points in a slightly different direction on every batch because each batch contains different examples. A raw SGD update can zig-zag or be unstable.

**Momentum** keeps a running direction. If several recent gradients point in roughly the same direction, momentum strengthens that direction. If a single batch is noisy, momentum reduces its influence.

In simplified form:

```text
momentum = momentum_factor * old_momentum + current_gradient
update_direction = momentum
```

This is the first major part of Muon. It is similar to momentum SGD.

## 3. What makes Muon different?

AdamW treats parameters mostly as individual numbers. It tracks statistics for each number and scales each coordinate of the update.

Muon treats a hidden-layer weight as a **matrix** and uses the relationships between rows and columns.

For a linear layer, the operation is often written as:

```text
output = weight_matrix × input
```

The weight matrix is not just a long list of unrelated numbers. It transforms one vector space into another. Muon therefore tries to make the update sensible as a **matrix transformation**, not only as independent scalar changes.

The key extra step is:

```text
gradient
   ↓
momentum
   ↓
orthogonalize the momentum update
   ↓
scale it for the matrix shape
   ↓
update the weight
```

## 4. What does “orthogonalize the update” mean?

The word **orthogonal** is related to perpendicular directions. In a matrix, it means that the main directions represented by the matrix are kept balanced and do not become excessively concentrated in one direction.

You do not need to calculate perpendicular vectors by hand to understand the optimizer. The practical idea is:

> Muon reshapes the momentum update so that its strongest directions do not overpower all the others.

### Why might this help?

A matrix update can have some directions that are enormous and other directions that are tiny. This can make learning inefficient:

- one direction changes too aggressively;
- another direction barely learns;
- the layer becomes poorly conditioned;
- the optimizer spends computation correcting its own imbalance.

Muon’s orthogonalization step attempts to make the update’s directional strengths more uniform.

This is an optimization choice, not a guarantee that every update is literally a perfect orthogonal matrix.

## 5. The mathematical property being used: singular values

A useful matrix property here is the **singular value decomposition**, usually abbreviated SVD.

Any suitable matrix can be thought of as combining three operations:

```text
rotate or change coordinates
→ stretch different directions by different amounts
→ rotate or change coordinates again
```

The amounts used in the stretching step are the matrix’s **singular values**.

The property being used is:

> Singular values tell us how strongly a matrix stretches different directions.

If one singular value is much larger than the others, the matrix strongly favors one direction. If the singular values are more balanced, the matrix treats directions more evenly.

Muon’s orthogonalization is closely related to replacing the singular values of the update with values that are approximately equal while preserving its main directional structure.

In a more advanced description, if an update matrix has SVD

```text
M = U × S × Vᵀ
```

then the orthogonalized version is approximately:

```text
U × Vᵀ
```

Here, `U` and `Vᵀ` preserve directional information, while `S` contains the unequal stretching amounts that are being removed.

You do not need to compute an SVD to use Muon. Direct SVD would usually be too expensive, so Muon uses Newton–Schulz iterations as a faster approximation.

## 6. What is Newton–Schulz iteration?

Newton–Schulz iteration is a repeated numerical procedure. It starts with the momentum matrix and repeatedly applies a carefully chosen matrix polynomial.

At a high level:

```text
start with the momentum matrix
repeat a small number of times:
    multiply matrices together
    apply a stabilizing polynomial
    move closer to the desired orthogonalized matrix
```

The commonly used polynomial has the form:

```text
f(x) = a·x + b·x³ + c·x⁵
```

The coefficients in the original Muon implementation are approximately:

```text
a = 3.4445
b = -4.7750
c = 2.0315
```

### The mathematical property being used: fixed points

The property being used here is a **fixed point**. A value is a fixed point of a function if applying the function leaves it unchanged:

```text
f(x) = x
```

The coefficients are chosen so that values near the desired scale are pulled toward a stable target. Repeating the operation gradually makes the singular values of the update more similar.

In practical implementations, about five iterations are often used as a speed/accuracy tradeoff. More iterations can produce a more accurate orthogonalization, but may not improve training enough to justify the extra work.

## 7. A simplified Muon update

Let:

- `W` = a weight matrix;
- `G` = the current gradient matrix;
- `M` = the momentum matrix;
- `μ` = the momentum factor;
- `η` = the learning rate;
- `O` = the orthogonalized update.

The conceptual update looks like this:

```text
M = μ × old_M + G
O = NewtonSchulzOrthogonalize(M)
W = W - η × O
```

The minus sign means “move opposite the gradient,” because the goal is to reduce the loss.

This is a simplified explanation. Production implementations also include details such as Nesterov momentum, numerical stabilizers, weight decay, and shape-dependent learning-rate adjustment.

## 8. What is Nesterov momentum?

Nesterov momentum is a variation of momentum that estimates the direction after looking slightly ahead.

You can think of ordinary momentum as asking:

> “Given the direction I have been moving, what should I do now?”

Nesterov momentum asks approximately:

> “Given where momentum is about to take me, what does the gradient suggest I should do?”

Muon implementations commonly enable Nesterov momentum, though it is a configurable option.

## 9. Why does matrix shape matter?

A matrix can be tall, wide, or square. For example:

```text
shape [1024, 4096]  → 1,024 rows and 4,096 columns
shape [4096, 1024]  → 4,096 rows and 1,024 columns
```

Even if two updates have similar mathematical structure, their average per-entry size can differ because the matrices contain different numbers of entries and have different geometry.

### The property being used: RMS

**RMS** means root mean square. In this context, it is a measure of the typical size of the entries in an update.

You can think of update RMS as answering:

> “How large is a typical update for this matrix?”

The scalable Muon work points out that the natural update RMS changes with matrix dimensions. Without a correction, a large matrix may receive updates that are too small, while a small matrix may receive updates that are too large.

The commonly described correction scales the update using a factor related to:

```text
sqrt(max(number_of_rows, number_of_columns))
```

The square root is the mathematical property being used to compensate for how matrix size affects the typical magnitude of an orthogonalized update. You do not need to memorize the formula; the practical purpose is to make updates more comparable across differently shaped matrices.

Some implementations instead adjust the update to match the observed RMS range of AdamW, often around 0.2–0.4 in the cited scaling experiments.

## 10. Weight decay matters for large-scale training

Weight decay gently pulls weights toward smaller values. In simplified form:

```text
weight = weight - learning_rate × weight_decay × weight
```

The property being used is simple multiplication: a larger weight receives a larger pull toward zero.

The original Muon work did not initially include weight decay in the same way AdamW does. Later scaling work reported that weights and layer outputs could grow too large during long training. Adding decoupled weight decay helped control this growth and improved large-scale behavior.

This is why modern implementations generally expose `weight_decay` and why “vanilla Muon” and “scaled Muon” should not be treated as exactly the same recipe.

## 11. Which parameters should use Muon?

A common setup divides parameters into two groups:

### Muon group

Use Muon mainly for 2-D matrix parameters in hidden linear layers, such as the weight matrices in transformer MLP and attention projections.

### AdamW group

Use AdamW or another standard optimizer for parameters such as:

- biases;
- token embeddings;
- positional embeddings, depending on the architecture;
- normalization scales and biases;
- language-model output heads;
- parameters that are not appropriate matrix-shaped hidden weights.

This hybrid setup is normal. Using Muon for every parameter can be incorrect because some parameters do not have the matrix structure Muon expects.

## 12. PyTorch-style example

Modern PyTorch documentation provides a built-in `torch.optim.Muon`. A beginner-friendly parameter split looks like this:

```python
muon_params = [
    p for p in model.parameters()
    if p.requires_grad and p.ndim == 2
]

other_params = [
    p for p in model.parameters()
    if p.requires_grad and p.ndim != 2
]

muon_optimizer = torch.optim.Muon(
    muon_params,
    lr=0.02,
    momentum=0.95,
    weight_decay=0.1,
)

adamw_optimizer = torch.optim.AdamW(
    other_params,
    lr=3e-4,
    weight_decay=0.01,
)

# Each training step:
muon_optimizer.zero_grad()
adamw_optimizer.zero_grad()

loss = loss_function(model(inputs), targets)
loss.backward()

muon_optimizer.step()
adamw_optimizer.step()
```

The exact learning rates are not universal. They depend on the model, data, batch size, parameterization, and implementation.

## 13. How Muon compares with AdamW

| Question | AdamW | Muon |
|---|---|---|
| Main view of parameters | Mostly coordinate-by-coordinate | Matrix-aware for hidden weights |
| Uses momentum? | Yes, through Adam’s first-moment estimate | Yes |
| Uses per-entry variance statistics? | Yes | Not in standard Muon |
| Orthogonalizes matrix updates? | No | Yes, approximately |
| Works for every parameter type? | Generally easier to apply broadly | Usually paired with AdamW for non-matrix parameters |
| Extra computation | Familiar and efficient | Matrix multiplications for Newton–Schulz iterations |
| Memory for optimizer state | Typically more state, including a variance estimate | Typically momentum state for Muon parameters, but implementation details matter |
| Maturity and defaults | Very established | Newer and more implementation-sensitive |

Muon is not “AdamW but faster in every situation.” It is a different geometric assumption about how matrix updates should behave.

## 14. Why people are interested in Muon

The original work reported strong speed records on smaller neural-network and language-model training tasks. Later work, **Muon is Scalable for LLM Training**, reported that weight decay and update-scale correction allowed Muon to scale to much larger language models. Its experiments reported approximately half the training FLOPs needed to match AdamW performance in their compute-optimal comparisons.

These are important results, but they should be read as results for particular implementations, models, datasets, and tuning procedures—not as a universal guarantee.

The main reasons for interest are:

- potentially faster loss reduction per unit of training compute;
- a matrix-aware update rule;
- less optimizer-state memory than Adam-like methods in some configurations;
- possible advantages at large batch sizes and large language-model scales;
- availability in modern PyTorch and distributed-training implementations.

## 15. Limitations and practical cautions

### 1. It is not universally applicable

Muon is designed around matrix-shaped hidden weights. Do not blindly place embeddings, biases, and normalization parameters into the Muon parameter group.

### 2. Results depend on scaling details

Weight decay, update RMS correction, learning rate, momentum, and the number of Newton–Schulz steps can materially affect results.

### 3. Orthogonalization is approximate

Using only a few Newton–Schulz steps is a deliberate efficiency tradeoff. The result is close enough for training, but not an exact SVD or exact polar decomposition.

### 4. Distributed implementations require care

For sharded training, the optimizer may need information from the full matrix before orthogonalizing it. A local shard alone may not contain enough information. Use a tested implementation designed for the distributed setup you are using.

### 5. Do not overgeneralize benchmark results

Muon has promising published results, but optimizer performance can change with architecture, data quality, batch size, numerical precision, and training length.

## 16. The whole algorithm in one picture

```text
current model weights
          |
          v
calculate gradient G
          |
          v
combine with old momentum M
          |
          v
orthogonalize M with Newton–Schulz
          |
          v
adjust update size for matrix shape
          |
          v
apply weight decay
          |
          v
update the weight matrix W
```

## 17. One-sentence summary

> Muon is a momentum-based optimizer for neural-network weight matrices that reshapes each momentum update so its important directions are more evenly balanced, using a fast Newton–Schulz approximation instead of an expensive exact matrix decomposition.

## 18. Recommended learning order

If this topic feels difficult, study it in this order:

1. Gradient descent: what a gradient and learning rate do.
2. Momentum SGD: why a running direction helps.
3. Matrices as transformations: rows, columns, and matrix multiplication.
4. Matrix norms and RMS: how to measure the size of a matrix.
5. Singular values: how a matrix stretches different directions.
6. SVD and polar decomposition: how a matrix can be separated into direction and stretching.
7. Newton–Schulz iteration: how repeated matrix operations approximate orthogonalization.
8. Muon implementation details: parameter grouping, scaling, weight decay, and distributed training.

## Sources and further reading

1. [Muon: An optimizer for hidden layers in neural networks — Keller Jordan et al.](https://kellerjordan.github.io/posts/muon/) — original design and motivation.
2. [Deriving Muon — Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon) — mathematical motivation and the connection to linear layers.
3. [Muon GitHub repository — KellerJordan/Muon](https://github.com/KellerJordan/muon) — original reference implementation and parameter-grouping guidance.
4. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — weight decay, update-RMS correction, distributed training, and Moonlight experiments.
5. [PyTorch `torch.optim.Muon` documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) — current API, defaults, and a hybrid Muon/AdamW example.

### Reading note

Some newer Muon variants and analyses continue to appear. Treat them as follow-up research rather than as part of the basic Muon algorithm described above.
