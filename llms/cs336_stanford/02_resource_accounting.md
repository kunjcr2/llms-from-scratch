# 2. Resource Accounting

## Why count resources?

The training goal is to build the best model possible with a fixed budget. The budget may include compute, memory, communication bandwidth, and data. This lecture focuses on estimating computation costs before trying to optimize them.

The aim is not to predict runtime to the last second. Back-of-the-envelope estimates help answer questions such as:

- Is this training run feasible on the available hardware?
- Will the model fit in accelerator memory?
- Is a workload limited by arithmetic or by moving data?
- Which part of the model is worth optimizing?

As a motivating example, the lecture estimates the cost of training a 70-billion-parameter model on 15 trillion tokens. Using the rough formula of six FLOPs per parameter per token gives about `6.3e24` FLOPs. With 1,024 H100 GPUs and an assumed 50% utilization, the lecture estimates about 143 days. The estimate is rough, but makes the scale of the workload clear.

Another example estimates model capacity from GPU memory. If an AdamW training setup uses roughly 12 bytes per parameter before counting activations, an 80 GB GPU can hold roughly 6.7 billion parameters' worth of this state. Eight GPUs could therefore hold roughly 53 billion parameters, before accounting for activations, temporary buffers, and other overhead.

## Tensors: the basic unit

Parameters, gradients, optimizer state, activations, and input data are all stored as tensors. A tensor generalizes scalars, vectors, and matrices to any number of dimensions.

The storage estimate is:

```text
number of tensor elements * bytes per element
```

For example, a 4-by-8 FP32 tensor has 32 elements. Each takes 4 bytes, so it occupies 128 bytes. Large model matrices quickly reach gigabytes.

PyTorch tensors are created on the CPU by default. To use a GPU, move the relevant tensors and model computation to the device; leaving tensors on the CPU prevents accelerator speedups.

## Numeric precision

Floating-point values use bits to represent sign, exponent, and fraction (also called mantissa). Exponent bits determine the range of values; fraction bits determine how finely values can be represented.

| Format | Storage | Useful property | Main limitation |
| --- | ---: | --- | --- |
| FP32 | 4 bytes | High precision and broad range | Uses more memory and may be slower on hardware optimized for lower precision |
| FP16 | 2 bytes | Lower storage than FP32 | Narrow exponent range can cause underflow or overflow during training |
| BF16 | 2 bytes | FP32-like exponent range | Less precise fraction than FP16 |

BF16 is often a practical training format because its broad range helps avoid overflow and underflow. Its lower precision is often acceptable for stochastic neural-network training.

Mixed-precision training uses more than one format. A common arrangement is:

- Parameters, activations, and gradients in BF16.
- Optimizer state in FP32 for more stable updates.
- Operations that need more range or precision can remain in FP32.

PyTorch automatic mixed precision can choose formats for many operations. Matrix multiplications are commonly safe to run in BF16, while some operations, such as exponentials, may remain in FP32.

Lower precision formats such as FP8 and FP4 can reduce memory and increase throughput, but require additional care. Block-scaled FP4, for example, uses a shared scale for a group of values. This expands the range represented by a block, but nearby values cannot be scaled independently. Training a model in low precision is also different from training in higher precision and quantizing it later for inference.

## Naming tensor dimensions with einops

Shape manipulations are a common source of mistakes. Code such as `transpose(-2, -1)` requires the reader to remember what those positions mean. Einops offers operations that describe dimensions by name.

For matrix multiplication, one can describe:

```text
x: sequence_1, hidden
y: hidden, sequence_2
output: sequence_1, sequence_2
```

The repeated `hidden` dimension is summed over. The output keeps the two sequence dimensions. This expresses the operation and its shape without needing to reason about a positional transpose.

Einops tools used in the lecture include:

- **einsum:** a named-dimension form of generalized multiplication and summation. Dimensions omitted from the output are reduced.
- **reduce:** performs sum, mean, max, or min over named dimensions.
- **rearrange:** splits, combines, or reorders dimensions. Parentheses can express that one dimension is a product, such as `heads * head_size`.

Named dimensions help with batched operations. A tensor may have batch, sequence, head, and hidden dimensions; explicit names make it clearer which dimensions are retained, matched, or summed.

## Counting FLOPs

A FLOP is one floating-point operation. FLOP/s is a rate: the number of operations a device can perform per second. Keeping the terms distinct matters when comparing how much work a model requires with how fast a GPU can perform it.

For matrix multiplication:

```text
(B x D) * (D x K) -> (B x K)
```

Each output uses roughly `D` multiplications and `D` additions. Across `B*K` outputs, the cost is approximately:

```text
2 * B * D * K FLOPs
```

The exact count includes a small correction for the final addition in each dot product; rough estimates ignore it. Elementwise operations such as addition scale with the number of elements. For large models, matrix multiplications usually dominate arithmetic, though simple elementwise operations can still be limited by memory movement.

For a linear layer with `N` parameters applied to `B` data points, the forward pass costs roughly `2 * B * N` FLOPs.

## Why training is about six times parameter count times tokens

Backpropagation through a matrix multiplication computes two gradients:

1. The gradient with respect to the input, which passes information to the preceding layer.
2. The gradient with respect to the weights, which updates the parameters.

Each backward computation has roughly the same order of FLOPs as the forward matrix multiplication. Thus, for a simple network:

- Forward pass: about `2 * B * N` FLOPs.
- Backward pass: about `4 * B * N` FLOPs.
- Total: about `6 * B * N` FLOPs.

For language-model training, replace the number of data points with the number of processed tokens. This gives the familiar estimate:

```text
training FLOPs ~= 6 * number of parameters * number of tokens
```

This is a useful approximation for Transformers when attention's sequence-length-squared computation is not the dominant cost. Very long contexts can add substantial attention work, so the estimate can undercount in that setting.

## From FLOP count to runtime

The FLOP count estimates work independent of a particular GPU. To estimate runtime, compare the required work with the hardware's throughput:

```text
runtime ~= required FLOPs / achieved FLOPs per second
```

GPU specification sheets give peak throughput by data type. Check the assumptions: a peak rate may depend on sparsity or other special conditions and may not represent dense work.

Measured throughput is often lower than peak. **Model FLOPs Utilization (MFU)** is:

```text
MFU = achieved model FLOPs per second / peak FLOPs per second
```

The lecture notes that around 50% MFU can be a good result for a modern model, while a straightforward large matrix multiplication may do better. Very low MFU can point to a bottleneck or an implementation problem.

GPU work is asynchronous. For meaningful timing, synchronize before and after the measured operation; otherwise the host may time only the launch rather than the completed computation. Repeat measurements and summarize them rather than relying on a single run.

## Memory movement and arithmetic intensity

An accelerator has compute units and memory. Tensors must be moved from high-bandwidth memory to the compute units, processed, and often written back. That movement takes time and can limit throughput even when the GPU's arithmetic units are fast.

**Arithmetic intensity** is the amount of arithmetic performed per byte moved:

```text
arithmetic intensity = FLOPs / bytes moved
```

The hardware's balance point can be estimated from:

```text
peak FLOPs per second / memory bandwidth in bytes per second
```

The lecture's H100 example gives a balance point near 300 FLOPs per byte. A workload with lower intensity tends to be memory-bound; one with higher intensity can become compute-bound.

Examples from the lecture, using BF16 values:

- **ReLU:** reads and writes each element but does little arithmetic. Its intensity is about 0.25 FLOPs per byte, so it is memory-bound.
- **GELU:** performs more arithmetic per element, but its estimated intensity is still only about 5 FLOPs per byte. It is also memory-bound on this hardware.
- **Dot product:** reads two vectors and produces one value. It does little work per byte and is memory-bound.
- **Matrix-vector product:** reads a large matrix but uses each value only a small number of times. It is generally memory-bound.
- **Matrix-matrix multiplication:** reuses values across many products. Work grows roughly cubically with matrix size while data volume grows roughly quadratically, so intensity increases with size and can exceed the hardware balance point.

The **roofline model** visualizes this relationship. At low arithmetic intensity, performance rises with memory bandwidth. Once intensity is high enough, performance is capped by peak compute. The plot helps explain why more arithmetic does not always mean a slower operation: if the operation is memory-bound, memory movement sets the pace.

Transformers are structured largely from matrix multiplications, which can achieve high arithmetic intensity during training when processing many tokens together. During autoregressive inference, a model generates one token at a time; some operations act more like matrix-vector products and can be memory-bound.

## Kernels, fusion, and multiple GPUs

PyTorch operations ultimately launch GPU kernels. For some operations, a custom kernel can run faster by organizing work to reduce data movement.

**Fusion** combines operations so intermediate values do not need to be written to memory and read back between each step. For example, instead of reading a tensor, applying one operation, writing it, reading it again for a second operation, then writing again, a fused kernel can read once, perform both operations, and write once.

Tiling is another way to reuse data while it is available near the compute units. The general principle is to do more work per byte transferred.

With multiple GPUs, communication between devices becomes another cost. Training must place or move parameters, activations, gradients, and optimizer state where each computation needs them. Data, model layers, sequences, or experts can be divided in different ways; each strategy has communication and memory trade-offs.

## Training memory

For a simple deep network with `L` layers and hidden width `D`, the parameter count is about `L * D^2`. Training also needs memory for activations, gradients, and optimizer state.

For a Transformer, activation storage depends on batch size, sequence length, hidden size, number of layers, and which intermediate values must be retained for backpropagation. This is why a parameter-only memory estimate is not enough to know whether a training run fits.

With BF16 parameters and gradients and Adam's two FP32 moment estimates, a rough per-parameter subtotal is:

| Item | Approximate storage per parameter |
| --- | ---: |
| BF16 parameter | 2 bytes |
| BF16 gradient | 2 bytes |
| First Adam moment (FP32) | 4 bytes |
| Second Adam moment (FP32) | 4 bytes |
| **Subtotal, excluding activations and temporary memory** | **12 bytes** |

The optimizer state is not necessarily the main compute cost, but it takes memory and can prevent a model from fitting. Memory accounting should include activations and temporary workspace as well as the persistent state.

## Reducing activation memory

### Gradient accumulation

If the desired batch is too large to fit at once, split it into microbatches. Compute each microbatch's gradients and add them together without updating the parameters. After all microbatches, perform one optimizer update.

This keeps the effective batch size while reducing the activation memory needed at any one time. It does not remove the arithmetic for processing the full batch, and the parameter update happens only after the accumulated gradients are ready.

### Activation checkpointing

Backpropagation normally stores intermediate activations from the forward pass. Activation checkpointing (also called gradient checkpointing or rematerialization) stores only selected activations and recomputes the missing ones during the backward pass.

- Fewer saved activations reduce memory use.
- Recomputing activations adds work and can increase runtime.
- Saving checkpoints at intervals balances the memory saved against recomputation.

In the extreme case, storing no intermediate activations can require repeatedly recomputing earlier layers and become very expensive. A practical checkpoint interval trades extra computation for a smaller activation footprint.

## Working checklist

Before a training or optimization run:

1. Count parameters and estimate their storage at the chosen precision.
2. Add gradients, optimizer state, activations, and temporary buffers to the memory estimate.
3. Estimate FLOPs from the matrix multiplications and the number of tokens.
4. Compare expected work with measured throughput, not only the hardware peak.
5. Estimate arithmetic intensity to decide whether the likely limit is memory or compute.
6. Profile the actual workload and optimize the bottleneck that the measurements reveal.
