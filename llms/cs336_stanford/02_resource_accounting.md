# 2. Resource Accounting

## Why account for resources?

The goal is to build the best model within a finite compute, memory, and data budget. Before optimizing, estimate where a computation spends its resources. The estimates are approximate, but useful for catching impossible plans and locating bottlenecks.

The lecture reports a Marin run at (10^{23}) FLOPs whose measured loss was within 0.05 of its scaling-law forecast. This illustrates how smaller runs can inform a larger training plan when the scaling recipe predicts well.

## Tensors and precision

Model parameters, gradients, activations, optimizer states, and data are stored as tensors. Storage is:

`number of elements × bytes per element`

- FP32: 4 bytes per value.
- FP16: 2 bytes, but its limited exponent range can cause overflow and underflow during training.
- BF16: 2 bytes with FP32-like exponent range and lower precision; a common training choice.
- Mixed precision commonly keeps parameters, activations, and gradients in BF16 while using FP32 optimizer states.

Precision choices trade memory and speed against numerical range and resolution. Optimizer states can use substantial memory: Adam stores two FP32 moments (8 bytes per parameter), in addition to parameters and gradients.

## FLOPs and training cost

FLOPs count arithmetic operations; FLOP/s measures hardware throughput. A matrix multiplication of shapes (B\times D) and (D\times K) takes approximately:

`2 × B × D × K FLOPs`

The factor of 2 counts a multiply and an add. Elementwise operations scale with the number of elements and are usually cheaper than large matrix multiplications.

For a typical Transformer training step, a useful rough estimate is **6 × parameters × tokens FLOPs**, when attention’s sequence-length-squared work is not dominant. It comes from roughly one forward pass plus two backward computations (parameter gradients and input gradients). For example, training 70B parameters on 15T tokens requires an estimated (6ND) FLOPs before accounting for hardware utilization and overhead.

## Measuring hardware use

- Theoretical FLOP/s comes from the hardware specification and depends on numeric format. Check specification details such as whether a quoted rate assumes sparsity.
- Measure actual runtime on the target hardware; GPU work is asynchronous, so synchronize around timed operations and repeat measurements.
- **MFU** (Model FLOPs Utilization) is achieved FLOP/s divided by the hardware’s advertised FLOP/s. It compares useful model work with peak throughput; communication and other overhead lower it.
- **Arithmetic intensity** relates computation to data moved. Hardware has a compute ceiling and a memory-bandwidth ceiling (roofline analysis); low-intensity work can be memory-bound even when peak compute is high.

Moving tensors between memory and compute can dominate runtime. Kernel fusion helps by loading data once, doing multiple operations, and writing the result once. For multi-GPU training, communication also matters: parameters, activations, gradients, and optimizer state must be distributed and synchronized.

## Memory-saving methods

- **Gradient accumulation:** compute gradients over microbatches and update parameters after several microbatches. This supports a larger effective batch while limiting activation memory per pass.
- **Activation checkpointing:** save selected activations and recompute the others during backpropagation. This trades extra compute for lower memory use.

### Practical rule

When writing or changing a model, estimate tensor memory and operation counts, then profile it. Use the estimates to identify whether the limiting resource is memory, compute, or communication before optimizing.
