# 1. Course Introduction

## Core idea

Build language models from the ground up to understand how the full system works. The course focuses on high-value components rather than rebuilding every dependency.

The central question is: **What is the best model we can build with a fixed data and compute budget?** Model quality depends on both resources and how efficiently they are used.

## What transfers from small-scale work

- **Mechanics:** how models, kernels, and parallelism work.
- **Engineering mindset:** profile and benchmark; minimize wasted compute, memory, and communication.
- **Modeling intuition:** which data and architecture choices improve quality. This is empirical and may not transfer reliably across scales.

Small models are useful for learning mechanics and testing ideas, but can differ from frontier models: compute bottlenecks can shift with scale, and some capabilities only appear at larger scales. Treat small-scale results accordingly.

Scaling is not a substitute for algorithms. The useful algorithms are those that continue to work efficiently as resources grow; efficiency matters more as runs become expensive.

## Course roadmap

1. **Basics:** implement tokenization, a Transformer, optimization, and training; account for compute and evaluate perplexity.
2. **Systems:** GPU kernels, profiling, memory and compute bottlenecks, multi-GPU training, and inference.
3. **Scaling laws:** use smaller experiments to choose a training recipe and predict behavior at larger compute budgets. Predictability across scales matters alongside local optimality.
4. **Data engineering:** define target capabilities through evaluation; curate, transform, filter, deduplicate, and mix data.
5. **Alignment:** use feedback or preference signals to improve model outputs, including DPO or reinforcement learning methods such as PPO and GRPO.

## Tokenization

A tokenizer maps text to token IDs and back. It should round-trip: decoding encoded text should reproduce the original text.

- Character tokenization has a large, sparse vocabulary and often uses many tokens.
- Byte tokenization has a small vocabulary but long sequences (one token per byte), increasing compute.
- Word tokenization can have an unbounded vocabulary and needs unknown-word handling.
- **Byte Pair Encoding (BPE)** starts with bytes and repeatedly merges the most frequent adjacent token pair. Common sequences become single tokens; rare sequences remain decomposable into smaller units.

More bytes per token means shorter token sequences and less sequence computation, but a larger vocabulary has costs. Tokenization is therefore a trade-off, and enables variable-granularity computation over text.

## Practical takeaways

- Optimize the whole training setup for quality under a fixed resource budget; architecture, tokenization, data, and systems choices interact.
- Training quality depends on balancing model expressiveness, stable optimization, and hardware efficiency.
- For tokenizer implementations, correctness includes round-trip behavior; practical implementations also need efficient merge lookup and handling of text chunks and special tokens.
- Inference matters for serving, evaluation, synthetic data, and RL. It has a prompt **prefill** phase and a token-by-token **decode** phase; decoding is often memory-bound.
- For alignment, generating candidate responses, scoring them, and training toward better ones can use human feedback, verifiers, or model judges. RL adds tuning and throughput challenges.
