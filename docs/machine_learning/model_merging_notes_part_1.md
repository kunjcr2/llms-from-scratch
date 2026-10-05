# Model Merging Notes

> Notes from the video introducing model merging and the MergeKit algorithms.

## Big Picture

Model merging tries to build one strong model by combining the weights of several existing models, usually without extra training.

- Goal: reuse capabilities already present in different fine-tuned models
- Benefit: low compute cost, fast turnaround, and no extra inference latency after merging
- Constraint: most methods assume compatible model architectures and matching tensor shapes

## Why It Matters

Instead of repeatedly fine-tuning a single model for every new use case, model merging treats existing models as a library of useful skills.

- One model may be good at summarization
- Another may be strong at translation or domain-specific reasoning
- Merging aims to combine those strengths into one multitask model

## 1. Model Soups

Model soups average the weights of several models trained from the same base architecture.

- Core idea: compute a weighted average of model parameters layer by layer
- Uniform soup: average all candidate models equally
- Greedy soup: add models one at a time and keep them only if validation improves

### Why it works

- Averaging often yields a model that is close to the best individual model on the original task
- It can generalize better than a single model on out-of-distribution data
- It is extremely cheap computationally because it is basically tensor averaging

## 2. Spherical Linear Interpolation (SLERP)

SLERP is a two-model interpolation method that moves along the sphere instead of taking a straight linear average.

- Works with exactly two models at a time
- Preserves vector magnitude better than plain averaging
- Useful when a direct average would shrink or distort the weight geometry too much

### Intuition

Linear interpolation cuts through the space between two vectors. SLERP follows the curved path between them, which is often a better match for model weight geometry.

## 3. Task Arithmetic

Task arithmetic operates on task vectors, which are the parameter updates created during fine-tuning.

- Start with a pretrained base model
- Fine-tuning produces a weight delta for a specific task
- Add or subtract those deltas to inject or remove capabilities

### Examples

- Add a task vector for a new classification domain
- Subtract a task vector if a capability is unwanted or noisy
- Combine task vectors from different fine-tuned models to build a multitask model

### Main takeaway

The useful signal is often in the update itself, not just in the final fine-tuned weights.

## 4. TIES: Trim, Elect Sign, and Merge

TIES improves merging by reducing parameter interference.

### Problem it addresses

- Some parameters are influential in one model but redundant in another
- Sign conflicts can cancel out useful updates during naive averaging

### Steps

1. Trim small-magnitude parameters
2. Elect the dominant sign for each parameter group
3. Merge only the parameters that survive both filters

### Why it helps

By keeping the influential parameters and dropping conflicting ones, TIES avoids destroying strong signals during the merge.

## 5. DARE: Drop and Rescale

DARE compresses task updates by randomly dropping most of them and rescaling the ones that remain.

- Drop a large fraction of update entries, sometimes up to 99%
- Rescale the surviving updates so their effect stays meaningful
- Produces smaller, cheaper task vectors that are easier to combine later

### Key observation

Larger models can tolerate much more dropping with less damage, which suggests many fine-tuning updates are redundant.

## 6. Franken-Merging

Franken-merging stitches together layers from different models instead of averaging weights.

- Also called pass-through merging
- Requires compatible layer shapes if you are mixing parts directly
- More experimental than the averaging-based methods

### Why it is interesting

It can combine pieces from different models, and in some cases even different architectures, to create a new hybrid model.

## Practical Summary

- Model soups are the simplest merge method
- SLERP is better than linear averaging for two-model interpolation
- Task arithmetic works on fine-tuning updates rather than whole models
- TIES filters out interference before merging
- DARE compresses task vectors by dropping redundant updates
- Franken-merging is the most experimental and the most unusual

## Overall Takeaway

Model merging is a lightweight way to reuse existing model capabilities without full retraining. The main tradeoff is that the best methods usually depend on compatible architectures and careful control of how weights or updates are combined.