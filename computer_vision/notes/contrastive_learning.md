# Contrastive Learning

## Problem and Motivation
In standard supervised learning, we rely on expensive human annotations (labels). **Self-supervised learning** aims to learn useful representations from unlabeled data.

Early self-supervised methods used pretext tasks (jigsaw puzzles, rotation prediction), which were heuristic and learned task-specific biases. **Contrastive Learning** emerged as a generalized framework: instead of predicting a specific label, the model learns to generally distinguish *identity*. Ideally, features of the "same" object (viewed differently) should be close, and features of "different" objects should be far apart.

## Core Ideas and Intuition
The core intuition is **Instance Discrimination**.
*   Every image is its own class.
*   **Positive Pair ($x, x^+$)**: Two augmented views of the *same* image (e.g., crop, color jitter).
*   **Negative Pair ($x, x^-$)**: The image $x$ and any *other* image in the batch.

The model pulls positive pairs together in embedding space and pushes negative pairs apart. This forces the encoder to capture high-level semantic structures (like "dog-ness") that are invariant to low-level noise (cropping, color) while being discriminative enough to distinguish different instances.

## Formalism / Objective
We assume a query view $q$ and a positive key $k^+$ (from the same image) and a set of negatives $\{k^-\}$.

The most common loss is **InfoNCE** (Information Noise Contrastive Estimation) or **NT-Xent** (Normalized Temperature-scaled Cross Entropy) used in SimCLR.

For a pair $(i, j)$ derived from the same image within a batch of size $N$ (resulting in $2N$ views), the loss for one geometric view $z_i$ is:

$$
\mathcal{L}_i = - \log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Where:
*   $z$ are the normalized feature vectors.
*   $\text{sim}(u, v) = u^\top v$ (cosine similarity).
*   $\mathbb{1}_{[k \neq i]}$ is an indicator to exclude self-similarity.
*   $\tau$ is the temperature parameter.

The loss basically asks: *Among all $2N-1$ other samples in the batch, can you identify the one that is my augmented twin?*

## Architecture / Design
A standard Contrastive Learning pipeline (e.g., SimCLR) has:

1.  **Stochastic Data Augmentation**: The engine of contrastive learning.
    *   Random Resized Crop (essential for forcing spatial invariance).
    *   Color Distortion (jitter/grayscale) (essential for preventing color histogram matching).
    *   Gaussian Blur.

2.  **Encoder Base $f(\cdot)$**:
    *   Standard ResNet-50 or ViT. Extracts representation $h = f(x)$.

3.  **Projection Head $g(\cdot)$**:
    *   A small MLP (e.g., Linear $\to$ ReLU $\to$ Linear) mapping $h$ to $z = g(h)$.
    *   **Crucial Insight**: The loss is applied on $z$, but we discard $g(\cdot)$ and use $h$ for downstream tasks. $z$ becomes invariant to augmentations, but $h$ retains more information (like color or orientation) that might be useful for other tasks.

## Training Procedure
*   **Large Batch Sizes**: Like VLMs, larger batches provide more negatives, making the task harder and the gradients more informative. SimCLR used batches up to 4096.
*   **Long Training**: Requires many epochs (800-1000) to converge compared to supervised learning (90).
*   **LARS Optimizer**: Often used for large-batch training to stabilize weights layer-wise.

## Practical Considerations and Pitfalls
*   **Dimensional Collapse**: If the model cheats, it maps everything to the same point. Contrastive loss prevents this by pushing negatives apart.
*   **Shortcut Learning**: If augmentations are too weak, the model matches low-level statistics (e.g., color histograms) rather than semantics. Aggressive color jitter is mandatory.
*   **Temperature**: $\tau$ controls the "hardness" of negatives. Low $\tau$ focuses strictly on the hardest negatives; high $\tau$ treats all negatives more equally. Typical range: 0.1 to 0.5.

## Minimal Implementation Pointer
See [`contrastive_learning.py`](../code/contrastive_learning.py).
*   **What it does**: Simulates the SimCLR forward pass.
*   **Components**: A `SimpleEncoder` (CNN), a projection head, and a function `nt_xent_loss`.
*   **Data**: Generates 2 random "views" for a batch of dummy images.

## References
*   **(Chen et al., 2020) A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**: Simplifies contrastive learning to augmentations + projection head + InfoNCE.
*   **(He et al., 2020) Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)**: Uses a momentum queue to decouple batch size from the number of negatives.
