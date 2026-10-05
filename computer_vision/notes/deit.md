# DeiT (Data-efficient Image Transformers)

## Problem: The "Hunger for Data"
The original Vision Transformer (ViT) was a breakthrough, showing that Transformers could match CNNs in computer vision. However, it had one massive drawback: **it was extremely data-hungry**. 

While a ResNet (CNN) could perform well on ImageNet-1K (1.3M images), a ViT trained on the same data would underperform significantly. ViT only truly excelled when pre-trained on massive proprietary datasets like JFT-300M. 

### Why is ViT so data-hungry? (The Inductive Bias Gap)
To understand why, we need to talk about **Inductive Bias**. Think of inductive bias as "pre-baked assumptions" about the data:
*   **CNNs have high inductive bias**: They naturally assume that images have *locality* (pixels near each other are related) and *translation invariance* (a cat is a cat whether it's in the top-left or bottom-right). This is "hard-coded" via sliding windows (convolutions).
*   **Transformers have low inductive bias**: They start with a "blank canvas." A Transformer doesn't know what an image is; it just sees a sequence of patches. It has to *learn* the concept of locality from scratch.

Because Transformers assume so little, they need **vast amounts of data** to figure out the basic rules of vision that CNNs get for free.

## The DeiT Solution: Efficiency through "Teaching"
DeiT (Data-efficient Image Transformers) was designed to train high-performance ViTs on standard datasets (like ImageNet-1K) without needing hundreds of millions of extra images. It does this through two main pillars:

1.  **The Rescue Recipe**: Aggressive data augmentation and regularization to "force" the model to learn locality and patterns.
2.  **Hard-Label Distillation**: "Borrowing" the inductive bias of a CNN. Instead of learning only from raw labels, the Transformer (the student) also watches a pre-trained CNN (the teacher) and tries to mimic its decisions.

### The Distillation Token: A New Specialty
DeiT introduces a special **Distillation Token** (`[DIST]`).
*   The `[CLS]` token focuses on predicting the **true label** (e.g., "Is this a dog?").
*   The `[DIST]` token focuses on predicting what the **CNN teacher predicts**.

By having a dedicated token that specifically mimics the CNN, the Transformer "absorbs" the CNN's locality and spatial understanding without needing a CNN architecture itself.

## Formalism / Objective
Let $Z_s$ be the student logits and $Z_t$ be the teacher logits. Let $y$ be the ground-truth label.

DeiT found that **Hard Distillation** works best: the student tries to predict the teacher's "hard" choice (the most likely class) rather than its full probability distribution.
$$
\mathcal{L}_{\text{hardDistill}} = \frac{1}{2} \text{CE}(\psi(Z_{\text{cls}}), y) + \frac{1}{2} \text{CE}(\psi(Z_{\text{distill}}), y_{\text{teacher}})
$$
where $y_{\text{teacher}} = \text{argmax}(Z_t)$. At test time, the final prediction is the average of both the `[CLS]` and `[DIST]` heads.

## Architecture / Design
The architecture is identical to a standard ViT, except for the input sequence:
1.  **Patches**: Image is split into fixed-size patches (e.g., $16 \times 16$).
2.  **Tokens**: Flattened patches + Position Embeddings.
3.  **Special Tokens**:
    *   `[CLS]`: Standard class token.
    *   `[DIST]`: **New distillation token**.
4.  **Interaction**: Self-attention layers allow `[CLS]` and `[DIST]` to interact with image patches and each other.

Crucially, the `[DIST]` token learns to mimic what the CNN teacher "sees", effectively injecting CNN-like inductive biases (locality) into the Transformer through the loss function. This allows the Transformer to benefit from CNN-style spatial assumptions while maintaining its own global attention capabilities.

## Training Procedure
The "DeiT recipe" is as famous as the architecture itself because ViTs are prone to overfitting on small data.
*   **Augmentations**:
    *   RandAugment suitable for Transformers.
    *   Mixup and CutMix (essential).
    *   Random Erasing.
*   **Regularization**:
    *   **Stochastic Depth**: Randomly dropping layers during training (row pruning in the residual branches).
    *   **Weight Decay**: carefully tuned (e.g., 0.05).
*   **Optimizer**: AdamW. SGD generally works poorly for ViTs compared to CNNs.

## Practical Considerations and Pitfalls
*   **Teacher Choice**: A ConvNet teacher typically works better than a Transformer teacher for a Transformer student, likely because the ConvNet teaches the "missing" inductive biases.
*   **Hyperparameters**: The recipe is brittle. Dropping Mixup or Stochastic Depth usually causes performance to collapse by huge margins (5-10% accuracy).
*   **Resolution**: Fine-tuning at higher resolution (e.g., train at 224, test at 384) requires interpolating positional embeddings.

## References
*   **(Touvron et al., 2021) Training data-efficient image transformers & distillation through attention**: The core DeiT paper.
