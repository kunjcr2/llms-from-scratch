# I-JEPA Preliminaries

Before reading the **I-JEPA** (Image Joint-Embedding Predictive Architecture) paper by Yann LeCun et al. (2023), it is helpful to understand the shift in self-supervised learning paradigms in computer vision. This document covers self-supervised learning (SSL) vs. supervised learning, the mechanics of Masked Autoencoders (MAE), the Joint-Embedding Predictive Architecture (JEPA) concept, and the theoretical foundations of Energy-Based Models (EBMs) and representation collapse.

---

## 1. Self-Supervised Learning Paradigms in Vision

In standard supervised learning, models learn using human-annotated labels. In **Self-Supervised Learning (SSL)**, models learn representations from unlabeled data by solving a "pretext task" where the labels are generated automatically from the data itself.

In computer vision, SSL methods generally fall into three main paradigms:

### A. Invariance-Based (Contrastive / Non-Contrastive) Methods
These methods learn representations by processing two or more augmented views of the same image. The goal is to make the representations invariant to semantic-preserving transformations (like random crops, color jittering, or blur).
*   **Contrastive Learning**: Pulls positive pairs (augmented views of the same image) close in representation space, while actively pushing negative pairs (views of different images) apart.
*   **Non-Contrastive Learning**: Learns invariant representations without negative pairs, using explicit regularization (e.g., variance/covariance constraints) to prevent all representations from collapsing to a single point.

### B. Generative / Reconstruction-Based Methods
These methods learn by predicting or reconstructing corrupted or masked parts of the input in the input (pixel) space. An example is the **autoencoder** (a type of neural network trained to compress input data into a lower-dimensional code and then reconstruct it back as closely as possible to the original input).

### C. Predictive (Joint-Embedding) Methods
This is the paradigm introduced by Yann LeCun's JEPA. Instead of reconstructing pixels, these methods predict the representation of a target region in a latent space, given a context region.

---

## 2. Masked Autoencoders (MAE): Pixel-Level Generation

**Masked Autoencoders** (a self-supervised learning technique where a model is trained to reconstruct missing patches of an image in pixel space from the remaining visible patches) are a highly popular generative SSL method.

```
                  ┌───────────────────┐
                  │   Masked Image    │  (e.g., 75% patches removed)
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  ViT Encoder      │  (only processes visible patches)
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  ViT Decoder      │  (processes visible + mask tokens)
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ Reconstructed Pix │  (compared to original pixels)
                  └───────────────────┘
```

### How MAE Works
1.  An image is split into patches (e.g., $16 \times 16$ pixels).
2.  A high percentage of patches (typically $75\%$ to $85\%$) are randomly masked and removed.
3.  The lightweight encoder (a Vision Transformer) processes only the remaining visible patches to produce latent tokens.
4.  The decoder (another Vision Transformer) takes the encoded visible tokens plus learnable mask tokens and attempts to reconstruct the original pixel values of the missing patches.

### The Objective
The loss function is the Mean Squared Error (MSE) computed in pixel space only over the masked patches:

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \|x_i - \hat{x}_i\|^2
$$

Where:
*   $M$ is the set of masked patches.
*   $x_i$ is the original pixel values of patch $i$.
*   $\hat{x}_i$ is the reconstructed pixel values of patch $i$.

### Limitations of Pixel Reconstruction (LeCun's Critique)
Yann LeCun argues that generative/reconstruction methods are highly inefficient for representation learning:
*   **Irrelevant High-Frequency Noise**: Pixel-space reconstruction forces the model to spend capacity learning low-level details (like the exact texture of fur, individual grass blades, or sensor noise) that are irrelevant for understanding the high-level semantic meaning of the scene.
*   **Aesthetic focus over semantics**: Reconstructing a realistic image is not the same as understanding it. A model can generate plausible pixels without learning structural concepts or object relations.

---

## 3. Joint-Embedding Predictive Architecture (JEPA)

To address the limitations of pixel-level reconstruction, LeCun proposed the **Joint-Embedding Predictive Architecture** (a self-supervised learning framework that learns representations by predicting the latent features of target regions from context regions, rather than reconstructing pixels).

Unlike generative architectures (which predict pixels) or standard **Joint-Embedding Architectures** (architectures that process two different views or parts of an input through two separate encoders to obtain representative vectors, attempting to maximize their similarity under positive pairings without generating the input), JEPA predicts representations of target regions *from* context regions using a learned predictor in latent space.

```
                              ┌───────────────┐
                              │  Input Image  │
                              └───────┬───────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                     │
         ┌─────────▼─────────┐                 ┌─────────▼─────────┐
         │   Context Block   │                 │   Target Blocks   │
         │     (Visible)     │                 │     (Masked)      │
         └─────────┬─────────┘                 └─────────┬─────────┘
                   │                                     │
         ┌─────────▼─────────┐                 ┌─────────▼─────────┐
         │  Context Encoder  │                 │  Target Encoder   │
         └─────────┬─────────┘                 └─────────┬─────────┘
                   │                                     │
             $s_x$ (latent)                        $s_y$ (latent)
                   │                                     │
         ┌─────────▼─────────┐                           │
         │     Predictor     ├───────┐                   │
         └───────────────────┘       │                   │
                                     │                   │
                            $\hat{s}_y$ (predicted)       │
                                     │                   │
                                     ▼                   ▼
                                  [ L2 Distance Loss ]
```

### Components of I-JEPA
**I-JEPA** (Image Joint-Embedding Predictive Architecture, a specific instantiation of JEPA applied to image data using Vision Transformers) uses three networks:

1.  **Context Encoder**: Processes a large, visible "context" block of the image to output context representations $s_x$.
2.  **Target Encoder**: Processes the entire image to output representations $s_y$ for the masked target blocks. The target encoder weights are updated using an **exponential moving average** (a weight-updating strategy where target encoder weights are updated as a slowly changing combination of current and past context encoder weights rather than direct gradient updates) of the context encoder.
3.  **Predictor**: A lightweight network that takes context representations $s_x$ along with the positional information of the target blocks (spatial coordinates) and predicts the target representations $\hat{s}_y$.

### The Objective
I-JEPA is trained by minimizing the $L_2$ distance between the predicted representations and the target encoder's output:

$$
\mathcal{L}_{\text{I-JEPA}} = \sum_{y \in Y} \| \text{Predictor}(s_x, \text{pos}_y) - s_y \|^2_2
$$

Where:
*   $Y$ represents the set of target blocks.
*   $s_x$ is the context representation.
*   $\text{pos}_y$ is the positional/spatial embedding of target block $y$.
*   $s_y$ is the actual representation computed by the target encoder.

---

## 4. Advanced Theoretical Foundations

### Energy-Based Models (EBMs)
JEPA is framed under the theory of **Energy-Based Models** (a machine learning framework that models data distributions by associating a scalar energy value with each configuration, where lower energy corresponds to high-compatibility/real configurations and higher energy to incompatible/unrealistic configurations).

In an EBM, training aims to shape an energy function $E(x, y)$ such that:
*   If $x$ and $y$ are compatible (e.g., $x$ is the context and $y$ is the corresponding target block representation from the same image), the energy $E(x, y)$ is small.
*   If $x$ and $y$ are incompatible (e.g., $y$ is from a different image or space), the energy $E(x, y)$ is large.

In I-JEPA, the energy is defined as the predictor's error in representation space:
$$
E(x, y) = \| \text{Predictor}(s_x, \text{pos}_y) - s_y \|^2_2
$$

### Collapse Prevention
A major challenge in joint-embedding models is **representation collapse** (a failure mode in self-supervised learning where the encoder maps all inputs to a single constant vector, rendering the representations useless). If the encoders collapse, the energy is always zero, but the model has learned nothing.

I-JEPA prevents representation collapse without needing negative samples (as in contrastive methods) or explicit covariance regularization (as in VICReg) through:
1.  **Asymmetrical Architecture**: The use of different networks for context ($s_x$) and target ($s_y$), where the target encoder is updated via Exponential Moving Average (EMA). The target encoder never receives gradients directly.
2.  **Spatially Guided Predictor**: Because the predictor must use positional coordinates to map a context representation to a target representation, it is impossible for the model to satisfy the loss with a simple flat, constant representation across different spatial coordinates.
3.  **Information Bottleneck**: The predictor is bottlenecked (has lower capacity than the encoders), which prevents it from learning identity mappings and forces the representation to capture macro-level semantic structures.
