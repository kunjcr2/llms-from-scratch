# Vision-Language Models (VLMs)

## Problem and Motivation
Standard computer vision models (like ResNet or separate ViTs) are typically trained on closed sets of fixed labels (e.g., ImageNet's 1000 classes). This limits their **generality** and **zero-shot** capabilities. If a new object category appears, the model cannot recognize it without retraining or architectural changes (adding a new head).

Vision-Language Models (VLMs) aim to solve this by learning a **joint embedding space** for images and text. By aligning visual features with semantic text features enabling, they allow for:
*   **Zero-shot classification**: Matching an image to arbitrary text descriptions (e.g., "a photo of a dog").
*   **Open-vocabulary retrieval**: Searching for images using natural language queries.
*   **Grounded understanding**: Linking visual concepts to language for tasks like VQA (Visual Question Answering) or captioning.

Earlier approaches often relied on pre-trained object detectors and complex fusion modules, which were brittle and hard to scale. End-to-end contrastive pre-training (like CLIP) revolutionized this by simply asking: *which text goes with which image?*

## Core Ideas and Intuition
The fundamental shift is moving from **predicting class IDs** (Softmax over $N$ classes) to **matching instances**.

In a dual-encoder VLM (like CLIP or ALIGN):
1.  **Dual Encoders**: You have an independant Image Encoder and Text Encoder.
2.  **Joint Space**: Both encoders project their outputs into a shared $d$-dimensional hypersphere.
3.  **Contrastive Objective**: The model is trained to maximize the similarity of correct (image, text) pairs while minimizing the similarity of incorrect pairings within a batch.

This learns a "multimodal lexicon" where the embedding for an image of a dog is geometrically close to the embedding for the text "dog".

## Formalism / Objective
We operate on a batch of $N$ image-text pairs $\{(I_i, T_i)\}_{i=1}^N$.

Let $f_I(I_i)$ be the image encoder and $f_T(T_i)$ be the text encoder. We project these into the joint embedding space and normalize them:
$$
\mathbf{v}_i = \frac{f_I(I_i)}{\|f_I(I_i)\|}, \quad \mathbf{t}_i = \frac{f_T(T_i)}{\|f_T(T_i)\|}
$$
The similarity score is the cosine similarity scaled by a learnable temperature parameter $\tau$:
$$
\text{sim}(i, j) = \frac{\mathbf{v}_i^\top \mathbf{t}_j}{\tau}
$$
The training objective uses a symmetric cross-entropy loss (InfoNCE style) over the batch. We compute the loss for images (aligning each image to its correct text) and for text (aligning each text to its correct image) and average them.

**Image-to-Text Loss:**
$$
\mathcal{L}_{I \to T} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(i, i))}{\sum_{j=1}^N \exp(\text{sim}(i, j))}
$$

**Text-to-Image Loss:**
$$
\mathcal{L}_{T \to I} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(i, i))}{\sum_{j=1}^N \exp(\text{sim}(j, i))}
$$

**Total Loss:**
$$
\mathcal{L} = \frac{\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}}{2}
$$

The denominator acts as the negative sampling mechanism, pushing apart the embeddings of mismatched pairs.

## Architecture / Design
A typical Dual-Encoder VLM consists of:

1.  **Image Encoder**:
    *   **Backbone**: Typically a ResNet-50 or a Vision Transformer (ViT-B/32, ViT-L/14).
    *   **Global Representation**: For ViT, the `[CLS]` token is used. For ResNet, Global Average Pooling.
    *   **Projection Head**: A linear layer (or MLP) transforming the backbone output dimension (e.g., 768) to the joint embedding dimension (e.g., 512).

2.  **Text Encoder**:
    *   **Backbone**: Usually a Transformer (BERT-style or GPT-style), but often smaller/shallower than LLMs.
    *   **Tokenization**: BPE or WordPiece. Key design choice: masking (causal vs bidirectional) depends on the exact model variant (CLIP uses causal).
    *   **Representation**: The feature from the `[EOS]` token (or `[CLS]`) is treated as the global sentence vector.
    *   **Projection Head**: Similar to the image side, projects to the same joint dimension $d$.

3.  **Temperature**:
    *   $\tau$ is essentially a scaling factor for the logits. It is usually a learnable parameter, clipped to prevent numerical instability (e.g., $\ln(1/\tau)$ clamped to max 100).

## Training Procedure
*   **Data**: Requires massive internet-scale datasets (e.g., LAION-400M/5B, WebImageText) of (image, alt-text) pairs.
*   **Preprocessing**:
    *   Images: Random resized crops are crucial.
    *   Text: Truncated to a fixed length (e.g., 77 tokens for CLIP).
*   **Batch Size**: **Crucial.** Since in-batch negatives are the only source of "wrong" answers, larger batch sizes (32k, 64k) provide a harder contrastive task and better learning signal.
*   **Optimization**: AdamW with cosine learning rate decay and warmup.

## Practical Considerations and Pitfalls
*   **Batch Size Dependency**: If the batch size is too small, the implementation of negatives is weak, and the model won't learn robust features.
*   **Polysemanticity**: Simply maximizing cosine similarity doesn't guarantee the model understands *relations* (e.g., "man riding horse" vs "horse riding man") well without specific architectural priors or hard negatives.
*   **Text Quality**: Alt-text is often noisy. Filtering logic (removing short captions, non-English usually) is a major part of the pipeline.
*   **Projection Dimension**: Higher isn't always better. 512-1024 is standard. Lower dims (64-128) cause information bottlenecks.

## Minimal Implementation Pointer
See [`vision_language_models.py`](../code/vision_language_models.py) for a self-contained toy example.
*   **What it does**: Defines a `SimpleVLM` with a tiny CNN image encoder and a tiny Transformer text encoder.
*   **Constraints**: Runs on CPU with random tensors. It demonstrates the projection and symmetric loss calculation.
*   **Key Function**: `contrastive_loss(image_features, text_features, temperature)`

## References
*   **(Radford et al., 2021) Learning Transferable Visual Models From Natural Language Supervision (CLIP)**: The seminal paper introducing the scalable dual-encoder contrastive approach.
*   **(Jia et al., 2021) Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN)**: Showed that with enough noisy data (1B+ pairs), strict filtering is less critical.
