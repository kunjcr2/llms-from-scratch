# Flamingo: A Vision Language Model

## Overview
Flamingo (Google DeepMind, 2022) is a highly capable **Vision-Language Model (VLM)** designed for **few-shot learning**. Unlike earlier models like CLIP (which was a retrieval model computing similarity between images and text), Flamingo is a true multi-modal, auto-regressive **text-generation model**. It can take interleaved sequences of images, videos, and text as input and successfully generate contextually aware text as output.

## Key Capabilities
1.  **Bridging Pre-trained Models**: Flamingo successfully combines a frozen, pre-trained vision model and a frozen, pre-trained large language model (LLM), bridging them with trainable intermediate layers.
2.  **Handling Interleaved Multi-modal Sequences**: It accepts arbitrary sequences of images and text (e.g., `Image -> Text -> Image -> Text`).
3.  **Video Understanding**: It can process video inputs (sampled at 1 fps) and temporally relate information across multiple frames (e.g., recognizing an object spread across multiple disjoint frames).
4.  **Few-Shot Learning**: Without fine-tuning, Flamingo can rapidly adapt to new tasks (like visual question answering, captioning, or classification) simply by being shown a few examples in the prompt, much like how text-only LLMs operate.

---

## Architecture

Flamingo consists of a frozen vision encoder, a learnable Perceiver Resampler, and a frozen LLM interleaved with learnable Gated Cross-Attention blocks.

### 1. Vision Encoder (Frozen)
*   Typically a ResNet-like or Vision Transformer (ViT) architecture.
*   **Frozen during Flamingo training** to save compute and leverage pre-existing vision capabilities.
*   Converts an image (or frames of a video) into a 2D spatial feature map (a sequence of tokens).
*   For video inputs, spatio-temporal positional embeddings are added so the model can distinguish sequence order.

### 2. Perceiver Resampler (Learnable)
*   **The Problem**: A vision encoder produces a variable number of tokens depending on whether the input is a single frame or a multi-frame video. This variable length is computationally expensive and unstable for the downstream LLM.
*   **The Solution**: The Perceiver Resampler condenses any variable-length visual input feature into a **fixed-length** sequence of visual tokens (e.g., exactly 64 tokens of 768 dimensions).
*   **How it works**: It initializes a fixed number of learnable "latent queries". These latent queries compute cross-attention with the flattened visual features from the vision encoder. The output is a fixed number of context vectors representing the entire visual input, drastically reducing sequence length.

### 3. Gated Cross-Attention Dense Layers (Learnable)
*   The pre-trained LLM transformer blocks are kept strictly frozen.
*   Just before every frozen LLM block, Flamingo inserts a new **Trainable Gated Cross-Attention Layer**.
*   **Cross-Attention**: The text tokens act as the *Query* ($Q$), while the output from the Perceiver Resampler (visual tokens) acts as the *Key* ($K$) and *Value* ($V$). This allows the text generation process to directly attend to the visual context.
*   **$\tanh$ Gating**: A novel initialization strategy where the output of the cross-attention layer is multiplied by $\tanh(\alpha)$, where $\alpha$ is a trainable parameter initialized to $0$.
    *   *Why?* Initially, $\tanh(0) = 0$. This means the entire inserted cross-attention layer outputs zero vectors, essentially neutralizing it. The network behaves *exactly* like the original, un-modified LLM at the start of training. As training progresses, $\alpha$ learns to deviate from 0, smoothly integrating visual information without destroying the LLM's valuable pre-trained knowledge.

---

## Training Data and Chunk Masking

Flamingo was trained on massive datasets scraped from the web, including single image-text pairs, video-text pairs, and most importantly, the **M3W (MultiModal Massive Web)** dataset, which contains interleaved text and images.

### Masking Strategy for Interleaved Data
In an input prompt like `[Image 1] [Text Chunk 1] [Image 2] [Text Chunk 2]`:
*   The text tokens in `[Text Chunk 1]` are only allowed to attend to `[Image 1]`.
*   The text tokens in `[Text Chunk 2]` are only allowed to attend to `[Image 2]`.
*   This specific cross-attention masking is crucial. Without it, `[Text Chunk 1]` could look ahead and attend to `[Image 2]`, allowing the model to cheat by looking into the "future" sequence.

---

## Limitations and Hallucinations
While highly capable, Flamingo models (especially smaller open-source replicas) are prone to **hallucinations**:
*   **Adversarial and Leading Prompts**: If biased with a leading question (e.g., asking "What is out the window?" when there is no window), the model relies heavily on its language prior to fabricate a sensible-sounding but visually incorrect answer (e.g., "a parking lot").
*   **Reasoning/Math Constraints**: They often struggle with spatial reasoning or reading handwritten mathematical equations unless carefully prompted with strong, relevant few-shot examples.
