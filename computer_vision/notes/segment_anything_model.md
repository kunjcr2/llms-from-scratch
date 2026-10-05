# Segment Anything Model (SAM)

## Problem and Motivation
Previous segmentation models (e.g., Mask R-CNN) were trained on fixed class sets — predict masks for cats and dogs, and the model only knows cats and dogs. **SAM** (Meta, 2023, ~16k citations) aims to be a **class-agnostic foundation model** for segmentation: click anywhere in any image, and get a mask around whatever you clicked, regardless of object type.

**Segmentation vs Detection**: Detection draws bounding boxes (coarse, rectangular). Segmentation produces **pixel-level masks** — every pixel is classified, giving precise object boundaries. Segmentation is strictly harder than detection.

### Types of Segmentation
| Type | Description |
|------|-------------|
| **Semantic** | All pixels of the same class share the same label/color (3 people → all same mask) |
| **Instance** | Same class but different instances get different labels (3 people → 3 separate masks) |

SAM is essentially universal — it can perform any style of segmentation via **promptable segmentation**.

## Core Ideas and Intuition

### Promptable Segmentation
Unlike traditional models where the output is fixed at training time, SAM takes **two inputs** at inference:
1. **Image** — the scene to segment
2. **Prompt** — tells the model *what* to segment

The model is class-agnostic: it doesn't predict "this is a cat," it predicts "here is a mask around whatever the prompt indicated."

### Four Prompt Types
| Prompt | Format | Category |
|--------|--------|----------|
| **Point** | $(x, y)$ coordinate + foreground/background label | Sparse |
| **Box** | Two corners $(x_1, y_1), (x_2, y_2)$ | Sparse |
| **Mask** | Rough binary mask ($1 \times 1024 \times 1024$) | Dense |
| **Text** | Natural language (requires CLIP; not natively supported) | Via CLIP |

**Sparse prompts** are just coordinate-based inputs. **Dense prompts** (masks) carry spatial structure matching the image dimensions.

### Handling Ambiguity — Three-Level Masks
A single point click is ambiguous — clicking on a bird's head could mean "the head," "the body," or "the whole bird." SAM resolves this by outputting **three masks per forward pass** at different granularity levels:

```
Click on bird's head →  Mask 0: Whole bird (global)
                        Mask 1: Body without legs (sub-global)
                        Mask 2: Head only (local)
```

Each mask also gets an **IoU confidence score** predicting how good that mask is.

### Text Prompts via CLIP
SAM doesn't take text directly. Instead:
1. During training, **CLIP image embeddings** of the target objects were used as prompt inputs
2. Since CLIP aligns text and image into the **same embedding space**, at inference you can pass CLIP text embeddings instead
3. This only works if you have a pre-trained CLIP model available

## Architecture / Design

SAM has **three components**:

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│ Image Encoder │     │ Prompt Encoder  │     │ Mask Decoder  │
│   (Part 1)   │     │   (Part 2)      │     │   (Part 3)   │
│  ViT (heavy) │     │ No transformer  │     │  Transformer │
└──────┬───────┘     └───────┬─────────┘     └──────┬───────┘
       │                     │                      │
  256×64×64            Sparse: N×256           3 Masks + IoU
                       Dense: 256×64×64
```

**Key insight for fast inference**: The image encoder (heaviest part) runs **only once** per image. Prompt encoder + mask decoder are lightweight, so clicking different points on the same image is near-instant.

---

### Part 1: Image Encoder (MAE Pre-trained ViT)

A standard Vision Transformer pipeline, but pre-trained with a **Masked Autoencoder (MAE)**:

**Dimension flow**:
$$
(3, 1024, 1024) \xrightarrow{\text{patch 16×16}} (4096, 768) \xrightarrow{\text{project}} (4096, 1280) \xrightarrow{\text{ViT}} (4096, 1280) \xrightarrow{\text{reshape}} (1280, 64, 64) \xrightarrow{\text{project}} (256, 64, 64)
$$

| Step | Operation | Dimensions |
|------|-----------|------------|
| Input image | — | $3 \times 1024 \times 1024$ |
| Divide into 16×16 patches | Patchify | $4096$ tokens, each $768$-dim ($16 \times 16 \times 3$) |
| Linear projection | Project to ViT embedding dim | $4096 \times 1280$ |
| ViT encoder + position embeddings | Self-attention blocks | $4096 \times 1280$ (context vectors) |
| Reshape to 2D grid | Un-flatten | $1280 \times 64 \times 64$ |
| Project to decoder dim | $1 \times 1$ conv | $256 \times 64 \times 64$ |

The final $256$ is the embedding dimension expected by the mask decoder. Spatial info ($64 \times 64$) is preserved — unlike classification ViTs that use only a CLS token, SAM uses **all** patch tokens.

#### What is a Masked Autoencoder (MAE)?
A self-supervised pre-training strategy:
1. **Mask ~75% of patches** from the input image randomly
2. Pass the **visible 25%** through a ViT encoder → embeddings
3. Add **learnable mask tokens** for hidden patches, then pass everything through a decoder
4. **Reconstruct** the original pixel values of the masked patches

The encoder learns deep image understanding because it must infer hidden content from partial views. This is analogous to BERT's masked token prediction but for images. The resulting ViT encoder is then used as SAM's image encoder.

---

### Part 2: Prompt Encoder (No Transformer)

Surprisingly, the prompt encoder contains **no transformer** — it's purely projections and convolutions.

It has **two branches**:

#### Sparse Branch (Points + Boxes)

**Point prompts** ($n$ points):
- Each point: $(x, y, \text{label})$ where label ∈ {foreground, background}
- Normalized coordinates + learnable positional encoding + type embedding → projected to $256$-dim
- Output: $n \times 256$

**Box prompts** ($m$ boxes):
- Each box = 2 corner points: $(x_1, y_1)$ and $(x_2, y_2)$
- Each corner gets its own type embedding (corner-1 vs corner-2)
- Output: $2m \times 256$

**Type embeddings**: Learnable 256-dim vectors that encode *what kind* of input this is (point, corner-1, corner-2). There are only 3 type vectors total regardless of how many points/boxes there are.

#### Dense Branch (Mask)

**Mask prompt**:
$$
(1, 1024, 1024) \xrightarrow{\text{Conv2D}} (256, 64, 64)
$$

Channel dimension expands $1 \to 256$, spatial dimension shrinks $1024 \to 64$ — same shape as image encoder output. This mask embedding is **element-wise added** to the image encoder output before entering the decoder.

#### Output Tokens (Mask Queries)
Additionally, **4 learnable tokens** of dim $256$ are created:
- 3 tokens → will become the 3 output masks (one per ambiguity level)
- 1 token → will become the IoU score predictions

These are analogous to **object queries** in DETR.

#### Final Concatenation
All sparse outputs + output tokens are concatenated along the token dimension:

$$
\underbrace{n}_{\text{points}} + \underbrace{2m}_{\text{box corners}} + \underbrace{4}_{\text{mask/IoU tokens}} = (n + 2m + 4) \times 256
$$

This combined sequence is one input to the mask decoder. The other input is the $256 \times 64 \times 64$ result of element-wise adding image embeddings + dense mask embeddings.

---

### Part 3: Mask Decoder (Lightweight Transformer)

The decoder receives two streams:
1. **Token sequence**: $(n + 2m + 4) \times 256$ — sparse prompts + output tokens
2. **Image sequence**: $256 \times 64 \times 64$ — image embedding (+ mask embedding if provided), flattened to $4096 \times 256$ for attention

#### Decoder Block (repeated 2×)

```
Token Sequence ──► [Self-Attention] ──► [Token-to-Image Cross-Attention] ──► [MLP] ──►
                                              ▲                                       │
Image Sequence ────────────────────────────────┘                                      │
      ▲                                                                               │
      └──────────────── [Image-to-Token Cross-Attention] ◄────────────────────────────┘
```

| Step | Query Source | Key/Value Source | Output Length |
|------|-------------|-----------------|---------------|
| Self-Attention | Token seq | Token seq | $n + 2m + 4$ |
| Token→Image Cross-Attn | Token seq | Image seq (4096) | $n + 2m + 4$ |
| Image→Token Cross-Attn | Image seq (4096) | Token seq | $4096$ → reshape to $64 \times 64$ |

This bidirectional cross-attention block runs **2×** in series.

#### Final Mask Generation

After the 2× decoder blocks, one more **token-to-image cross-attention** is applied. Then:

**For masks** (3 output tokens):
1. Each of the 3 mask tokens → separate MLP → $256$-dim vector
2. Image features → 2× transposed convolution upscaling: $(256, 64, 64) \to (256, 256, 256)$
3. **Dot product** of each $256$-dim mask token with the $256$ channel dimension → collapses channels to 1
4. Result: $(1, 256, 256)$ per mask → bilinear interpolation → $(1, 1024, 1024)$

**For IoU scores** (1 output token):
- IoU token → MLP → 3 scalar scores (one confidence per mask)

```
3 Mask Tokens ──► 3 MLPs ──► 3 × (256,) ──┐
                                           ├──► dot product ──► 3 × (1, 256, 256) ──► interpolate ──► 3 × (1, 1024, 1024)
Image Features ──► 2× Upsample ──► (256, 256, 256) ──┘

IoU Token ──► MLP ──► 3 scores
```

## Dataset: SA-1B

SAM was trained on a custom dataset of **11 million images** with **1.1 billion masks** (~100 masks/image on average).

### Three-Stage Data Engine
Creating 1.1B masks manually is impossible, so Meta used an iterative approach:

| Stage | Method | Quality | Quantity |
|-------|--------|---------|----------|
| **1. Manual** | Human annotators create masks from scratch | Highest | Low |
| **2. Semi-Auto** | SAM proposes masks from point/box prompts; humans correct/refine | High | Medium |
| **3. Fully Auto** | SAM generates masks on millions of images; no human in the loop | Lower (but compensated by scale) | Highest |

Stage 3 masks may contain errors but add diversity — the model finds objects humans might not think to annotate, reducing human annotation bias. All three stages' data are combined for the final SAM training.

## Training Procedure (What the Lecture Skipped)

This is where things get concrete. The lecture explained the architecture but was vague about how SAM actually *learns*. Here's the full picture.

### What Does a Single Training Step Look Like?

The training data consists of **(image, ground-truth mask, simulated prompt)** triplets. Here is exactly what happens in one forward pass + backward pass:

```
Step 1: Sample an image + one of its ground-truth masks from SA-1B

Step 2: Simulate a prompt from the ground-truth mask
        (e.g., sample a random point inside the mask,
         or derive a bounding box around the mask)

Step 3: Forward pass
        Image ──► Image Encoder ──► image embedding (256×64×64)
        Prompt ──► Prompt Encoder ──► token sequence
        Both ──► Mask Decoder ──► 3 predicted masks + 3 IoU scores

Step 4: Compare each predicted mask against the ground-truth mask
        Pick the best-matching predicted mask (lowest loss)

Step 5: Compute loss on that best mask + its IoU score

Step 6: Backpropagate gradients through decoder ──► encoders ──► update weights
```

### How Are Prompts Created During Training?

At inference, a human provides prompts by clicking. During training, there is no human — **prompts are simulated from the ground-truth masks**:

| Simulated Prompt | How It's Generated |
|------------------|--------------------|
| **Point** | Sample a random $(x, y)$ coordinate from inside the ground-truth mask region. Label = foreground. Optionally sample background points outside the mask. |
| **Box** | Compute the tight bounding box around the ground-truth mask, optionally add random noise/jitter to corners. |
| **Mask** | Use the output (logit mask) from a *previous iteration* as a coarse mask input. This is how SAM learns iterative refinement. |

So the model never sees human clicks during training — it sees automatically derived prompts that mimic what a human would do. This is what makes it generalizable to any prompt at inference.

### The Loss Function — Three Components

After the forward pass, SAM produces **3 predicted masks** (because of ambiguity) and **3 IoU scores**. The loss is computed as follows:

#### Step 1: Select the Best Mask
Out of the 3 predicted masks, compute the loss of each against the single ground-truth mask. **Only backpropagate through the mask with the lowest loss** (the best one). This is critical — it means each of the 3 output heads can specialize in a different granularity level without being penalized for the other two.

#### Step 2: Mask Loss (Focal Loss + Dice Loss)
The selected mask is compared pixel-by-pixel against the ground-truth mask using a **linear combination** of two losses:

$$
\mathcal{L}_{\text{mask}} = \lambda_{\text{focal}} \cdot \mathcal{L}_{\text{focal}} + \lambda_{\text{dice}} \cdot \mathcal{L}_{\text{dice}}
$$

**Focal Loss** — a modified binary cross-entropy:
$$
\mathcal{L}_{\text{focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)
$$

- $p_t$ = predicted probability for the correct class (foreground or background) at each pixel
- $(1 - p_t)^\gamma$ = **down-weights easy pixels** (clearly background), **up-weights hard pixels** (ambiguous boundaries)
- Why needed: In a $1024 \times 1024$ mask, >95% of pixels might be background. Without focal loss, the model could predict "all background" and still get low loss. Focal loss forces it to get the foreground pixels right.

**Dice Loss** — measures overlap between predicted and ground-truth mask:
$$
\mathcal{L}_{\text{dice}} = 1 - \frac{2 |P \cap G|}{|P| + |G|}
$$

- $P$ = set of predicted foreground pixels, $G$ = ground-truth foreground pixels
- If prediction perfectly matches ground truth: dice = 1, loss = 0
- If no overlap at all: dice = 0, loss = 1
- Why needed: Dice loss directly optimizes the overlap metric, complementing focal loss which operates per-pixel independently

#### Step 3: IoU Prediction Loss (MSE)
The IoU head predicts a confidence score for each mask. For the selected best mask, its predicted IoU is compared against the **actual IoU** between that predicted mask and the ground truth:

$$
\mathcal{L}_{\text{IoU}} = \text{MSE}(\hat{\text{IoU}}, \text{IoU}_{\text{actual}})
$$

Where $\text{IoU}_{\text{actual}} = \frac{|P \cap G|}{|P \cup G|}$ is computed on the fly from the predicted and ground-truth masks.

This teaches the model to be self-aware — to know *how good its own mask is*. At inference, you use these scores to pick the best mask.

#### Total Loss
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \mathcal{L}_{\text{dice}} + \mathcal{L}_{\text{IoU}}
$$

(with appropriate weighting coefficients, which the paper sets to 20:1:1 for focal:dice:IoU)

### What Is Trainable vs Pre-trained?

| Component | Pre-trained? | Trainable during SAM training? |
|-----------|-------------|-------------------------------|
| **Image Encoder (ViT-H)** | Yes — MAE pre-trained on ImageNet | Yes — fine-tuned end-to-end (but slowly, small learning rate) |
| **Prompt Encoder** | No | Yes — learned from scratch (it's just learned embeddings + projections) |
| **Mask Decoder** | No | Yes — learned from scratch |
| **Type/Position Embeddings** | No | Yes — learned from scratch |
| **Output Token Embeddings** | No | Yes — the 4 learnable tokens (3 mask + 1 IoU) are learned from scratch |

The image encoder starts pre-trained (MAE gives it strong image understanding) and is fine-tuned. Everything else is trained from scratch. Gradients flow **all the way back** from the loss through the decoder, through cross-attention, back into both encoder outputs.

### Iterative Mask Refinement During Training

One subtlety: SAM can take a **mask as a dense prompt**. During training, this is exploited for iterative refinement:

1. **First pass**: Give a point/box prompt → get a predicted mask
2. **Second pass**: Feed the predicted mask (from step 1) back as a dense prompt → get a refined mask
3. Compute loss on the refined mask

This teaches the model to take a rough/wrong mask input and improve it — which is useful during the semi-automatic annotation stages (Stage 2 of the data engine) where humans provide coarse corrections.

### Connecting Training to the Data Engine

The training and data creation are **interleaved**, not sequential:

```
Stage 1: Train SAM on manually-annotated masks (small dataset, high quality)
              ↓
Stage 2: Use partially-trained SAM to propose masks
         Humans correct → add corrected masks to training data
         Retrain SAM on expanded dataset
              ↓
Stage 3: Use better SAM to auto-generate masks on 11M images
         No human correction → add all to training data
         Train final SAM on entire SA-1B dataset
```

Each stage produces a better model AND a bigger dataset. The final SAM is trained on all data from all three stages combined.

## Practical Considerations

*   **Inference Speed**: Image encoder (ViT-H/L) runs once per image (~heavyweight). Prompt encoder + mask decoder run per-click (~lightweight, near real-time).
*   **Three Masks Always**: Every forward pass produces 3 masks at different granularity levels plus IoU scores. Select the one you need.
*   **No Direct Text Input**: Text prompts require a separate CLIP model to generate embeddings first.
*   **SAM Versions**: The original paper describes SAM v1. SAM 2 and SAM 3 extend to video and 3D respectively.

## Quick Inference Example

```python
from segment_anything import sam_model_registry, SamPredictor
import cv2

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Set image (runs image encoder once)
image = cv2.imread("image.jpg")
predictor.set_image(image)

# Predict with point prompt
input_point = np.array([[600, 400]])     # (x, y)
input_label = np.array([1])              # 1 = foreground
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True                # returns 3 masks
)

# masks[0] = global, masks[1] = sub-global, masks[2] = local
```

## Comparison: SAM vs Traditional Segmentation

| Aspect | Traditional (Mask R-CNN) | SAM |
|--------|--------------------------|-----|
| Classes | Fixed set (trained classes only) | Class-agnostic (anything) |
| Input | Image only | Image + Prompt |
| Output | Class-specific masks | 3 ambiguity-level masks + IoU |
| Prompts | None | Points, Boxes, Masks, Text (via CLIP) |
| Reusability | Retrain for new domains | Zero-shot generalization |
| Architecture | CNN-based | ViT encoder + lightweight decoder |

## References
*   **(Kirillov et al., 2023) Segment Anything**: Original SAM paper from Meta AI, ~16k citations. Introduced promptable segmentation and the SA-1B dataset.
*   **(He et al., 2022) Masked Autoencoders Are Scalable Vision Learners**: MAE pre-training strategy used for SAM's image encoder.
*   **(Radford et al., 2021) Learning Transferable Visual Models From Natural Language Supervision (CLIP)**: Enables text-based prompting in SAM via shared image-text embedding space.
