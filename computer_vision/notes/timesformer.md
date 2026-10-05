# TimeSformer (Time-Space Transformer)

## Problem and Motivation
Previous lectures applied transformers to images for classification, detection, and segmentation. **TimeSformer** extends the Vision Transformer to **video understanding** — classifying actions in videos by attending across both space and time.

Key insight: Videos are sequences of frames. Some actions (e.g., bicep curl vs. Zottman curl) look nearly identical in a single frame and require **temporal information** to distinguish. Others (e.g., basketball vs. volleyball) can be classified from a single snapshot using only **spatial information**.

**Paper**: *"Is Space-Time Attention All You Need for Video Understanding?"* — Facebook AI (~3,300 citations)

## Core Ideas and Intuition

### Video as Input
Unlike images, a video has an additional **frame dimension** $F$:
- **Image**: $W \times H$ pixels
- **Video**: $W \times H \times F$ pixels (no audio — audio is not considered)

### Token Counts
- **Spatial tokens** per frame: $N = \frac{W}{P} \times \frac{H}{P} = \left(\frac{\text{image size}}{P}\right)^2$ where $P$ is patch size
- **Temporal tokens**: $F$ (one per frame)
- **Total tokens**: $N \times F + 1$ (the $+1$ is for the CLS token used for video classification)
- For a plain ViT (single image): tokens = $N + 1$

### Kinetics-400 Dataset
- ~400 human action classes (running, lifting weights, arm wrestling, etc.)
- Short videos (~8–10 seconds each)
- Each video → one action class label
- For this implementation: **binary classification** (bench press vs. deadlift) using a small subset

## Attention Variants

The paper explores five attention strategies. For a given query patch at frame $t$:

### 1. Space-Only Attention (S)
- Query attends **only to patches within the same frame**
- No temporal information — sufficient when appearance alone distinguishes classes
- Keys: all patches in frame $t$

### 2. Joint Space-Time Attention (ST)
- Query attends to **every patch in every frame**
- Maximum context but **very expensive**
- Keys: all patches across all frames

### 3. Divided Space-Time Attention (T+S)
- **Best performer** — what we implement
- Two separate attention steps:
  1. **Temporal attention**: query attends to the **same spatial patch across all frames**
  2. **Spatial attention**: query attends to **all patches within the same frame**
- Separate learnable $W_Q, W_K, W_V$ matrices for temporal and spatial attention

### 4. Sparse Local-Global Attention
- Local attention to immediate neighboring patches (in space and time)
- Sparse global attention to patches one step beyond neighbors
- Reduces computation while capturing both local and global context

### 5. Axial Attention
- Spatial attention decomposed along y-axis then x-axis separately
- Temporal attention on the same patch across time

## Attention Complexity Analysis

### Joint Space-Time Attention
| Metric | Formula |
|--------|---------|
| Total queries | $NF + 1$ |
| Total keys | $NF + 1$ |
| **Total complexity** | $\mathcal{O}((NF)^2) = \mathcal{O}(N^2 F^2)$ |
| **Per-token complexity** | $\mathcal{O}(NF)$ |

As $F$ increases (longer videos), complexity grows **quadratically** — impractical for long or high-resolution videos.

### Divided Space-Time Attention
| Metric | Formula |
|--------|---------|
| Total queries | $NF + 1$ |
| Temporal keys per query | $F + 1$ |
| Spatial keys per query | $N + 1$ |
| Total keys | $N + F + 2$ |
| **Total complexity** | $\mathcal{O}(NF \cdot (N + F)) = \mathcal{O}(N^2F + NF^2)$ |
| **Per-token complexity** | $\mathcal{O}(N + F)$ |

Per-token complexity scales **linearly** with $N + F$ instead of $NF$ — dramatically cheaper for longer videos.

### Scaling Behavior
- If image resolution doubles (both $W$ and $H$ → $2W, 2H$): number of pixels $\times 4$, attention complexity $\times 16$ (quadratic)
- Both joint and divided scale quadratically with image size, but divided grows at a **much slower rate**
- The paper shows joint attention goes out of memory at ~96 frames; divided remains feasible

## Architecture: Divided Space-Time Transformer Block

```
Input: Z^(l-1) (context vectors from previous block)
  │
  ├──► Temporal Attention ──► Layer Norm ──► + Residual
  │
  ├──► Spatial Attention  ──► Layer Norm ──► + Residual
  │
  └──► MLP ──► Layer Norm ──► + Residual
  │
Output: Z^(l)
```

- **Temporal attention**: rearrange tokens so batch & spatial dims are merged; attend across frames
- **Spatial attention**: rearrange tokens so batch & temporal dims are merged; attend across patches
- **MLP**: Linear → GELU → Linear (project to 4× dim then back)
- **Layer normalization** applied after each sub-layer (omitted in paper figure for simplicity)

## Key Results from the Paper

### t-SNE Embedding Visualization
- Vanilla ViT: class clusters are mixed, poorly separated → hard to classify
- TimeSformer (space only): better clustering but still overlapping
- TimeSformer (divided T+S): **clear, well-separated clusters** → hyperplane separation is straightforward

Good cluster separation = embeddings of different action classes are far apart in the 768-dim space, making classification via linear head much easier.

### Classification Accuracy (Kinetics-400)

| Model | Attention Type | Params | Top-1 Accuracy |
|-------|---------------|--------|----------------|
| TimeSformer | Space Only (S) | 85.4M | 76.3% |
| TimeSformer | Joint (ST) | 121.4M | 77.4% |
| **TimeSformer** | **Divided (T+S)** | **121.4M** | **78.0%** |

### Why Divided Beats Joint (despite less context)
Divided attention has **separate learnable parameters** for space and time:
- Temporal: $W_Q^{(t)}, W_K^{(t)}, W_V^{(t)}$
- Spatial: $W_Q^{(s)}, W_K^{(s)}, W_V^{(s)}$

Joint attention uses a single set $W_Q, W_K, W_V$ for everything — it has full context but fewer specialized parameters. The separate parameterization allows the model to learn distinct spatial vs. temporal patterns.

> [!NOTE]
> The comparison isn't perfectly apples-to-apples (85M vs 121M params). The paper may have controlled for this in the appendix.

## Attention Map Visualization
- Bright-colored patches overlaid on frames show where the model focuses
- Model correctly highlights regions where action occurs (e.g., Rubik's cube, moving body parts)
- Similar to explainable AI techniques (e.g., LIME, GradCAM) and ViT/DINOv3 attention visualization

## Implementation Notes

### Model Configuration
- **Patch size**: 16×16
- **Image size**: 224×224
- **Embedding dim**: 768 (must match if loading pretrained ViT weights)
- **Depth**: 12 transformer blocks
- **Heads**: 12 attention heads
- **Frames**: 8 (kept small due to computational cost — even 16 frames is expensive)
- **Pretrained backbone**: ViT-Base/16 from `timm` library

### Position Embeddings
Two separate learnable position embeddings:
1. **Temporal PE**: shape $(1, F+1, D)$ — $+1$ for CLS token
2. **Spatial PE**: shape $(1, N, D)$ — no extra for CLS (only added once, via temporal PE)

### CLS Token
- Single CLS token prepended to the token sequence
- Position embedding for CLS added via the temporal PE (index 0)
- After all transformer blocks, CLS token → layer norm → linear classification head

### Key Dependency: `einops.rearrange`
Used to reshape tensors between formats:
- For temporal attention: merge batch + spatial patches → attend across frames
- For spatial attention: merge batch + temporal frames → attend across patches
- After attention: rearrange back to original $(B, T, N, D)$ shape

### Open Question from Lecture
The instructor flagged a potential issue: in `nn.MultiheadAttention`, the query is passed as the full rearranged input, while key and value come from the same rearranged input (XT for temporal, XS for spatial). The query input might more correctly come from the un-rearranged `x`, but dimensionality constraints of `nn.MultiheadAttention` (expects 3D: batch, seq, dim) make this tricky. The current implementation works and reaches ~80% training accuracy.

### Training Details
- **Optimizer**: AdamW with small learning rate
- **Loss**: Cross-entropy
- **Batch size**: 2 (very small dataset — ~80 training videos total)
- **Results**: ~80% training accuracy at 10 epochs, ~90–92% at 30 epochs (likely overfitting)
- No validation split due to insufficient data

## Practical Considerations
- **Dataset preparation**: Videos → extract frames → use only 8 sampled frames per video
- **Without pretrained ViT**: Can use smaller dimensions (embed_dim, depth, image_size) but lose transfer learning benefits
- **Computational limits**: Even at 16 frames, divided attention is expensive; 32+ frames approaches memory limits
- **Pooling trick**: Spatial dimension can be pooled to a single value, removing need for spatial attention entirely (similar accuracy on this task since bench press vs. deadlift is spatially distinct)

## Assignments
1. Accept **video input directly** instead of pre-extracted frames — handle frame extraction in code
2. Extend to **multiclass classification** (5+ classes from Kinetics-400)
3. Remove pretrained ViT weights and experiment with **smaller model dimensions**
4. Add proper **validation data** (download additional videos from YouTube)
5. Tune hyperparameters (learning rate, epochs, image size, embedding dim)

## References
- **(Bertasius et al., 2021) Is Space-Time Attention All You Need for Video Understanding?**: Original TimeSformer paper from Facebook AI
- **Kinetics-400**: Large-scale video action recognition dataset (available on Kaggle — search "Kinetics 400 dataset 5 percent" for a smaller subset)
- **timm (Torch Image Models)**: Library for loading pretrained ViT weights
