# Detection Transformer (DETR)

## Problem and Motivation
Traditional object detection models like YOLO, Fast R-CNN, Faster R-CNN rely on **hand-engineered components**:
1.  **Anchor Boxes**: Predefined boxes at grid points with various aspect ratios that get refined during training
2.  **Non-Maximum Suppression (NMS)**: Post-processing to eliminate duplicate bounding box predictions

These approaches have limitations:
- Anchor box design (number, aspect ratios) is manually tuned
- NMS thresholds (confidence, IOU) are hand-engineered
- Multiple predicted boxes can correspond to the same object

**DETR** (DEtection TRansformer) eliminates both anchors and NMS by treating object detection as a **direct set prediction problem**.

## Core Ideas and Intuition

### Direct Set Prediction
Instead of predicting many boxes and filtering duplicates, DETR predicts a fixed set of $N$ objects (e.g., 100) in parallel:
- If fewer objects exist in the image, remaining predictions become "no object" ($\varnothing$)
- Each prediction uniquely matches to one ground truth via **bipartite matching**
- No duplicate predictions → No need for NMS

### Object Queries
DETR introduces **object queries** — learnable vectors that act as "slots" for potential objects:
- Initialized as zero vectors with dimensionality matching the embedding space
- Each query is responsible for predicting exactly one object (or background)
- Number of queries = Maximum objects detectable

### Hungarian Matching
To assign predictions to ground truths, DETR uses the **Hungarian algorithm**:
- Finds optimal one-to-one matching between predicted and ground truth boxes
- Minimizes total matching cost across all pairs
- If $N$ predictions and $M$ ground truths ($M < N$), pad ground truth with $(N-M)$ "no object" labels

## Architecture / Design

DETR has four main components:

### 1. CNN Backbone
- Uses pretrained ResNet-50 or ResNet-101
- Extracts feature map of shape $(H, W, C)$ from input image
- Lower spatial resolution, higher channel depth than input

### 2. Transformer Encoder
- Feature map is flattened into a sequence of tokens: $(H \times W, D)$
- Each token = one spatial position projected to embedding dimension $D$
- Standard transformer encoder blocks with multi-head self-attention
- **Key difference**: Sinusoidal position embeddings are added at every encoder layer (not just once)
- Position embedding added to Query and Key, not Value

### 3. Transformer Decoder
The decoder has **two types of attention**:

**Self-Attention (among object queries)**:
- Object queries attend to each other
- Uses **learnable position embeddings** (not sinusoidal)
- Allows queries to reason about relationships between predicted objects

**Cross-Attention (queries ↔ encoder output)**:
- Query ($Q$): From object query context vectors
- Key ($K$), Value ($V$): From encoder output (image features)
- Position embeddings:
  - Query gets **learnable** PE (no inherent spatial meaning)
  - Key gets **sinusoidal** PE (from 2D image grid)

### 4. Prediction Heads (FFN)
Two parallel MLPs process each transformed object query:
1.  **Class MLP**: Predicts class probabilities (including "no object")
2.  **Box MLP**: Predicts normalized bounding box coordinates $(x, y, w, h)$

```
                    ┌──────────────┐
   Object Query ──► │   Class MLP  │ ──► Class + Confidence
                    └──────────────┘
                    ┌──────────────┐
   Object Query ──► │   Box MLP    │ ──► (x, y, w, h)
                    └──────────────┘
```

## Loss Function

DETR uses a **set-based loss** with Hungarian matching:

### Step 1: Hungarian Matching
Find optimal bipartite matching $\hat{\sigma}$ that minimizes:
$$
\hat{\sigma} = \arg\min_{\sigma} \sum_{i}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})
$$

### Step 2: Compute Loss on Matched Pairs
Total loss combines three components:

**Classification Loss** ($\mathcal{L}_{cls}$):
- Cross-entropy loss for class prediction
- Penalizes wrong class assignments

**Localization Loss** ($\mathcal{L}_{loc}$):
1.  **L1 Loss**: $|x - \hat{x}| + |y - \hat{y}| + |w - \hat{w}| + |h - \hat{h}|$
2.  **Generalized IoU Loss (GIoU)**:

$$
\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
$$

Where $C$ is the smallest enclosing box of both prediction and ground truth.

**Why GIoU?** Standard IoU = 0 for non-overlapping boxes regardless of distance. GIoU captures "how far apart" boxes are.

**Total Loss**:
$$
\mathcal{L} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{L1} \mathcal{L}_{L1} + \lambda_{GIoU} \mathcal{L}_{GIoU}
$$

### Training Detail
During training, loss is computed using outputs from **all decoder layers** (not just the final one), then summed. During inference, only the final layer's output is used.

## Hungarian Algorithm Summary

For matching $N$ predictions to $N$ ground truths (padded with $\varnothing$):

1. **Build Cost Matrix**: Compute $\mathcal{L}_{\text{match}}$ for all $(pred_i, gt_j)$ pairs
2. **Row Reduction**: Subtract row minimum from each row
3. **Column Reduction**: Subtract column minimum from each column
4. **Cover Zeros**: Draw minimum lines to cover all zeros
5. **Iterate**: If lines < N, adjust matrix and repeat
6. **Extract Matching**: Assign predictions where zeros appear (one per row/column)

This provides the optimal one-to-one assignment minimizing total cost.

## Practical Considerations and Pitfalls

*   **Number of Object Queries**: Typically 100. Must exceed max objects expected per image.
*   **Training Time**: DETR requires significantly longer training (300+ epochs on COCO) compared to CNN-based detectors.
*   **Small Object Detection**: Original DETR struggles with small objects. Variants like Deformable DETR address this.
*   **No GPU Required for Inference**: Lightweight enough for CPU inference on images.
*   **Inference Code**: Can be implemented in ~50 lines using pretrained models.

## Quick Inference Example

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Process image
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Inference
outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# Display results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.5:
        print(f"{model.config.id2label[label.item()]}: {score:.2f}")
```

## Comparison: DETR vs Traditional Detectors

| Aspect | Traditional (YOLO, RCNN) | DETR |
|--------|--------------------------|------|
| Anchors | Hand-designed anchor boxes | No anchors |
| NMS | Required post-processing | Not needed |
| Predictions | Many → filter duplicates | Fixed set, no duplicates |
| Matching | Heuristic (IoU-based) | Optimal (Hungarian) |
| Architecture | Mostly CNN | CNN + Transformer |
| Training | Fast convergence | Slow (300+ epochs) |

## References
*   **(Carion et al., 2020) End-to-End Object Detection with Transformers**: Original DETR paper from Facebook AI, 21k+ citations.
*   **Hungarian Algorithm**: Named after Hungarian mathematicians who laid groundwork for the assignment problem.
