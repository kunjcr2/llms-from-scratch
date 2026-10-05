# Swin Transformers

## Problem and Motivation
The original ViT computes **global self-attention** between all patches. Ideally, for an image with $H \times W$ pixels and patch size $P$, the number of tokens is $N = \frac{HW}{P^2}$.
Attention complexity is $O(N^2)$, which is **quadratic** with respect to image resolution. This makes ViT prohibitively expensive for dense prediction tasks like object detection or semantic segmentation where high-resolution inputs ($800 \times 800$ or more) are standard.

**Swin Transformer** (Hierarchical Vision Transformer using Shifted Windows) aims to bring the **hierarchical** nature of CNNs (pyramidal feature maps) and linear computational complexity to Transformers.

## Core Ideas and Intuition
1.  **Window-based Attention**: Compute attention *only* within local windows (e.g., $7 \times 7$ patches). This reduces complexity to $O(N)$ (linear with image size).
2.  **Shifted Windows**: Standard local windows limit information flow—pixels in window A never talk to pixels in window B. Swin **shifts** the partitioning grid between layers. A pixel on the edge of a window in Layer $l$ becomes the center of a window in Layer $l+1$, enabling cross-window connections.
3.  **Hierarchy**: It starts with small patches ($4 \times 4$ pixels) and merges them deeper in the network (Patch Merging), doubling channel depth and halving spatial resolution, mimicking the downsampling in ResNet/VGG.

## Formalism / Objective
**W-MSA (Window Multi-Head Self Attention)**:
Given a feature map, divide it into non-overlapping windows of size $M \times M$.
Perform standard Self-Attention independently in each window.

**SW-MSA (Shifted Window MSA)**:
In the next layer, the window partitioning is shifted by $(\lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor)$ pixels.
This leads to "cyclic shifting" and efficient batch computation using masking, rather than actually padding the image, to handle the edge cases created by the shift.

The Block structure is always a pair:
$$
\begin{aligned}
& \hat{\mathbf{z}}^l = \text{W-MSA}(\text{LN}(\mathbf{z}^{l-1})) + \mathbf{z}^{l-1} \\
& \mathbf{z}^l = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^l)) + \hat{\mathbf{z}}^l \\
& \hat{\mathbf{z}}^{l+1} = \text{SW-MSA}(\text{LN}(\mathbf{z}^l)) + \mathbf{z}^l \\
& \mathbf{z}^{l+1} = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^{l+1})) + \hat{\mathbf{z}}^{l+1}
\end{aligned}
$$

## Architecture / Design
Swin is divided into 4 stages:
1.  **Stage 1**: "Linear Embedding" projects raw inputs to dimension $C$. Then Swin Transformer blocks. Output: $\frac{H}{4} \times \frac{W}{4}$.
2.  **Stage 2**: **Patch Merging** (concatenates $2 \times 2$ neighboring groups, reduces resolution by 2x, projects channels $C \to 2C$). Then Swin Blocks. Output: $\frac{H}{8} \times \frac{W}{8}$.
3.  **Stage 3**: Patch Merging + Blocks. Output: $\frac{H}{16} \times \frac{W}{16}$.
4.  **Stage 4**: Patch Merging + Blocks. Output: $\frac{H}{32} \times \frac{W}{32}$.

This pyramidal output makes Swin a drop-in backbone replacement for ResNet in setups like Mask R-CNN or U-Net.

## Training Procedure
*   **Relative Position Bias**: Unlike ViT's absolute position embeddings, Swin adds a learnable relative position bias $B$ to the attention logic: $\text{Softmax}(QK^T / \sqrt{d} + B)$. This handles variable input resolutions naturally.
*   **General Backbone**: Swin is often trained on ImageNet-1K or ImageNet-22K and then fine-tuned on COCO (Detection) or ADE20K (Segmentation).

## Practical Considerations and Pitfalls
*   **Window Size**: Default is 7. Increasing this improves performance slightly but raises cost.
*   **Implementation Complexity**: The "cyclic shift" mechanism with masking is tricky to implement from scratch. Most use the official CUDA kernels or optimized Pytorch versions for speed.
*   **Input Resolution**: Because of the relative position bias and windowing, Swin is very flexible with input size changes during inference.

## References
*   **(Liu et al., 2021) Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**: Introduced the architecture and established it as a state-of-the-art general purpose backbone.
