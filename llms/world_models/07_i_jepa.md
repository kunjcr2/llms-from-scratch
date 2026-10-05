# World Models From Scratch - Lecture 7

## I-JEPA: Predicting Image Representations Rather Than Pixels

Lecture 7 introduces the **Joint Embedding Predictive Architecture (JEPA)**, specifically **I-JEPA** for images. Earlier lectures built simulators that predict the next observation after an action. JEPA is a representation-learning architecture that supplies a different kind of visual understanding: it learns to predict the representation of missing image content from its visible context, without reconstructing pixels or using labels.

The central premise is that a useful model should retain semantic information--for example, that multiple views depict the same dog--while ignoring incidental pixel-level details such as the background or lighting. This is relevant to world models because a simulator ultimately needs representations that capture what changes and what matters, rather than spending all of its capacity reproducing every visual detail.

## 1. The Two Earlier Self-Supervised Families

JEPA combines useful aspects of two broad self-supervised-learning families.

### Joint-embedding learning

A joint-embedding model sends two augmentations of the same image through encoders and makes their latent vectors agree. If $x$ and $x'$ are two views of an image, it minimizes a representation-space distance such as

$$
\mathcal{L} = \lVert f(x) - f(x') \rVert.
$$

This is attractive because the comparison is made in latent space, where the model can focus on meaning rather than every pixel. However, the objective has a degenerate solution: the encoder can return the same constant vector for every image. The loss then becomes zero, even though the representation distinguishes neither dogs from birds nor any other images. This failure mode is **representation collapse**.

### Generative / masked-pixel learning

A masked autoencoder hides parts of an image, encodes the remainder, and decodes the missing pixels. This avoids the simple constant-output solution because the output must reproduce different targets for different inputs. But it makes every pixel detail equally important to the objective. A model may devote substantial capacity to texture, grass, sky, or other details that are less useful than high-level semantics.

JEPA keeps masked prediction from the generative family, but moves the prediction target from pixel space into representation space.

## 2. The JEPA Objective

Start with a target image $y$. A masking procedure removes one or more blocks, leaving a visible **context** $x$. JEPA asks:

> Given the visible context and the locations of the hidden blocks, can the model predict the target encoder's representations for those blocks?

The three trainable components are:

1. **Context encoder** $f_\theta$: encodes the visible image tokens.
2. **Predictor** $g_\phi$: uses the context representation and a description of the target-block positions to predict representations for the hidden blocks.
3. **Target encoder** $f_{\bar\theta}$: encodes the full image and provides the representation targets.

For a hidden block $B$, a schematic objective is

$$
\hat z_B = g_\phi\big(f_\theta(x), B\big),
\qquad
z_B = \operatorname{sg}\!\left(f_{\bar\theta}(y)_B\right),
$$

$$
\mathcal{L}_{\text{JEPA}} = \sum_B \lVert \hat z_B - z_B \rVert.
$$

Here $\operatorname{sg}$ means **stop-gradient**: the loss does not backpropagate through the target encoder. The lecture uses L1 loss in its implementation discussion, while noting that descriptions of JEPA variants may instead use an L2 distance.

Unlike an autoencoder, the predictor never needs to render the absent pixels. It must predict a representation that is sufficient to match the hidden region's learned meaning.

## 3. I-JEPA Uses Vision Transformers

In I-JEPA, the encoders are Vision Transformers (ViTs). An image is split into patches, each patch becomes a token embedding, and the ViT produces contextual token representations.

The target encoder sees tokens from the full image. The context encoder sees only the tokens outside the chosen target blocks. The predictor additionally receives the target-block locations, so it knows *which* absent region it is expected to predict. In other words, masking conceals image content, not the spatial identity of the prediction request.

The target can contain multiple blocks. The lecture's example samples four target blocks, each covering roughly 15--20% of the image, with aspect ratios from 0.75 to 1.5; blocks may overlap. All target-patch tokens are removed from the context encoder's input.

## 4. Why the Teacher Needs a Guard

Masked latent prediction makes collapse harder than simply forcing two augmented views to have identical embeddings, but it does not make collapse impossible. I-JEPA stabilizes learning with a slowly changing teacher:

- Gradients update the context encoder and predictor, the **student** side.
- The target encoder receives no loss gradient.
- Its parameters follow an exponential moving average (EMA) of the context encoder's parameters:

$$
\bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta.
$$

The target encoder therefore provides a delayed target rather than immediately changing in whatever direction most quickly reduces the current loss. The combination of stop-gradient and EMA is called the lecture's **guard**. It is an empirically effective anti-collapse strategy; the lecture notes that a complete theoretical explanation of why it works remains an open question.

This makes a decreasing pretraining loss insufficient evidence of success. A collapsed encoder can achieve an excellent-looking, near-zero loss by producing nearly constant fingerprints for all images.

## 5. Experimental Comparisons and Collapse Diagnostics

The lecture compares five self-supervised variants on the unlabeled STL-10 image dataset using a ViT encoder:

| Variant | Prediction target | Guard | Observed behavior |
| --- | --- | --- | --- |
| Naive joint embedding | Latent representation of an augmentation | No | Collapses to a constant representation |
| JEPA without guard | Hidden latent representation | No | Also collapses |
| Masked autoencoder (MAE) | Pixels | Not applicable | Learns nonconstant features and reconstructs images |
| MAE with JEPA block masks | Pixels | Not applicable | Also learns nonconstant features |
| I-JEPA | Hidden latent representation | Stop-gradient + EMA | Learns distinct, stable representations |

One practical collapse diagnostic is variation across a batch. For each embedding dimension, calculate the standard deviation over examples. Near-zero variation across all dimensions indicates that every image has been mapped to essentially the same vector. Healthy representations display meaningful variation across examples and dimensions.

The loss curve for guarded I-JEPA need not monotonically fall to zero: as the EMA target encoder becomes a stronger teacher, the loss can rise from its initial random-teacher behavior and settle near a baseline. The representation diagnostics matter more than a superficially attractive loss value.

## 6. Reconstruction Is Not the Same as Semantics

The MAE experiments reconstruct heavily masked images plausibly: pose and broad background can return even when about 75% of the input is masked. This shows that pixel-generative learning is useful, not that it optimizes the same representation as JEPA.

Its reconstruction target rewards accurate local appearance. Consequently, an MAE can learn texture and visual detail without arranging images with similar semantic meaning into especially coherent neighborhoods. JEPA deliberately predicts a representation target, encouraging nearby points in latent space to share meaning rather than merely appearance.

## 7. Evaluating Frozen Representations

Self-supervised pretraining does not use class labels, so training loss alone cannot tell whether a representation will support a downstream task. The lecture freezes the learned encoder and evaluates it with two probes:

1. **Linear probe:** train only a linear classifier on top of frozen embeddings. Strong accuracy indicates that the required class separation is already present in the representation.
2. **$k$-nearest-neighbor (k-NN) probe:** classify using nearby embeddings. Strong performance indicates that semantically similar examples form local clusters.

On the small-data comparison, I-JEPA and generative approaches can both do well under a linear probe, but I-JEPA is stronger under k-NN. This distinction is meaningful: a linear classifier can exploit global separability, whereas k-NN directly tests whether nearby representations themselves share labels and semantic content.

On the larger ImageNet-100K experiment, the lecture reports that I-JEPA gives much stronger label efficiency. With only 1% of labels for a linear probe, it reports about 77% accuracy for I-JEPA versus about 58.4% for the masked-autoencoder comparison. JEPA's performance remains relatively consistent as the fraction of probe labels increases, consistent with well-formed semantic clusters. These figures are experimental results from the lecture setup, not universal performance guarantees.

Two-dimensional visualization of embeddings and nearest-neighbor examples support the same interpretation: JEPA neighbors tend to be semantically related, while MAE neighbors can be tied more strongly by texture or appearance.

## 8. Relation to World Models

I-JEPA is not yet an action-conditioned simulator. It learns from images alone and predicts the representation of missing spatial content, rather than predicting a future observation from state and action. Its role in this lecture is foundational: it demonstrates how a model can learn useful visual features self-supervisedly by predicting in representation space.

The design principle can extend to temporal prediction. A future JEPA-style world model can use current context and actions to predict representations of future observations, thereby modeling dynamics while avoiding the requirement to generate every pixel. The next lecture moves toward that video-oriented setting.

## 9. Key Takeaways

1. JEPA predicts hidden content in latent representation space, not in pixel space.
2. I-JEPA uses a context ViT, a predictor conditioned on target-block positions, and a full-image target ViT.
3. Naive joint embedding can minimize loss by mapping every image to the same vector; low loss is therefore not proof of learning.
4. Stop-gradient plus an EMA target encoder forms an essential anti-collapse guard in I-JEPA.
5. Masked pixel reconstruction can produce credible images but may prioritize visual details over semantic organization.
6. Linear and k-NN probes of frozen embeddings test different downstream properties; k-NN is especially informative about local semantic clustering.
7. I-JEPA provides a self-supervised visual-representation foundation that can later be adapted to temporally predictive world models.
