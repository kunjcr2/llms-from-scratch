# World Models From Scratch - Lecture 8

## JEPA Motivation: World Models as Energy Landscapes

Lecture 8 develops the conceptual motivation for the Joint Embedding Predictive Architecture (JEPA) introduced in Lecture 7. It follows the argument in Yann LeCun's *A Path Towards Autonomous Machine Intelligence*: an autonomous system needs an internal world model that can anticipate compatible future states before it acts. JEPA is proposed as a way to learn the representations required for that ability from raw observations, without labels or rewards.

The lecture is not another implementation walkthrough. Instead, it asks what kind of prediction objective gives a world model useful structure, and explains the answer using **energy landscapes**: compatible context--target pairs should have low energy; incompatible pairs should have high energy.

## 1. Why a World Model Needs Prediction

Humans can mentally simulate consequences that they have not actually experienced. A new driver near a cliff can anticipate the result of driving over its edge without executing that action. This ability to evaluate possible outcomes is the motivating role of a world model: infer what can happen from observations of the world, then use those predictions to guide behavior.

The lecture relates this to early human development. Infants acquire progressively richer regularities from observation--faces and objects, object permanence, gravity, stability, and containment--without receiving explicit class labels or scalar reward for each concept. The proposed machine-learning analogue is **self-supervised learning**: learn from the structure already present in observations.

## 2. Desired Properties of a JEPA-Style World Model

The lecture identifies four broad requirements:

1. **Self-supervision:** learn from observations alone, not a manually supplied label or reward for each example.
2. **Abstract prediction:** predict what matters at a representation level, instead of accounting equally for every irrelevant pixel.
3. **Multiple possible outcomes:** represent the set of futures compatible with the current situation, rather than assuming only one deterministic continuation.
4. **A general recipe across modalities:** apply the same basic approach to images, audio, and video, with the model configured for its input rather than designed as a wholly separate algorithm for every sense.

The last point is inspired by the idea that the cortex uses common computational principles across sensory inputs. The lecture calls the modality-specific adaptation mechanism a **configurator**: the overall predictive recipe remains shared, while its configuration accommodates images, video, audio, or another signal type.

## 3. Predict One Representation from Another

At the heart of JEPA is a simple predictive relationship. Given a context representation $x$, predict a target representation $y$:

$$
x \longrightarrow y.
$$

What counts as context and target depends on the modality:

| Modality | Context $x$ | Target $y$ |
| --- | --- | --- |
| Image | Visible crop or unmasked patches | Hidden image region / another view |
| Video | Frame or past clip | A later frame or continuation |
| Audio | Audio segment | Its continuation |

For an image, visible patches may be enough to make a reasonable prediction of a masked region. For a video, however, one frame alone may not specify the next frame: a camera could pan, an object could move, or an agent could act. Prediction therefore requires additional information.

## 4. The Role of Latent Variables

JEPA conditions prediction on a latent variable $z$:

$$
\hat y = g(x, z).
$$

The latent supplies information needed to choose among possible compatible targets but not fully specified by $x$ itself. In I-JEPA, the target-block positions play this role: the visible context does not say which absent region should be predicted until the predictor is told its spatial location.

For temporal or action-conditioned world modeling, $z$ can represent a camera trajectory, an action, or another source of variation. For example, a frame of a room does not reveal that a camera will rotate by 80 degrees. The camera-motion information is necessary to predict the next view. Calling this information a latent emphasizes that it may not be directly recoverable from the visible context alone.

This distinction is important: a good predictive model should not be penalized for failing to infer an outcome that is genuinely underdetermined without the relevant action, motion, or target-location information.

## 5. Compatibility as an Energy Function

The lecture frames prediction as learning an **energy function**

$$
E(x, y, z),
$$

where lower energy means that target $y$ is more compatible with context $x$ and latent $z$. The physical analogy is a ball settling at a low point of potential energy. For a given context, the observed or otherwise valid target should lie at a low-energy location, while unrelated targets should score highly:

$$
E(x, y_{\text{compatible}}, z)
<
E(x, y_{\text{incompatible}}, z).
$$

This viewpoint naturally supports uncertainty. If two distinct future states are genuinely possible, the desired landscape can have two separate low-energy wells. It should not assign low energy uniformly to all candidates, because that would say every future is equally plausible.

## 6. What a Healthy Energy Landscape Looks Like

Fix a context $x$ and consider all candidate targets $y$. A healthy world model has localized basins of low energy around compatible targets and higher energy away from them.

| Landscape | Interpretation |
| --- | --- |
| Tight low-energy wells at valid targets; high energy elsewhere | The model distinguishes compatible outcomes from incompatible ones. |
| Flat, low energy everywhere | The model does not distinguish outcomes; this corresponds to collapse. |
| Broad or diffuse low-energy region | The model has some predictive structure but treats too many near-matches as similarly good. |

In this framing, training lowers energy for observed compatible pairs. The architecture and objective must also ensure that it does not obtain a trivial low-energy solution for every pair. The shape of the learned energy landscape, rather than training loss alone, is what determines whether the representation is useful.

## 7. Three Candidate Objectives

The lecture contrasts three ways of constructing this landscape.

### 7.1 Joint embedding alone: flat landscape

A basic joint-embedding objective makes representations of two augmentations agree. It can be trained with images alone, but its easiest solution maps every image to the same representation. The resulting fingerprints are identical, and every context--target pairing has similarly low energy.

This is **representation collapse**. The loss may be near zero, but the model has learned no meaningful distinction between images. Joint embedding alone is therefore rejected as a sufficient world-model objective.

### 7.2 Masked pixel prediction: useful but diffuse landscape

A masked autoencoder receives a partially hidden image and reconstructs the missing pixels. It can reconstruct object pose, broad scene structure, and background convincingly, so it avoids the constant-representation failure.

Its loss, however, is calculated in pixel space. Grass, sky, texture, and an object's semantic identity all contribute to the objective. This produces a more structured energy map than collapsed joint embedding, but its low-energy region can be diffuse: it gives substantial credit to candidates that match superficial visual details even when they are not the most semantically appropriate target.

### 7.3 JEPA: tight semantic wells

JEPA encodes the visible context, uses a predictor to infer the target representation, and compares that prediction with the target encoder's representation. It predicts in representation space while retaining masked prediction rather than simply aligning two views.

The intended result is a tight low-energy well around the semantically compatible target. The lecture's nearest-neighbor examples illustrate the difference: JEPA tends to place semantically similar examples, such as birds with birds, near one another. A masked autoencoder may instead group examples using texture or visual features that create implausible semantic neighbors, such as a goose and a cauliflower.

## 8. Why Semantic Clustering Matters

Tight energy wells imply useful organization in representation space. Examples with shared meaning form clusters, and unrelated examples remain apart. This organization supports downstream tasks because a simple classifier or nearest-neighbor rule can use structure that has already been learned during unlabeled pretraining.

Pixel reconstruction can be visually impressive without yielding the same organization. Reconstructing a masked image only proves that the model can reproduce visual content; it does not guarantee that the model's nearest neighbors encode object-level meaning. JEPA's goal is deliberately different: form a predictive representation whose geometry reflects compatibility and semantics.

## 9. Connection Back to World Models

I-JEPA applies the principle spatially within a single image. The broader JEPA idea applies it to time: use a present context plus the relevant latent information--such as actions or camera motion--to predict a compatible future representation.

That makes the energy view a useful design lens for world models. A simulator should assign low energy to feasible next states under the supplied action, high energy to incompatible states, and potentially multiple low-energy basins when the future is uncertain. It need not generate every pixel to reason about which futures make sense.

## 10. Key Takeaways

1. JEPA is motivated by learning an internal predictive world model from unlabeled observations.
2. It learns a relationship between context and target representations, conditioned on latent information that resolves ambiguity.
3. Target positions in I-JEPA and camera trajectory or actions in temporal settings are examples of such latent information.
4. The energy function should be low for compatible context--target pairs and high for incompatible ones.
5. A flat low-energy landscape is representation collapse, even if its numerical training loss looks excellent.
6. Pixel reconstruction gives useful structure but can reward irrelevant detail and yield diffuse semantic compatibility.
7. JEPA seeks tight low-energy wells and semantic clusters, which makes its frozen representations effective for downstream tasks.
