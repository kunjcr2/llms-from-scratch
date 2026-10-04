# World Models From Scratch - Lecture 9

## V-JEPA: Learning Video Representations in Latent Space

Lecture 9 extends the image-based JEPA ideas from Lecture 7 to video. **V-JEPA** applies the same masked latent-prediction principle to a sequence of frames: the model receives a partially masked video, encodes the visible context, and predicts representations for the hidden regions. The important new issues are video tokenization, temporal position information, and masks that extend through time.

The lecture evaluates whether the learned features capture meaningful actions and temporal structure on **UCF101**, an action-recognition dataset. The central comparison is between feature-space prediction and pixel-space prediction. Pixel losses can reward a model for reproducing static appearance, whereas latent prediction is better suited to organizing videos by semantic activity and rejecting implausible futures.

## 1. From Image JEPA to Video JEPA

I-JEPA predicts the representation of hidden image regions from visible image context. V-JEPA keeps this architecture but changes the input from one image to a short video clip:

1. A video is divided into spatiotemporal tokens.
2. Some tokens are masked from the student/context encoder.
3. The student encoder represents the visible video context.
4. A predictor uses the context representation and the hidden-token positions.
5. A teacher/target encoder represents the unmasked video.
6. The predictor output is compared with the teacher's target representations.

For a video context $x$, target representation $y$, and mask-position information $z$, the relationship is schematically

$$
\hat y = g_\phi(f_\theta(x), z),
$$

with a latent-space loss between $\hat y$ and the target encoder's representation of the original video. As in I-JEPA, the target encoder is updated by an exponential moving average of the student encoder rather than by the prediction loss directly.

The architecture itself is therefore not the main conceptual change. The difficult part is deciding what a video token represents and how to mask it.

## 2. UCF101 and the Learning Setup

The lecture uses **UCF101**, a dataset of roughly 13,000 short videos covering 101 human-action classes, such as playing basketball, playing guitar, tossing a pizza, horse riding, and typing on a keyboard.

During self-supervised pretraining, the model does not receive the action labels. The labels are reserved for later evaluation with probes. This separates the two questions:

- **Pretraining:** did the model learn useful video representations from raw clips alone?
- **Evaluation:** do those frozen representations encode action and temporal information well enough for a simple downstream method to use?

The lecture uses held-out clips for linear-probe, attentive-probe, and nearest-neighbor evaluations. These tests are more informative than pretraining loss alone because a collapsed or texture-focused representation can still obtain an attractive numerical loss.

## 3. Video Tokenization with Tubelets

Consider a clip with 16 frames, where each frame has spatial dimensions $112 \times 112$ and three color channels. The raw clip contains

$$
16 \times 112 \times 112 \times 3 = 602{,}112
$$

pixel values. Applying attention directly to a very large sequence of raw visual units would be expensive, so the clip is converted into a smaller sequence of learned token vectors.

### 3.1 Group frames in time

The example groups frames two at a time. Sixteen frames therefore produce eight temporal groups:

$$
16 \text{ frames} \;\longrightarrow\; 8 \text{ frame pairs}.
$$

The temporal grouping is a **tubelet**. It is the video analogue of an image patch, but it has spatial extent and temporal extent.

### 3.2 Divide each group spatially

Each two-frame group is divided into a $7 \times 7$ spatial grid, giving 49 spatial locations per tubelet. Across all eight temporal groups, the clip becomes

$$
8 \times 7 \times 7 = 392 \text{ tokens}.
$$

Each token represents a local volume rather than a single image patch. Its temporal dimension is absorbed into the token's feature vector, while its position in the sequence retains both spatial and temporal location.

This is the key generalization from two dimensions to three:

| Image                    | Video                                        |
| ------------------------ | -------------------------------------------- |
| 2D patch                 | 3D tubelet                                   |
| Spatial height and width | Spatial height, width, and temporal duration |
| Patch token              | Tubelet token                                |

The grouping also reduces the number of units on which the transformer operates. Instead of reasoning over hundreds of thousands of pixel values, it reasons over 392 vector tokens.

## 4. Temporal Position Information

Token content alone does not tell a transformer the order in which tubelets occurred. Positional information is therefore needed for the model to distinguish, for example, an action's beginning from its end.

Video positions include both spatial and temporal structure. Swapping the order of frames should change the meaning of the clip, just as shuffling words changes the meaning of a sentence. Positional encoding supplies this ordering information without changing the token width: in the example, the clip still has 392 tokens, each represented by a 384-dimensional vector after embedding and positional information are combined.

Without temporal positions, a model could treat a video as an unordered collection of visual snapshots. That would make it much harder to represent motion and temporal coherence.

## 5. Spatiotemporal Masking

The mask must match the structure of the token. Since a token is a tubelet, masking a token hides a spatial region across a short temporal interval rather than hiding only one 2D patch in one frame. The same spatial masking pattern can be extended through time, producing a 3D mask.

The lecture describes two complementary mask scales:

| Mask             | Purpose                                                                            |
| ---------------- | ---------------------------------------------------------------------------------- |
| Short-range mask | Hides smaller portions of the clip and creates local prediction tasks.             |
| Long-range mask  | Hides large blocks, forcing the model to use broader spatial and temporal context. |

The combination is motivated by ablation results: video representation learning benefits from both local gaps and larger missing regions. For a $7 \times 7$ spatial tubelet grid, the long-range mask can hide most of the cells, while the short-range mask hides a smaller but still substantial subset.

Masking is not merely a data-augmentation detail. It defines the prediction problem. The context encoder must infer what is compatible with the visible video, and the predictor must know the positions of the hidden tubelets so that it predicts the correct target representations.

## 6. V-JEPA Architecture and Anti-Collapse Guard

For an original video $v$ and masked video $\tilde v$:

- The **student/context encoder** $f_\theta$ processes the visible tubelets.
- The **teacher/target encoder** $f_{\bar\theta}$ processes the original video.
- The **predictor** $g_\phi$ receives the student features and hidden-tube positions.
- The target features are treated as stop-gradient targets.

A simplified objective is

$$
\hat y = g_\phi\big(f_\theta(\tilde v), z\big),
\qquad
y = \operatorname{sg}\!\left(f_{\bar\theta}(v)\right),
$$

$$
\mathcal{L}_{\text{V-JEPA}} = d(\hat y, y),
$$

where $z$ identifies the hidden tube positions and $d$ is a latent-space distance.

The teacher is updated using an EMA rule:

$$
\bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta.
$$

The stop-gradient and EMA together form the collapse guard. If the teacher is copied directly from the student at every step, the representation can collapse: all clips acquire nearly identical features, the loss becomes deceptively small, and the model no longer distinguishes actions or temporal content. A slowly changing teacher provides a stable target that prevents this immediate feedback loop.

## 7. Comparing Three Training Variants

The lecture compares three versions on the video data:

| Variant                                    | Prediction space | Teacher update                    | Expected behavior                                     |
| ------------------------------------------ | ---------------- | --------------------------------- | ----------------------------------------------------- |
| V-JEPA                                     | Latent features  | EMA from student                  | Learns noncollapsed semantic video features.          |
| V-JEPA without the stop-gradient/EMA guard | Latent features  | Direct student-to-teacher copying | Representation collapse.                              |
| Pixel twin / masked pixel prediction       | Pixels           | Not applicable in the same way    | Can learn appearance but may miss temporal semantics. |

The important diagnostic is not just whether the loss decreases. The feature standard deviation across examples must also remain healthy. A decreasing loss accompanied by shrinking feature spread indicates that the model is mapping different clips to increasingly similar vectors. This is a sign of collapse, even if the loss curve looks better than the guarded model's curve.

## 8. Evaluating the Learned Video Features

After pretraining, the encoder is used as a frozen feature extractor. A clip from the test set is mapped to a feature vector, and simple probes assess what information survived in that representation.

### Linear and attentive probes

A **linear probe** trains a linear classifier on top of frozen features. An **attentive probe** adds an attention layer, allowing a more expressive, nonlinear readout. Good probe performance indicates that action information is accessible to a downstream classifier, but it does not fully describe the geometry of the representation space.

### k-nearest-neighbor evaluation

A k-nearest-neighbor probe classifies a clip using nearby feature vectors. It directly tests whether semantically similar videos cluster together. For example, rafting clips should be close to other rafting clips, and handstand clips should be close to other handstand clips.

The lecture reports that V-JEPA is especially strong under this neighborhood-based test. Pixel-space models may match texture, color, or broad visual appearance while failing to represent the activity or the relevant objects. Latent prediction better organizes clips by the meaning of the action.

## 9. Does the Representation Encode Time?

The lecture probes temporal information by modifying clips and asking whether the encoder can distinguish the altered versions:

1. **Intact versus shuffled:** the frame order is randomly permuted.
2. **Forward versus reversed:** the clip is played backward.

The model distinguishes intact from shuffled clips with roughly 72% binary accuracy in the reported experiment, suggesting that it has learned temporal coherence. It performs much less reliably on forward versus reversed clips.

This result is an important limitation rather than a contradiction. A short 2.1-second clip of waving, drumming, or juggling can be approximately time-symmetric. The dataset may contain few irreversible events, such as pouring, shattering, or splashing, that make the arrow of time unmistakable. The model may therefore understand that frames belong together in a coherent sequence without learning which direction time flows.

The experiment also shows why temporal evaluation should be designed around the property being tested. Shuffle detection tests temporal order and coherence; reversal detection tests directional asymmetry. They are related but not equivalent.

## 10. Future Prediction as an Energy Test

The lecture evaluates whether the model can rank candidate futures. Given a context clip, it presents several possible continuations:

- the true future,
- a reversed future,
- a frozen frame,
- an unrelated or incorrect future.

The model assigns an energy to each candidate. A good predictive representation should assign low energy to the true continuation and higher energy to incompatible candidates:

$$
E(\text{context}, \text{true future})
<
E(\text{context}, \text{incorrect future}).
$$

V-JEPA ranks the true future as low energy and treats the alternatives as less compatible. The pixel twin instead often assigns the lowest energy to a frozen frame. This reveals a weakness of pixel-space comparison: a static frame can remain visually similar to the current frame and therefore incur a small pixel error, even though it is not a plausible continuation of a moving event.

In feature space, the frozen frame is recognized as temporally anomalous. This is more useful for a world model because planning requires a representation of how states evolve, not merely a preference for predictions that change as little as possible.

## 11. Relation to World Models

V-JEPA is still a representation learner rather than a complete action-conditioned world model. It does not yet specify an action and generate the exact future observation. Its contribution is to learn a latent space in which video context, motion, and plausible continuation can be compared meaningfully.

This is a suitable foundation for later world-model tasks:

- represent the current video state compactly,
- predict compatible future representations,
- score candidate futures using energy,
- support retrieval, classification, planning, or control without reconstructing every pixel.

The lecture's freeze-frame experiment captures the distinction well. A useful world model must learn temporal change and compatibility, whereas a pixel objective can treat “nothing changes” as a safe prediction because it minimizes visual difference.

## 12. Key Takeaways

1. V-JEPA extends I-JEPA from images to videos by predicting hidden tubelet representations.
2. A two-frame temporal grouping plus a $7 \times 7$ spatial grid turns a 16-frame clip into $392$ spatiotemporal tokens.
3. Tubelets and masks must include the temporal dimension; positional information must preserve frame order.
4. Combining short-range and long-range masks exposes the model to both local and broad spatiotemporal prediction tasks.
5. Stop-gradient plus an EMA teacher is essential for preventing representation collapse.
6. Feature standard deviation and downstream probes are necessary diagnostics; a low training loss alone is not enough.
7. V-JEPA organizes semantically similar actions more effectively than pixel-space prediction, especially under k-nearest-neighbor evaluation.
8. The learned features capture temporal coherence better than temporal direction in the short clips tested.
9. In future-ranking experiments, V-JEPA prefers the true continuation, while pixel prediction can incorrectly prefer a frozen frame.
10. Latent video prediction provides a representation foundation for future world models that need to plan over plausible temporal states.
