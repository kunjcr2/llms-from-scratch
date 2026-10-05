# World Models From Scratch - Lecture 4

## Recurrent State-Space Models (RSSMs)

Lecture 4 replaces the first Mini Pong world model with a more robust model for a real robot trajectory. The central idea is a **recurrent state-space model (RSSM)**: a latent state with two complementary parts:

- a deterministic recurrent memory that carries information forward through time; and
- a stochastic latent that represents uncertainty and alternative plausible futures.

Together they form the model's **belief state**. This architecture underlies the PlaNet paper and the Dreamer family of world models.

## 1. Motivation: Why the Earlier Model Drifts

The Lecture 3 model predicts one next latent/frame and repeatedly feeds its own prediction back as input. A small early mistake changes the next input, which causes a larger later mistake. This closed-loop failure is called **error compounding**.

It appeared in Mini Pong as a ball that became blurred or stretched during long imagined rollouts. The same problem is much more consequential for robotics: a camera may be turned off after a few real frames, leaving the world model to imagine the remainder of a robot pick-and-place trajectory from actions alone.

The lecture uses about 50 SO-101 robot episodes (roughly 11,000 frames). Each observation is a $64 \times 64 \times 3$ image, and each action has six robot-control values:

- shoulder pan;
- shoulder lift;
- elbow flex;
- wrist flex;
- wrist roll; and
- gripper.

The supervised modeling objective is still:

$$
(o_t, a_t) \longmapsto \hat{o}_{t+1}.
$$

For these demonstrations, actions come from the recorded trajectory. In a deployed agent, a policy or planner would have to choose them.

## 2. Observations Are Not States

An observation is not necessarily enough to determine what will happen next. For example, when the robot gripper occludes a cube, the cube is absent from the image but has not disappeared from the scene. Its location and whether it is grasped must be inferred from earlier observations and actions.

The true state may involve information such as:

- robot joint configuration and motion;
- object position and contact state;
- whether a grasp is secure or slipping; and
- hidden scene details during occlusion.

A world model should therefore maintain a representation that summarizes both history and uncertainty about unobserved variables. This representation is the **belief state**.

## 3. Memory Versus Belief

A recurrent memory solves part of the partial-observability problem. A single image of a moving object does not reveal its velocity, but a sequence of images can. The memory summarizes such history.

Belief is broader than memory: it represents what the model thinks is currently true, including uncertainty. If the gripper clearly does not cover the cube, the model may be confident about the scene. If it covers the cube, several physical states may remain plausible.

Thus a good next-state model should produce a distribution, not only one point estimate. This is analogous to an LLM producing a probability distribution over next tokens before sampling or selecting a token.

### Why a Single Deterministic Prediction Can Blur

When two different futures are plausible, a deterministic pixel/latent regression model often learns their average. If a car could turn left or right, an average prediction may incorrectly proceed through the middle. Averaging distinct visual futures also creates blurry images.

Stochastic latents let the model represent different modes of the future instead of averaging them together.

## 4. Two Incomplete Designs

The RSSM is motivated by the complementary failures of two simpler designs.

### Design A: Deterministic Recurrent Model

The first design encodes an observation and updates a GRU-like memory deterministically:

$$
h_t = f(h_{t-1}, z_{t-1}, a_{t-1}).
$$

For the robot experiment, a linear probe of this memory can recover the robot's joint angles, showing that recurrent memory learns meaningful structure. However, given exactly the same history, this model must always predict exactly the same future. It cannot explicitly represent ambiguity, and long rollouts become blurry and drift.

### Design B: Stochastic State Without Persistent Memory

A purely stochastic design samples a state from a predicted distribution. It can produce multiple possible futures from the same initial situation, which captures uncertainty.

But sampling a fresh state at every step without carrying a stable history loses temporal continuity. Its predictions become effectively random because it forgets facts established in the past.

The two failure modes are:

- deterministic recurrence preserves memory but collapses uncertain alternatives; and
- pure stochasticity represents alternatives but leaks memory.

## 5. RSSM Belief State

An RSSM combines the two designs in one latent state:

$$
b_t = [h_t, s_t].
$$

Here:

- $h_t$ is the deterministic recurrent state (the memory); and
- $s_t$ is a stochastic latent sampled from a distribution.

In the lecture configuration, $h_t \in \mathbb{R}^{256}$ and $s_t \in \mathbb{R}^{32}$, so the concatenated belief state has 288 values.

The deterministic part retains what the model has committed to in the past. The stochastic part carries uncertainty—such as alternative outcomes of a grasp. Predictions, including the decoded image, are made from **both** parts rather than from either component alone.

## 6. RSSM Transition Model

At each step, the previous belief state and action update the deterministic memory:

$$
h_t = f_\theta(h_{t-1}, s_{t-1}, a_{t-1}).
$$

The transition network then predicts a **prior** distribution for the new stochastic state:

$$
p_\theta(s_t \mid h_t).
$$

Sampling from this distribution gives a possible next latent state. Crucially, the sample becomes part of the input at the following recurrent update, so an imagined trajectory remains internally consistent instead of independently re-rolling uncertainty each step.

The latent state is decoded into an observation prediction:

$$
\hat{o}_t = D_\theta(h_t, s_t).
$$

In a rollout without new camera images, this prior is all the model has. It generates the next belief state and observation using its current belief and the supplied action.

## 7. Prior and Posterior During Training

Training has access to the real observation $o_t$, so it can form a more informed **posterior** distribution:

$$
q_\phi(s_t \mid h_t, o_t).
$$

The posterior can use the camera image; the prior cannot. It is useful to view the posterior as a learner with the answer sheet and the prior as the model that must later predict without it.

The posterior supplies the latent used to reconstruct the real frame during training. The model also trains the prior to match it, so the prior can be used successfully at inference time.

This distinction does not mean an observation encoder is unnecessary: an encoder is normally used to extract observation features for the posterior. The key architectural change from Lecture 3 is that the decoder reconstructs from the full belief $[h_t,s_t]$, rather than predicting solely through a separate next-frame code.

## 8. Training Objective

RSSM training combines two losses.

### Observation Reconstruction Loss

The decoder must reconstruct the observed frame from the posterior belief state:

$$
\mathcal{L}_{\text{obs}} = -\log p_\theta(o_t \mid h_t, s_t).
$$

For a simple image decoder, this can be implemented with a pixel reconstruction loss. It forces the latent belief to retain the information needed to explain the visual world.

### Prior–Posterior Matching Loss

The prior must approximate the posterior distribution that was informed by the real image. This is commonly a KL-divergence term:

$$
\mathcal{L}_{\text{KL}} =
D_{\mathrm{KL}}\!\left(
q_\phi(s_t \mid h_t,o_t)\;\|\;p_\theta(s_t \mid h_t)
\right).
$$

The total objective is typically a weighted sum over time:

$$
\mathcal{L} = \sum_t
\left(\mathcal{L}_{\text{obs}} + \beta\mathcal{L}_{\text{KL}}\right).
$$

The reconstruction term teaches the state to represent the scene. The KL term makes its action-conditioned, image-free prediction usable for imagination.

## 9. Training and Imagination

### Training

For each recorded transition:

1. Update $h_t$ from $h_{t-1}$, $s_{t-1}$, and $a_{t-1}$.
2. Compute the prior $p(s_t\mid h_t)$.
3. Use $o_t$ to compute the posterior $q(s_t\mid h_t,o_t)$ and sample $s_t$.
4. Decode $[h_t,s_t]$ to reconstruct $o_t$.
5. Optimize reconstruction and prior–posterior matching losses.

### Imagination / Rollout

After the initial observed frames, no new image is needed:

1. Supply an action.
2. Update deterministic memory.
3. Sample $s_t$ from the prior.
4. Decode the resulting belief state if a visual prediction is needed.
5. Repeat.

This lets a world model generate a robot video trajectory after the camera is effectively switched off.

## 10. Why RSSMs Improve Rollouts

The deterministic path acts as a conveyor belt for persistent facts: prior movement, object identity, and inferred physical state survive through time. The stochastic path acts like a controlled dice roll: it represents uncertainty without discarding the trajectory history.

This combination substantially improves the robot rollout compared with either component alone. It is the architectural basis on which the Dreamer lineage performs planning and policy learning in imagined trajectories.

## 11. Key Takeaways

1. Recursive use of predicted observations causes errors to compound during long world-model rollouts.
2. A camera observation is not the full state, especially under occlusion; the model needs a belief about hidden variables.
3. Deterministic recurrent memory preserves history but cannot naturally represent multiple plausible futures.
4. A stochastic state can represent uncertainty but must be coupled to recurrent memory to avoid forgetting history.
5. An RSSM represents state as $[h_t,s_t]$: deterministic memory plus stochastic latent.
6. The transition model predicts an image-free prior; training also uses an observation-informed posterior.
7. Reconstruction teaches the belief state to explain images, while KL matching teaches the prior to imitate the posterior.
8. At inference, the model rolls forward with actions and its prior alone, enabling imagination without further camera input.
