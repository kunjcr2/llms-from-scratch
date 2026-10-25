# World Models From Scratch - Lecture 11

## DIAMOND: Diffusion as a World Model

Lecture 11 introduces **DIAMOND**, an action-conditioned world model that predicts future observations directly in continuous pixel space using diffusion. It is presented as an alternative to Iris, which predicts discrete visual tokens, and DINO-WM, which predicts continuous latent features.

## 1. From Iris to DIAMOND

The key differences are:

| Model | Prediction space | Main prediction method |
| --- | --- | --- |
| Iris | Discrete image tokens | Autoregressive next-token prediction |
| DINO-WM | Continuous latent features | Transformer feature prediction |
| DIAMOND | Continuous pixels | Diffusion denoising |

DIAMOND does not use a tokenizer or a discrete codebook. Instead, it starts with noise and repeatedly denoises it into the next image.

## 2. Conditional Diffusion

A diffusion noise predictor receives:

- Several recent observation frames.
- The action associated with the transition.
- A diffusion time step, usually written as $\tau$.
- A noisy version of the next observation.

It predicts the noise added to the target image. During inference, the model starts from random noise and repeatedly applies the denoiser until it produces a clean next frame.

The observation history is stacked along the channel dimension. For example, four RGB frames create twelve conditioning channels. Action conditioning is added through adaptive normalization in the U-Net, similar to feature-wise conditioning.

## 3. Training and Environment Rollouts

The denoiser is trained on observation-action-next-observation data. DIAMOND also learns reward and termination predictions because a world model used for reinforcement learning must predict more than visual appearance.

After training, the denoiser is used autoregressively:

1. Start with a history of real frames.
2. Provide an action and generate the next frame by denoising.
3. Feed the generated frame back into the history.
4. Repeat to produce a dream rollout.

An actor-critic or REINFORCE-style learner can then use rewards from imagined episodes to improve its actions.

## 4. DDPM and EDM

The lecture explains the usual diffusion process: a forward process adds noise, while a learned reverse process removes it. DIAMOND uses **EDM**-style diffusion in the paper, although the conceptual process is the same as DDPM. EDM can reach a clean image with fewer denoising steps and often converges faster.

The model uses an observation horizon of three frames in the discussed setup. The denoiser is typically implemented with a U-Net, which is well suited to predicting image-shaped noise.

## 5. Strengths and Limitation

DIAMOND achieved strong results on Atari 100K because it preserves information that can be lost by aggressive tokenization. Its predictions are continuous and visually detailed, and the model can be trained from scratch without a large pretrained encoder.

The cost is computational: generating one frame requires multiple denoising steps. DIAMOND also needs action-labeled data, which makes it difficult to train directly from large collections of unlabeled internet videos. This motivates later video world models that learn general physical structure from video first and add action conditioning afterward.
