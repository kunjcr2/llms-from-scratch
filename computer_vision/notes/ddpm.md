# DDPM (Denoising Diffusion Probabilistic Models)

## Overview
**DDPM** (by Ho et al., UC Berkeley, 2020) is a generative model that produces new images by learning to **reverse a gradual noising process**. The core idea is deceptively simple: systematically destroy an image by adding Gaussian noise over many small steps (forward diffusion), then train a neural network to undo that destruction step-by-step (denoising / sampling). The paper has amassed over 34,000 citations and is considered the cornerstone of modern diffusion-based generative AI — not only for images but increasingly for language as well.

### Relationship to VAE
In a Variational Autoencoder, an **encoder** (a trained neural network) maps an image into a compressed latent space, and a **decoder** reconstructs from it. In DDPM:
*   The "encoding" direction (image → noise) is the **forward diffusion process**, which is entirely deterministic / formula-based and requires **no learned encoder**.
*   The "decoding" direction (noise → image) is the **DDPM sampling process**, which uses a trained **U-Net** to predict and remove noise iteratively.

The "prompt" in a VAE is a latent vector sampled from the learned Gaussian. In DDPM, the "prompt" is pure Gaussian noise — a random tensor where every pixel is independently sampled from $\mathcal{N}(0, 1)$.

---

## Part 1: Forward Diffusion (Image → Noise)

### Single-Step Formulation
Given a clean image $x_0$, forward diffusion constructs a sequence of progressively noisier images $x_1, x_2, \dots, x_T$. A single step is:

$$x_t = \sqrt{\alpha_t}\; x_{t-1} + \sqrt{1 - \alpha_t}\; \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

where:
*   $\alpha_t$ is the scaling factor for the previous image (close to 1 initially, shrinking over time).
*   $\beta_t = 1 - \alpha_t$ is the complementary noise weight.
*   The constraint $\alpha_t + \beta_t = 1$ (equivalently $(\sqrt{\alpha_t})^2 + (\sqrt{\beta_t})^2 = 1$) ensures the variance stays controlled.
*   $\epsilon$ is fresh Gaussian noise sampled independently at each step.

Intuitively: at each step, you slightly fade the image (multiply by $\sqrt{\alpha_t} < 1$) and add a small amount of random noise (scaled by $\sqrt{\beta_t}$).

### Closed-Form Jump (The $\bar{\alpha}$ Trick)
By recursively substituting $x_{t-1}$ in terms of $x_{t-2}$, then $x_{t-2}$ in terms of $x_{t-3}$, etc., you can collapse all $T$ steps into a **single closed-form equation**:

$$x_t = \sqrt{\bar{\alpha}_t}\; x_0 + \sqrt{1 - \bar{\alpha}_t}\; \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

where:
$$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$$

This is extremely important because it means **you can jump from $x_0$ to any arbitrary $x_t$ in a single computation** — no need to iterate through all intermediate steps.

### Key Insight: No Neural Network Needed
Forward diffusion is **not a learned process**. There are no trainable parameters. All you need is:
1.  The original image $x_0$.
2.  A predefined **alpha schedule** that specifies how $\alpha_t$ (and therefore $\beta_t$) change across time steps.
3.  A random sample $\epsilon \sim \mathcal{N}(0, I)$.

This is a major difference from VAEs, where an encoder network must be trained to map images into latent space.

### Alpha Scheduling
The alpha schedule controls how aggressively noise is added:
*   $\alpha_t$ **starts close to 1** and **decreases toward 0** over the time steps.
*   $\beta_t = 1 - \alpha_t$ starts close to 0 and increases toward 1.
*   $\bar{\alpha}_t$ (the cumulative product) therefore decays from ~1 to ~0.

**Interpretation of $\bar{\alpha}_t$:**
| $\bar{\alpha}_t$ value | Meaning |
|---|---|
| Close to 1 (early steps) | Image information is largely preserved; very little noise |
| Close to 0 (late steps) | Almost no original image information remains; nearly pure noise |

The schedule shape (linear, cosine, etc.) and the total number of steps $T$ determine how quickly the image is destroyed. An aggressive schedule can produce near-pure noise in ~100 steps; a gentler schedule may require 1,000+ steps.

### Physical Analogy
The process mirrors physical **diffusion** — like placing a watercolor painting in rain. The colors gradually blur, mix, and spread until the result is a uniform, meaningless wash. In the same way, pixel values that once formed a structured distribution (e.g., a trimodal histogram for a particular image) gradually converge to a single-mode Gaussian distribution.

---

## Part 2: DDPM Sampling / Denoising (Noise → Image)

### Goal
Starting from pure Gaussian noise $x_T$, iteratively produce $x_{T-1}, x_{T-2}, \dots, x_0$ such that $x_0$ resembles a sample from the original image distribution.

### The Denoising Equation
One step of DDPM sampling is:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \; \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t}\; z, \qquad z \sim \mathcal{N}(0, I)$$

Breaking this down into three components:

#### Component 1: Scaled Noise Subtraction
$$x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \; \epsilon_\theta(x_t, t)$$

*   $\epsilon_\theta(x_t, t)$ is the **noise predicted by the U-Net** given the current noisy image $x_t$ and the current time step $t$.
*   The prefactor $\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}$ can be decomposed as $\frac{\sqrt{1-\alpha_t} \cdot \sqrt{1-\alpha_t}}{\sqrt{1 - \bar{\alpha}_t}}$:
    *   $\sqrt{1 - \alpha_t}$ is the noise scale for the **current step** (from the forward equation).
    *   $\sqrt{1 - \bar{\alpha}_t}$ is the **total cumulative noise** scale from step 0 to step $t$.
    *   Together, this ratio points the denoising in the direction of the total noise removal but calibrated for just the current step.

#### Component 2: Overall Scaling by $\frac{1}{\sqrt{\alpha_t}}$
This mirrors rearranging the forward diffusion equation to isolate $x_{t-1}$ on the left-hand side.

#### Component 3: Stochastic Noise Re-injection $\sqrt{\beta_t} \cdot z$
A small amount of **fresh random noise** is added back after denoising. This is critical and perhaps the most counter-intuitive part of DDPM.

### Why Re-inject Noise? (The Stochasticity Argument)
Without this term, the denoising process becomes essentially deterministic: you are subtracting one Gaussian-like prediction from another Gaussian-like signal, which causes all generated images to **collapse toward a blurry mean** — regardless of the starting noise. The generated images lose diversity and sharpness.

**2D Manifold Intuition:** Imagine the data distribution as a spiral in 2D pixel space. Denoising should navigate from a random point in Gaussian space back onto the spiral. Without stochastic re-injection, all paths collapse to the same blurry region near the center. With it, the random perturbations at each step allow different starting points to explore different paths and land on diverse, sharp points along the spiral.

**Key detail:** The magnitude of re-injected noise is $\sqrt{\beta_t}$, which **decreases as $t \to 0$**. Early denoising steps (high $t$) add more random perturbation; later steps (low $t$, close to the final image) add almost none, preserving the structure that has formed.

---

## Part 3: Time Embedding

The U-Net must know **which time step** it is currently denoising at, because the amount of noise present varies dramatically between $t = T$ (pure noise) and $t = 1$ (nearly clean).

*   The scalar time step $t$ is encoded via **sinusoidal (sine/cosine) positional encoding**, analogous to the positional encoding in the original "Attention Is All You Need" Transformer.
*   The sinusoidal encoding is then passed through a small **MLP** to project it into the appropriate embedding dimension (e.g., 256).
*   This time embedding vector is injected into **every residual block** of the U-Net, conditioning the network's predictions on the current noise level.

---

## Part 4: U-Net Architecture (Noise Predictor Network)

### Inputs and Outputs
*   **Input 1:** $x_t$ — the noisy image at the current time step (e.g., shape $B \times 1 \times 28 \times 28$ for grayscale MNIST).
*   **Input 2:** $t$ — the time step, converted to a time embedding vector.
*   **Output:** $\epsilon_\theta(x_t, t)$ — the predicted noise (same spatial shape as the input image).

### Why "U-Net"?
The architecture forms a **U-shape**: spatial resolution decreases (downsampling) while channels increase, hits a bottleneck, then spatial resolution increases (upsampling) while channels decrease back to the original. Skip connections bridge corresponding levels.

### Architecture Walkthrough

#### Initial Convolution
$B \times 1 \times 28 \times 28 \xrightarrow{\text{Conv2d}} B \times 64 \times 28 \times 28$

Channels increase from 1 (grayscale) to 64; spatial resolution unchanged.

#### Downsampling Path
| Operation | Output Shape | Notes |
|---|---|---|
| Residual Block ×2 | $B \times 64 \times 28 \times 28$ | Time embedding injected |
| Average Pooling | $B \times 64 \times 14 \times 14$ | Spatial halved |
| Residual Block ×2 | $B \times 128 \times 14 \times 14$ | Channels doubled, time embedding injected |

#### Bottleneck
| Operation | Output Shape | Notes |
|---|---|---|
| Residual Block | $B \times 128 \times 14 \times 14$ | Time embedding injected |
| **Self-Attention** | $B \times 128 \times 14 \times 14$ | Flatten spatial dims → sequence, apply transformer-style self-attention |
| Residual Block | $B \times 128 \times 14 \times 14$ | Time embedding injected |

The self-attention layer is where the **Transformer** component enters DDPM. The 2D feature map is flattened into a sequence, self-attention is computed, and the result is reshaped back. This allows the model to capture global dependencies.

#### Upsampling Path
| Operation | Output Shape | Notes |
|---|---|---|
| Upsample | $B \times 128 \times 28 \times 28$ | Spatial doubled |
| **Concatenate** skip connection | $B \times 256 \times 28 \times 28$ | Features from downsampling path |
| Residual Block ×2 | $B \times 64 \times 28 \times 28$ | Channels reduced, time embedding injected |

#### Final Convolution
$B \times 64 \times 28 \times 28 \xrightarrow{\text{Conv2d}} B \times 1 \times 28 \times 28$

Output is $\epsilon_\theta$ — the predicted noise, same shape as the input image.

### Skip Connections
Like in ResNet, skip connections from downsampling layers to corresponding upsampling layers improve gradient flow and allow the network to preserve fine spatial details that would otherwise be lost through the bottleneck.

### Symmetry
The architecture is highly symmetric:
*   Input Conv ↔ Output Conv
*   2 Residual Blocks in down ↔ 2 Residual Blocks in up
*   2 Residual Blocks in bottleneck (with self-attention in the middle)
*   Avg Pooling (down) ↔ Upsample (up)

---

## Training the U-Net

### Training Objective
The U-Net is trained to predict the noise $\epsilon$ that was added during forward diffusion. The loss is a simple **Mean Squared Error** between the predicted noise and the actual noise:

$$\mathcal{L} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

### Training Procedure
For each training step:
1.  Sample a clean image $x_0$ from the dataset.
2.  Sample a random time step $t \sim \text{Uniform}(1, T)$.
3.  Sample random noise $\epsilon \sim \mathcal{N}(0, I)$.
4.  Construct the noisy image using the closed-form: $x_t = \sqrt{\bar{\alpha}_t}\; x_0 + \sqrt{1 - \bar{\alpha}_t}\; \epsilon$.
5.  Pass $x_t$ and $t$ through the U-Net to get $\epsilon_\theta(x_t, t)$.
6.  Compute loss $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$ and backpropagate.

### What the U-Net Learns
The U-Net does **not** learn to predict the noise added in a single step ($x_{t-1} \to x_t$). It learns to predict the **total noise** $\epsilon$ associated with the entire forward diffusion from $x_0$ to $x_t$. During sampling, the scaling prefactor $\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}$ ensures only the appropriate fraction of this predicted noise is subtracted at each step.

**Analogy:** The U-Net knows the general "direction" back to the clean data manifold (total noise), but the scaling ensures it only takes one small step at a time in that direction.

---

## Generation / Inference (Putting It All Together)

1.  Sample $x_T \sim \mathcal{N}(0, I)$ — pure random noise.
2.  For $t = T, T-1, \dots, 1$:
    *   Predict noise: $\epsilon_\theta(x_t, t)$ using the trained U-Net.
    *   Sample fresh noise: $z \sim \mathcal{N}(0, I)$ (set $z = 0$ for the final step $t = 1$).
    *   Compute: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} \cdot z$
3.  The final $x_0$ is the generated image.

If trained on MNIST, the generated image will resemble a handwritten digit. If trained on flowers, it will resemble a flower. The model generates images that belong to the same distribution as the training data.

---

## Summary of Key Concepts

| Concept | Description |
|---|---|
| **Forward Diffusion** | Deterministic formula that adds noise to an image over $T$ steps; no neural network needed |
| **$\bar{\alpha}_t$ (alpha bar)** | Cumulative product of all $\alpha_i$ up to step $t$; enables one-shot jump from $x_0$ to $x_t$ |
| **Alpha Scheduling** | Predefined curve controlling how $\alpha_t$ and $\beta_t$ change over time; analogous to learning rate scheduling |
| **DDPM Sampling** | Iterative denoising from $x_T$ to $x_0$ using U-Net predictions |
| **U-Net** | Noise-predicting network with encoder-bottleneck-decoder structure, skip connections, and self-attention |
| **Time Embedding** | Sinusoidal encoding of the current time step, injected into all residual blocks |
| **Noise Re-injection** | Adding $\sqrt{\beta_t} \cdot z$ at each denoising step to maintain diversity and prevent mode collapse to blurry outputs |
| **Loss Function** | MSE between predicted noise $\epsilon_\theta$ and actual noise $\epsilon$ used in forward diffusion |

## References
*   **(Ho et al., 2020) Denoising Diffusion Probabilistic Models**: The foundational DDPM paper from UC Berkeley.
*   **Welch Labs (on 3Blue1Brown channel)**: Highly recommended visual explanation of DDPM with excellent animations.
