# Direct Preference Optimization (DPO): Loss & Gradients (Quick Guide)

This note explains how the **DPO** loss is computed and what happens when we call `loss.backward()`—i.e., how gradients flow to update the policy.

---

## 1) Setup

- **Policy** (trainable): $\pi_\theta(y \mid x)$
- **Reference model** (frozen): $\pi_{\text{ref}}(y \mid x)$ — typically the base pretrained model.
- **Data**: preference pairs per prompt $x$: a **preferred** completion $y^{+}$ and a **rejected** completion $y^{-}$.
- **Hyperparameter**: temperature $\beta > 0$ (how sharply we enforce preferences).

We work with **sequence log-probabilities** (sum of token log-probs):

$$
\log \pi_\theta(y \mid x) \,=\, \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t}).
$$

---

## 2) DPO objective

DPO treats pairwise preferences as a **logistic regression** over a *relative advantage* of the preferred sequence compared to the rejected one, **relative to the reference model**:

$$
\mathcal{L}_{\text{DPO}}(\theta)
= - \mathbb{E}_{(x, y^+, y^-)} \Big[ \log \sigma \Big( 
\beta \Big[ 
\underbrace{\log \pi_\theta(y^+ \mid x) - \log \pi_\theta(y^- \mid x)}_{\text{policy gap}}
\;-\;
\underbrace{\big( \log \pi_{\text{ref}}(y^+ \mid x) - \log \pi_{\text{ref}}(y^- \mid x) \big)}_{\text{reference gap (constant)}}
\Big] \Big) \Big].
$$

- $\sigma(\cdot)$ is the **sigmoid**.
- The **reference gap** is a constant w.r.t. $\theta$ (it does **not** backprop).
- The **policy gap** is *learned*: increasing it makes the preferred sequence relatively more likely under $\pi_\theta$.

---

## 3) Per-pair loss (batch implementation)

For a minibatch of size $B$, define for each pair $i$:

$$
z_i \,=\, \beta \Big[ 
\big(\log \pi_\theta(y_i^+ \mid x_i) - \log \pi_\theta(y_i^- \mid x_i)\big)
-
\big(\log \pi_{\text{ref}}(y_i^+ \mid x_i) - \log \pi_{\text{ref}}(y_i^- \mid x_i)\big)
\Big].
$$

Then the per-example loss and the batch loss are:

$$
\ell_i \,=\, -\log \sigma(z_i), \qquad
\mathcal{L} \,=\, \frac{1}{B} \sum_{i=1}^B \ell_i.
$$

---

## 4) What `loss.backward()` actually does

Let’s look at the derivative for one pair (drop index $i$ for clarity). Define $ z = \beta \, \Delta $, where

$$
\Delta
= \big(\log \pi_\theta(y^+ \mid x) - \log \pi_\theta(y^- \mid x)\big)
- \big(\log \pi_{\text{ref}}(y^+ \mid x) - \log \pi_{\text{ref}}(y^- \mid x)\big).
$$

Since $ \ell = -\log \sigma(z) $, we have:

$$
\frac{\partial \ell}{\partial z}
= \sigma(-z)
= 1 - \sigma(z).
$$

Chain rule gives:

$$
\frac{\partial \ell}{\partial \theta}
= \frac{\partial \ell}{\partial z} \cdot \frac{\partial z}{\partial \theta}
= \big(1 - \sigma(z)\big) \cdot \beta \cdot \frac{\partial \Delta}{\partial \theta}.
$$

And

$$
\frac{\partial \Delta}{\partial \theta}
= \frac{\partial}{\partial \theta}\log \pi_\theta(y^+ \mid x)
- \frac{\partial}{\partial \theta}\log \pi_\theta(y^- \mid x).
$$

So, **`loss.backward()`** accumulates gradients that:

- **Increase** $ \log \pi_\theta(y^+ \mid x) $ (push tokens in $y^+$ up),
- **Decrease** $ \log \pi_\theta(y^- \mid x) $ (push tokens in $y^-$ down),  

scaled by $ \beta \cdot (1 - \sigma(z)) $.

---

## 5) Token-level view

Because sequence log-prob is the **sum of token log-probs**, the gradient distributes over time steps:

$$
\frac{\partial}{\partial \theta}\log \pi_\theta(y \mid x)
= \sum_{t=1}^{T} \frac{\partial}{\partial \theta}\log \pi_\theta(y_t \mid x, y_{<t}).
$$

This is exactly what teacher-forcing cross-entropy does—except here it’s applied to **two sequences with opposite signs** and **weighted by the logistic factor** above.

---

## 6) TL;DR

DPO minimizes

$$
-\log \sigma\Big(\beta\big[(\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x)) - (\log \pi_{\text{ref}}(y^+|x) - \log \pi_{\text{ref}}(y^-|x))\big]\Big).
$$

Calling `loss.backward()` **increases** the log-prob of the preferred response and **decreases** that of the rejected response (relative to the reference), with gradients strongest when the model is unsure or wrong.
