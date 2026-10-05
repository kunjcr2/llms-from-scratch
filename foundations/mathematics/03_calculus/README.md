# ∫ Calculus for Machine Learning

> *"To understand gradient descent, you must first understand gradients. To understand gradients, you must understand derivatives. To understand derivatives, you must understand limits. But practically — you just need to understand: which way is downhill?"*

Calculus is fundamentally about one thing: **how does a function change when its inputs change?** The derivative captures this precisely. Once you understand derivatives, gradients follow naturally — and with gradients, you can find the minimum of almost any function.

---

## Table of Contents

1. [Functions, Limits & Continuity](#1-functions-limits--continuity)
2. [Derivatives — Rate of Change](#2-derivatives--rate-of-change)
3. [Derivative Rules](#3-derivative-rules)
4. [The Chain Rule — Heart of Backpropagation](#4-the-chain-rule--heart-of-backpropagation)
5. [Partial Derivatives](#5-partial-derivatives)
6. [Gradients — Derivatives in Multiple Dimensions](#6-gradients--derivatives-in-multiple-dimensions)
7. [The Jacobian](#7-the-jacobian)
8. [The Hessian](#8-the-hessian)
9. [Gradient Descent — From Scratch](#9-gradient-descent--from-scratch)
10. [Loss Landscapes — Local Minima, Saddle Points & Convexity](#10-loss-landscapes--local-minima-saddle-points--convexity)
11. [Backpropagation — The Chain Rule in Action](#11-backpropagation--the-chain-rule-in-action)
12. [Advanced Optimisers](#12-advanced-optimisers)

---

## 1. Functions, Limits & Continuity

### What is a Function?

A function $f: \mathbb{R}^n \to \mathbb{R}^m$ maps inputs to outputs. Common examples:

- $f(x) = x^2$ maps a number to its square
- $f(x, y) = x^2 + y^2$ maps a 2D point to a scalar
- $f(\mathbf{x}) = \mathbf{Ax}$ maps a vector to another vector via matrix multiplication

### The Limit — Formalising "Getting Close"

$$\lim_{x \to a} f(x) = L$$

Means: as $x$ gets arbitrarily close to $a$, $f(x)$ gets arbitrarily close to $L$.

**Intuition**: Think of walking toward $x = a$ from both sides. If you always end up approaching the same value $L$ regardless of which side you approach from — that's the limit. The key word is "approaching" — you never have to actually reach $a$, only get close enough that $f(x)$ stabilizes.

### Continuity

$f$ is **continuous** at $a$ if $\lim_{x \to a} f(x) = f(a)$ — no jumps or holes in the graph.

**Why it matters for calculus**: You can only take the derivative of a function if it's continuous (and "smooth"). Discontinuous functions have sudden jumps — you can't define a tangent line at a jump.

---

## 2. Derivatives — Rate of Change

### Definition from First Principles

The derivative $f'(a)$ is the slope of $f$ at $a$:

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

**Intuition**: You're computing rise-over-run (slope) from $a$ to $a+h$, then asking what happens as $h$ shrinks to zero. The secant line (between two points) becomes the tangent line (touching at one point). This limiting slope is the derivative.

Alternative notation: $\frac{df}{dx}\bigg|_{x=a}$, or simply $\dot{f}(a)$.

**Geometric meaning**: Slope of the tangent line to $f$ at $a$.

**Physical meaning**: If $f(t)$ is position at time $t$, then $f'(t)$ is instantaneous velocity — how quickly position is changing right now.

### Common Derivatives

| Function $f(x)$ | Derivative $f'(x)$ |
|-----------------|-------------------|
| $c$ (constant) | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $1/x$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |

The sigmoid derivative is elegant: $\sigma'(x) = \sigma(x)(1-\sigma(x))$. If you already know $\sigma(x)$, the derivative costs almost nothing to compute — you reuse the forward pass value. If $\sigma(x) = 0.8$, then $\sigma'(x) = 0.8 \times 0.2 = 0.16$. No extra computation needed.

---

## 3. Derivative Rules

### Power Rule

$$\frac{d}{dx} x^n = n x^{n-1}$$

**Why**: Think of $x^n = x \cdot x \cdot x \cdots$ ($n$ times). When you differentiate a product, each factor takes a turn being differentiated while the others stay put (product rule, applied $n$ times). The result is $n$ copies of $x^{n-1}$.

### Sum Rule

$$\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$$

How fast a sum changes = sum of how fast each part changes. Completely intuitive.

### Product Rule

$$\frac{d}{dx}[f(x) \cdot g(x)] = f'(x) g(x) + f(x) g'(x)$$

**Intuition**: Imagine a rectangle with sides $f(x)$ and $g(x)$. When $x$ changes by a tiny bit $dx$, both sides change. The change in area is: the new width times the original height ($f'g \cdot dx$) plus the original width times the new height ($fg' \cdot dx$). That's exactly the product rule.

### Quotient Rule

$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

Derived from the product rule: write $f/g = f \cdot g^{-1}$ and apply product rule plus chain rule on $g^{-1}$.

### Chain Rule ← Most Important

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

Or with $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Intuition**: Think of it as exchange rates. If 1 dollar buys 0.9 euros, and 1 euro buys 130 yen, then 1 dollar buys 0.9 × 130 = 117 yen. The chain rule is the same: how much does $y$ change per unit change in $u$, times how much $u$ changes per unit change in $x$.

---

## 4. The Chain Rule — Heart of Backpropagation

The chain rule computes derivatives of **composed functions**. Neural networks are compositions of functions — each layer applies a function to the previous layer's output — so the chain rule is literally what makes backpropagation work.

### Extended Chain Rule

For $y = f_n(f_{n-1}(\cdots f_1(x) \cdots))$:

$$\frac{dy}{dx} = \frac{df_n}{df_{n-1}} \cdot \frac{df_{n-1}}{df_{n-2}} \cdots \frac{df_1}{dx}$$

Think of a chain of currency exchanges: $y$ in terms of $f_{n-1}$, $f_{n-1}$ in terms of $f_{n-2}$, all the way back to $x$. The rate of change from $x$ to $y$ is the product of all the exchange rates along the chain.

### Example: 2-Layer Network

$$z_1 = \mathbf{w}_1^T \mathbf{x}, \quad a_1 = \sigma(z_1), \quad z_2 = w_2 a_1, \quad \mathcal{L} = (y - z_2)^2$$

By chain rule:

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

$$= 2(z_2 - y) \cdot w_2 \cdot \sigma'(z_1) \cdot x$$

**Reading this backwards from output to input**: We start with how the loss responds to $z_2$, multiply by how $z_2$ responds to $a_1$, multiply by how $a_1$ responds to $z_1$ (which involves the sigmoid derivative), and finally by how $z_1$ responds to $w_1$ (which is just $x$). The gradient of $w_1$ is a *product of four local exchange rates*, each one answering "how does this piece affect the next piece?"

The gradient of the loss w.r.t. a parameter deep in the network = **product of all downstream derivatives**. This is backpropagation.

---

## 5. Partial Derivatives

When $f$ depends on multiple variables, the **partial derivative** w.r.t. $x_i$ treats all other variables as constants:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

**Intuition**: You're asking "if I only wiggle $x_i$ — and hold everything else absolutely fixed — how does $f$ respond?" Freezing the other variables reduces a multivariable function back to a single-variable problem, and you differentiate that.

### Example

For $f(x, y) = x^2 + 3xy + y^2$:

$$\frac{\partial f}{\partial x} = 2x + 3y \quad (\text{treat } y \text{ as constant})$$
$$\frac{\partial f}{\partial y} = 3x + 2y \quad (\text{treat } x \text{ as constant})$$

The $3xy$ term contributes $3y$ to $\partial f/\partial x$ (because $y$ is just a constant multiplier), and $3x$ to $\partial f/\partial y$. Same symmetry, different perspective.

---

## 6. Gradients — Derivatives in Multiple Dimensions

The **gradient** of $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of all partial derivatives:

$$\nabla_\mathbf{x} f = \begin{pmatrix} \partial f / \partial x_1 \\ \partial f / \partial x_2 \\ \vdots \\ \partial f / \partial x_n \end{pmatrix} \in \mathbb{R}^n$$

Think of the gradient as the multi-dimensional analog of the derivative. For a single-variable function, the derivative is a number telling you slope. For multi-variable functions, the gradient is a **vector** pointing in the direction of steepest ascent.

### Key Properties

- The gradient **points in the direction of steepest ascent**
- Moving **against** the gradient ($-\nabla f$) is the direction of steepest descent
- The gradient is **perpendicular to the level curves** of $f$
- At a minimum: $\nabla f(\mathbf{x}^*) = \mathbf{0}$

**Intuition for "perpendicular to level curves"**: A level curve is where $f = c$ (constant). If you move along a level curve, $f$ doesn't change — so there's no slope along it. The direction of maximum slope must therefore be perpendicular to the level curve. Imagine a bowl: the "altitude lines" (level curves) run horizontally around the bowl, and the steepest direction is straight down — which is perpendicular to those rings.

### The Directional Derivative

The rate of change of $f$ in direction $\mathbf{u}$ (unit vector):

$$D_\mathbf{u} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos\theta$$

Maximum when $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$ — confirming the gradient is the steepest direction. When $\theta = 90°$ (moving perpendicularly to the gradient), $\cos\theta = 0$ — you're on a level curve, no change in $f$.

The negative gradient $-\nabla f$ points toward the steepest decrease. This is the geometric reason why **gradient descent** works — by taking small steps in that direction, you walk downhill on the function surface.

---

## 7. The Jacobian

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ (vector-valued function), the **Jacobian** is the matrix of all partial derivatives:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & \ddots & \vdots \\ \partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n \end{pmatrix} \in \mathbb{R}^{m \times n}$$

**Intuition**: When your function takes a vector in and spits a vector out, the derivative is no longer a single number — it's a whole matrix. Each row $i$ answers "how does output $f_i$ respond to each of the inputs?" The Jacobian $\mathbf{J}$ is the best linear approximation to $\mathbf{f}$ near a point: $\mathbf{f}(\mathbf{x} + \mathbf{d}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}\mathbf{d}$.

The gradient is a special case: when $m=1$, $\mathbf{J} = \nabla f^T$ — the single row of the Jacobian is just the gradient transposed.

**When you encounter the Jacobian**: Any time you're computing gradients through a layer that maps vectors to vectors (e.g., the softmax layer, a batch normalization layer, or a neural network that outputs a vector). The Jacobian tells you how to propagate gradients backwards through that layer.

---

## 8. The Hessian

For $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** is the matrix of second derivatives:

$$\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

$$\mathbf{H} = \nabla^2 f = \begin{pmatrix} \partial^2 f/\partial x_1^2 & \partial^2 f/\partial x_1 \partial x_2 & \cdots \\ \partial^2 f/\partial x_2 \partial x_1 & \partial^2 f/\partial x_2^2 & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

**Intuition**: The gradient tells you which way the function is sloping. The Hessian tells you how the slope itself is changing — the **curvature**. At a mountain top, the gradient is zero but the Hessian is "all negative" (curves downward in every direction). In a valley, gradient is also zero but the Hessian is "all positive" (curves upward).

The Hessian is **symmetric** ($H_{ij} = H_{ji}$ by Schwarz's theorem) — so it has real eigenvalues and orthogonal eigenvectors.

### What the Hessian Tells You

At a critical point ($\nabla f = 0$):
- All eigenvalues of $\mathbf{H}$ positive → **local minimum** (curves up in every direction — bowl)
- All eigenvalues negative → **local maximum** (curves down in every direction — hill)
- Mixed signs → **saddle point** (curves up in some directions, down in others — like a saddle or a mountain pass)

### Condition Number & Speed of Convergence

The **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ measures how *asymmetrically* the function curves.

**Intuition**: Imagine a loss landscape shaped like a bowl. If the bowl is perfectly circular (equal curvature in all directions), gradient descent takes a straight shot to the bottom. But if the bowl is stretched into a long, narrow valley — high curvature in one direction, very low curvature in the other — you have a large condition number. Gradient descent then has to zigzag slowly back and forth across the narrow valley, making steps that are too large in the steep direction and too small in the flat direction. Large $\kappa$ → many small zig-zag steps → slow convergence.

This is why **preconditioning** (scaling coordinates to make the loss landscape more circular) dramatically speeds up optimization.

### Newton's Method

**Newton's method** uses the Hessian to take curvature-aware steps:

$$\theta \leftarrow \theta - \mathbf{H}^{-1} \nabla f$$

**Intuition**: Instead of stepping blindly in the gradient direction, Newton's method asks "given the curvature, how far should I step?" In flat directions (low curvature), it takes larger steps. In steep directions (high curvature), it takes smaller steps. This is exactly what the inverse Hessian does — it rescales the gradient by the curvature, giving curvature-corrected steps. This converges much faster than vanilla gradient descent for smooth functions.

The downside: inverting $\mathbf{H}$ costs $O(n^3)$, which is infeasible for large problems (imagine inverting a billion-parameter weight matrix). That's why in practice we use quasi-Newton methods (like L-BFGS) that approximate $\mathbf{H}^{-1}$ without forming it explicitly, or adaptive optimizers like Adam that approximate per-parameter curvature for much less cost.

---

## 9. Gradient Descent — From Scratch

The fundamental parameter update rule:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

Where $\eta$ is the **learning rate** — controls step size. Too large: you overshoot and diverge. Too small: you converge agonizingly slowly. Getting $\eta$ right is one of the most important practical tuning problems in deep learning.

### Variants

| Variant | Update | Batch Size |
|---------|--------|-----------|
| Batch GD | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$ | Entire dataset |
| Stochastic GD (SGD) | $\theta \leftarrow \theta - \eta \nabla_\theta \ell(\theta; x_i, y_i)$ | 1 sample |
| Mini-batch SGD | $\theta \leftarrow \theta - \eta \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_\theta \ell_i$ | Batch of $B$ |

**Why mini-batch?** Batch GD uses the full dataset for every step — cheap in terms of steps needed, expensive per step. SGD is cheap per step but jumpy (single sample gradient is noisy). Mini-batch is a practical compromise: smooth enough gradients, fast enough updates. The noise in mini-batch SGD also acts as a regularizer — the randomness helps escape sharp local minima.

### SGD with Momentum

```
v_t = β v_{t-1} + (1-β) ∇L(θ_t)
θ_{t+1} = θ_t - η v_t
```

**Intuition**: Without momentum, gradient descent makes independent steps at each iteration — it can oscillate in the narrow valley (condition number problem). Momentum keeps a *running average* of gradients. Directions where gradients consistently point the same way build up momentum; directions where gradients cancel out get dampened. Like a ball rolling downhill: it accelerates in consistent downhill directions and smooths out the oscillations across the valley walls.

$\beta$ (typically 0.9) controls how long momentum persists. $\beta = 0.9$ means each step uses ~10 recent gradient steps worth of memory.

### Adam (Adaptive Moment Estimation)

Keeps running estimates of first moment (mean gradient) $m_t$ and second moment (mean squared gradient) $v_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t) \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Intuition for the two moments**: $m_t$ is a running average of recent gradients — like momentum, it tells you "which direction have gradients been pointing?" $v_t$ is a running average of recent *squared* gradients — it tells you "how large have gradients been in each direction?"

Dividing by $\sqrt{v_t}$: parameters that have seen large gradients in the past will have large $v_t$, so they get a *smaller* effective learning rate. Parameters that have seen small gradients get a *larger* effective learning rate. Adam is essentially giving each parameter its own auto-tuned learning rate based on its own history.

**Why bias correction?** At $t = 1$, both $m_t$ and $v_t$ start at zero (cold start). With $\beta_1 = 0.9$, after the first step: $m_1 = 0.9 \times 0 + 0.1 \times g_1 = 0.1 g_1$. This massively underestimates the true mean gradient — we only saw one gradient, but we're averaging it with 9 parts of nothing. Dividing by $(1 - \beta_1^t) = (1 - 0.9^1) = 0.1$ corrects this: $\hat{m}_1 = 0.1 g_1 / 0.1 = g_1$. As $t \to \infty$, $\beta_1^t \to 0$, so $(1 - \beta_1^t) \to 1$, and bias correction has no effect — you've seen enough data to trust the running average.

---

## 10. Loss Landscapes — Local Minima, Saddle Points & Convexity

### Convex vs Non-Convex

**Intuition first**: A function is convex if its shape is "bowl-like" — no valleys within valleys, no ridges. If you pick any two points on the function and draw a straight line between them, the function lies *below* that line. A convex function has one globally lowest point.

The formal definition: $f$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0,1]$:

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

The left side is the function value at a blend of $\mathbf{x}$ and $\mathbf{y}$. The right side is the same blend of the function values. The inequality says: the function at the blend is no worse than the blend of the function values — which is exactly "the chord lies above the function."

- **Convex**: Every local minimum is a global minimum — gradient descent is guaranteed to find it. Simple, beautiful, reliable.
- **Non-convex**: Multiple local minima and saddle points may exist. Neural network losses are non-convex. Gradient descent can get stuck, but in practice deep networks seem to have many good local minima that are approximately equivalent.

### Types of Critical Points (where $\nabla f = 0$)

| Type | Gradient | Hessian Eigenvalues |
|------|----------|---------------------|
| Global minimum | $\mathbf{0}$ | All positive |
| Local minimum | $\mathbf{0}$ | All positive |
| Saddle point | $\mathbf{0}$ | Mixed signs |
| Maximum | $\mathbf{0}$ | All negative |

**Saddle points in high dimensions**: In a 1000-dimensional loss landscape, a saddle point requires mixed Hessian eigenvalues — some positive, some negative. The deeper you are in a network, the more dimensions there are, and the *more likely* you are to encounter saddle points rather than local minima. Gradient descent near a saddle point slows down dramatically (gradient ≈ 0), but momentum and noise from SGD typically help escape them.

---

## 11. Backpropagation — The Chain Rule in Action

Backprop is just the chain rule applied systematically through a computational graph. For each layer:

**Forward pass**: Compute and cache intermediate values ($z^{(l)}$, $a^{(l)}$). You cache them because you'll need them during the backward pass.

**Backward pass**: Start at the loss, propagate gradients backwards from output to input using the chain rule.

### The Four Fundamental Backprop Equations (Neural Network)

For layer $l$ with $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$:

$$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} \mathcal{L} \odot \sigma'(\mathbf{z}^{(L)}) \quad \text{(output layer error)}$$
$$\boldsymbol{\delta}^{(l)} = \left((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)}) \quad \text{(hidden layer error)}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T \quad \text{(weight gradient)}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)} \quad \text{(bias gradient)}$$

Where $\odot$ is elementwise multiplication (Hadamard product).

**Reading these equations**:

$\boldsymbol{\delta}^{(l)}$ is the "error signal" at layer $l$ — how much this layer's pre-activations ($\mathbf{z}^{(l)}$) are responsible for the total loss.

Equation 1: The output error is the loss gradient w.r.t. outputs, **gated** by how much the activation function was responding. If $\sigma'$ is near zero (saturated sigmoid), the error signal gets killed — this is the **vanishing gradient problem**.

Equation 2: To propagate error backwards through a weight matrix, you use the *transpose* of that weight matrix. Intuitively: if $\mathbf{W}^{(l+1)}$ routes information forward (from $\mathbf{a}^{(l)}$ to $\mathbf{z}^{(l+1)}$), then $(\mathbf{W}^{(l+1)})^T$ routes error signals backward via the same connections.

Equations 3 & 4: Once you have $\boldsymbol{\delta}^{(l)}$, the weight gradient is just an outer product (which layer added it times how much it matters), and the bias gradient is just $\boldsymbol{\delta}^{(l)}$ directly.

---

## 12. Advanced Optimisers

### Learning Rate Schedules

- **Step decay**: Multiply by $\gamma$ every $k$ epochs — learning rate drops like a staircase. Simple and effective.
- **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ — smoothly oscillates from $\eta_{\max}$ down to $\eta_{\min}$ over the training period. The smooth decay avoids the sharp drop of step decay and lets the optimizer "explore" before settling.
- **Warmup + Decay**: Start small, ramp up, then decay. This is the most common schedule in large-scale training (e.g., transformers).

### Why Warmup?

At the very beginning of training, the model is in a terrible state — weights are random, gradients are large and unreliable, and the variance in gradient estimates is huge. If you start with a large learning rate, a single bad batch can throw the model wildly off course. The model hasn't yet learned which directions are generally good vs. noisy.

By starting with a tiny learning rate, you let the model take a few "careful" steps and build up a more reliable picture of the landscape. The moving averages in optimizers like Adam also need a few steps to warm up (their bias correction helps, but early estimates are still rough). Once training stabilizes, you ramp the learning rate up to its full value, then decay it as you approach convergence.

In transformers (the "Attention is All You Need" paper), the warmup schedule is: $\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5},\ t \cdot t_{\text{warmup}}^{-1.5})$.

### Gradient Clipping

$$\text{if } \|\mathbf{g}\|_2 > \tau: \quad \mathbf{g} \leftarrow \frac{\tau}{\|\mathbf{g}\|_2} \mathbf{g}$$

**Intuition**: Occasionally, especially with recurrent networks, a gradient can explode — the chain rule multiplies many numbers together and the product can be astronomically large. A single weight update with a massive gradient can completely destroy weeks of training progress. Gradient clipping says: "if the total gradient magnitude exceeds threshold $\tau$, rescale the gradient vector to have norm exactly $\tau$." This preserves the direction (you still step in the right direction) but limits the size of the step. It's a simple safeguard that costs almost nothing but prevents catastrophic updates.

---

## 📓 Notebook

Open [`calculus.ipynb`](./calculus.ipynb) for hands-on derivations, gradient descent from scratch, backpropagation implementation, and loss landscape visualisations.
