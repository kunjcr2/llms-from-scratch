# Adam Optimizer (Adaptive Moment Estimation)

> Part 4 of 4 in the optimization series: SGD → Momentum → RMSProp → **Adam**

---

## What is Adam?

**Adam = Adaptive Moment Estimation**  
(Not named after a scientist!)

> Adam combines the **best of both worlds**: Momentum + RMSProp

| Component | What it provides |
|-----------|------------------|
| **Momentum** (1st moment) | Average direction of gradients |
| **RMSProp** (2nd moment) | Adaptive learning rate based on gradient magnitude |

---

## Recap: Building Blocks

### Vanilla Gradient Descent
$$\theta_t = \theta_{t-1} - \eta \cdot g_t$$

### Momentum
$$\theta_t = \theta_{t-1} - \eta \cdot v_t$$
Where $v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot g_t$

### RMSProp
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \cdot g_t$$

### Adam (Combines Both!)
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

---

## Adam Algorithm

### Step 1: Initialize

| Parameter | Initial Value | Description |
|-----------|---------------|-------------|
| $\theta_0$ | Random | Model parameters |
| $m_0$ | 0 | First moment (mean of gradients) |
| $v_0$ | 0 | Second moment (variance of gradients) |
| $\eta$ | 0.001 - 0.1 | Learning rate |
| $\beta_1$ | 0.9 | First moment decay |
| $\beta_2$ | 0.999 | Second moment decay |
| $\epsilon$ | $10^{-8}$ | Numerical stability |

### Step 2: Compute Gradient
$$g_t = \nabla_\theta J(\theta)$$

### Step 3: Update First Moment (Momentum-like)
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

**Interpretation:** 90% previous direction + 10% current gradient

### Step 4: Update Second Moment (RMSProp-like)
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

**Interpretation:** 99.9% previous magnitude + 0.1% current gradient squared

### Step 5: Bias Correction
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Step 6: Update Parameters
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

---

## Why Bias Correction?

### The Problem
At $t=1$ with $m_0 = 0$ and $\beta_1 = 0.9$:
```
m₁ = 0.9 × 0 + 0.1 × g₁ = 0.1 × g₁
```
The moment is **biased toward zero** because of the zero initialization!

### The Solution
Divide by $(1 - \beta^t)$ to scale up early steps:

| Time Step | $1 - \beta_1^t$ | Scale Factor |
|-----------|-----------------|--------------|
| t=1 | $1 - 0.9^1 = 0.1$ | ×10 |
| t=10 | $1 - 0.9^{10} \approx 0.65$ | ×1.5 |
| t=100 | $1 - 0.9^{100} \approx 1$ | ×1 |

As $t \to \infty$, the correction factor approaches 1 (no correction needed).

---

## Adam's Two Key Contributions

### 1. Momentum Term ($\hat{m}_t$)
- Smooths gradient direction
- Carries inertia through flat regions
- Reduces noise from sparse/noisy gradients

### 2. Adaptive Learning Rate ($\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$)
- Scales down for large gradients
- Scales up for small gradients
- Per-parameter adaptation

---

## Comparison: All Optimization Methods

| Method | Gradient | Learning Rate | Oscillation Handling |
|--------|----------|---------------|---------------------|
| **Vanilla GD** | Raw $g_t$ | Fixed $\eta$ | None |
| **SGD** | Single random point | Fixed $\eta$ | Random sampling |
| **Momentum** | Velocity $v_t$ | Fixed $\eta$ | Inertia resists flipping |
| **RMSProp** | Raw $g_t$ | Adaptive $\eta_t$ | Magnitude normalization |
| **Adam** | Momentum $\hat{m}_t$ | Adaptive $\eta_t$ | **Both!** |

---

## Implementation

### Adam Optimizer from Scratch
```python
def adam_optimizer(grad_fn, lr, beta1, beta2, epsilon, epochs, start):
    x, y = start
    m = np.array([0.0, 0.0])  # First moment
    v = np.array([0.0, 0.0])  # Second moment
    
    for t in range(1, epochs + 1):
        grad = np.array(grad_fn(x, y))
        
        # Update biased moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Parameter update
        x -= (lr / (np.sqrt(v_hat[0]) + epsilon)) * m_hat[0]
        y -= (lr / (np.sqrt(v_hat[1]) + epsilon)) * m_hat[1]
    
    return x, y
```

### Key Difference from RMSProp
```diff
  # RMSProp: uses raw gradient
- x -= (lr / (sqrt(v) + eps)) * grad

  # Adam: uses momentum-adjusted gradient  
+ x -= (lr / (sqrt(v_hat) + eps)) * m_hat
```

---

## Experimental Results

### Test Case: $J = x^2 + 10y^2$
Starting point: $(1.5, 1.5)$ → Target: $(0, 0)$

### Low Learning Rate ($\eta = 0.01$)

| Method | Behavior |
|--------|----------|
| Vanilla GD | Slow but converges |
| Adam | Also converges, slightly slower initially |

### High Learning Rate ($\eta = 0.2$)

| Method | Final Loss | Behavior |
|--------|------------|----------|
| Vanilla GD | $10^{262}$ 💥 | **EXPLODES!** |
| Adam | ≈ 0 ✓ | Converges smoothly |

---

## Adam vs Momentum: Oscillation Suppression

### Path Comparison

| Method | Path Shape |
|--------|------------|
| **Momentum** | Ball rolling, oscillating path |
| **Adam** | Smoother, damped oscillations |

### Loss Curve Comparison

```
Loss
 ↑
 │  ╭╮  ╭╮  ╭╮
 │ ╱  ╲╱  ╲╱  ╲___  Momentum (oscillating)
 │╱
 │
 │╲
 │ ╲_____________  Adam (heavily damped)
 └─────────────────→ Epochs
```

**Analogy:** Both are like spring-mass-damper systems, but Adam has stronger damping.

---

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| $\eta$ | 0.001 | Often works well out-of-the-box |
| $\beta_1$ | 0.9 | First moment decay (momentum) |
| $\beta_2$ | 0.999 | Second moment decay (RMSProp) |
| $\epsilon$ | $10^{-8}$ | Numerical stability |

> **Pro tip:** Adam's defaults work well for most problems. Start with $\eta = 0.001$.

---

## When to Use Adam

### Adam Excels At:
- ✅ Deep neural networks
- ✅ Sparse gradients (NLP, embeddings)
- ✅ Noisy loss surfaces
- ✅ When you don't know the optimal learning rate
- ✅ Production machine learning models

### When Others Might Be Better:
- Simple convex problems → Vanilla GD may suffice
- Very large datasets → SGD with momentum
- Fine-tuning after Adam → SGD can find sharper minima

---

## Summary: The Adam Formula

$$\boxed{\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t}$$

Where:
- $\hat{m}_t$ = bias-corrected first moment (momentum)
- $\hat{v}_t$ = bias-corrected second moment (adaptive LR)

---

## Key Takeaways

1. **Adam = Momentum + RMSProp** — combines both techniques
2. **Bias correction** — scales up early updates to counter zero initialization
3. **Two moments**: mean (direction) and variance (magnitude) of gradients
4. **Robust to hyperparameters** — works well with defaults
5. **Industry standard** — most popular optimizer in production ML
6. **Suppresses oscillations** — better than pure momentum

---

## Complete Optimization Series Summary

| Lecture | Method | Key Innovation |
|---------|--------|----------------|
| 1 | **SGD** | Use single random point per update |
| 2 | **Momentum** | Accumulate velocity (inertia) |
| 3 | **RMSProp** | Adaptive learning rate per parameter |
| 4 | **Adam** | Combine momentum + adaptive LR + bias correction |

```
Vanilla GD
    ↓
    ├── SGD (faster on large data)
    │
    ├── Momentum (inertia through flat regions)
    │       ↘
    │         ↘
    ├── RMSProp (adaptive learning rate)
    │       ↙
    │     ↙
    └── Adam (best of both worlds) ← Most widely used!
```
