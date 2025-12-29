# RMSProp (Root Mean Square Propagation)

> Part 3 of 4 in the optimization series: SGD → Momentum → **RMSProp** → Adam

---

## Recap: Problems with Vanilla Gradient Descent

| Issue | Description |
|-------|-------------|
| **Fixed learning rate** | Same step size regardless of gradient magnitude |
| **Oscillations** | High LR + steep gradients = diverging oscillations |
| **Slow in flat regions** | Small gradients = tiny updates, crawling progress |
| **Uneven parameters** | Some gradients much larger than others |

### The Multi-Parameter Problem

For $J = x_1^2 + 100x_2^2$:
- $\frac{\partial J}{\partial x_1} = 2x_1$
- $\frac{\partial J}{\partial x_2} = 200x_2$

Same $(x_1, x_2)$ values → vastly different gradients!  
Vanilla GD treats them equally → suboptimal updates.

---

## RMSProp: Adaptive Learning Rate

### Core Philosophy
> **Normalize gradients by their magnitude history** — scale down large gradients, scale up small ones.

### Key Difference from Previous Methods

| Method | Learning Rate | Gradient |
|--------|---------------|----------|
| **Vanilla GD** | Fixed $\eta$ | Raw gradient |
| **SGD** | Fixed $\eta$ | Single-point gradient |
| **Momentum** | Fixed $\eta$ | Velocity-adjusted gradient |
| **RMSProp** | **Adaptive** $\eta_t$ | Raw gradient |

---

## Mathematical Formulation

### Step 1: Compute Gradient
$$g_t = \nabla_\theta J(\theta)$$

### Step 2: Update Moving Average of Squared Gradients
$$E[g^2]_t = \beta \cdot E[g^2]_{t-1} + (1-\beta) \cdot g_t^2$$

Where:
- $E[g^2]_t$ = moving average of squared gradients
- $\beta$ = decay rate (typically 0.9)
- $g_t^2$ = current gradient squared

### Step 3: Update Parameters with Adaptive Learning Rate
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \cdot g_t$$

Where:
- $\eta$ = base learning rate
- $\epsilon$ = small constant (e.g., $10^{-8}$) to prevent division by zero

---

## Why RMSProp Works

### Case 1: Gradient is Very Small (Flat Region)

**Problem with Vanilla GD:**
```
gradient = 10⁻⁸
Δθ = 0.1 × 10⁻⁸ = 10⁻⁹  ← Barely moving!
```

**RMSProp Solution:**
```
E[g²] ≈ 10⁻¹⁶
effective_lr = 0.1 / √(10⁻¹⁶) = 0.1 / 10⁻⁸ = 10⁷
Δθ = 10⁷ × 10⁻⁸ ≈ 0.1  ← Reasonable step!
```

> Small gradients get **amplified** by the adaptive learning rate.

### Case 2: Gradient is Very Large (Steep Region)

**Problem with Vanilla GD:**
```
gradient = -10³
Δθ = 0.1 × 10³ = 100  ← Jumping wildly!
```

**RMSProp Solution:**
```
E[g²] ≈ 10⁶
effective_lr = 0.1 / √(10⁶) = 0.1 / 10³ = 10⁻⁴
Δθ = 10⁻⁴ × 10³ ≈ 0.1  ← Controlled step!
```

> Large gradients get **dampened** by the adaptive learning rate.

---

## Effective Learning Rate Formula

$$\eta_{effective} = \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon}$$

| Gradient History | $E[g^2]$ | Effective LR | Effect |
|------------------|----------|--------------|--------|
| Small gradients | Low | High | Larger steps |
| Large gradients | High | Low | Smaller steps |

---

## RMSProp vs Momentum

| Aspect | Momentum | RMSProp |
|--------|----------|---------|
| **Modifies** | Gradient (via velocity) | Learning rate |
| **Accumulates** | Gradient direction | Gradient magnitude |
| **Goal** | Carry inertia through flat regions | Normalize step sizes |
| **Oscillation handling** | Resists direction changes | Scales down large updates |

> **Important:** RMSProp does NOT incorporate momentum. Adam combines both!

---

## Implementation

### Vanilla Gradient Descent
```python
def gradient_descent(grad_fn, lr, epochs, start):
    x, y = start
    for _ in range(epochs):
        grad = grad_fn(x, y)
        x -= lr * grad[0]
        y -= lr * grad[1]
    return x, y
```

### RMSProp
```python
def rmsprop(grad_fn, lr, beta, epsilon, epochs, start):
    x, y = start
    E_g2 = np.array([0.0, 0.0])  # Moving average of squared gradients
    
    for _ in range(epochs):
        grad = np.array(grad_fn(x, y))
        grad_sq = grad ** 2
        
        # Update moving average
        E_g2 = beta * E_g2 + (1 - beta) * grad_sq
        
        # Adaptive learning rate update
        x -= (lr / (np.sqrt(E_g2[0]) + epsilon)) * grad[0]
        y -= (lr / (np.sqrt(E_g2[1]) + epsilon)) * grad[1]
    
    return x, y
```

### Key Difference
```diff
  # Vanilla GD
- x -= lr * grad[0]

  # RMSProp
+ E_g2 = beta * E_g2 + (1 - beta) * grad**2
+ x -= (lr / (sqrt(E_g2) + epsilon)) * grad
```

---

## Hyperparameters

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| $\eta$ (learning rate) | 0.001 - 0.1 | Base step size |
| $\beta$ (decay rate) | 0.9 - 0.99 | How much history to retain |
| $\epsilon$ | $10^{-8}$ | Numerical stability |
| Epochs | 50 - 200 | Number of iterations |

---

## Experimental Results

### Test Case: $J = x^2 + 10y^2$
Starting point: $(1.5, 1.5)$ → Target: $(0, 0)$

### Low Learning Rate ($\eta = 0.01$)

| Method | Convergence | Iterations Needed |
|--------|-------------|-------------------|
| Vanilla GD | Smooth, direct | ~200 |
| RMSProp | Direct | ~200 |

### Medium Learning Rate ($\eta = 0.1$)

| Method | Behavior | Path |
|--------|----------|------|
| Vanilla GD | **Oscillating** | Zigzag (y: +1.5 → -1.5 → +1.5) |
| RMSProp | Smooth | Direct to (0,0) ✓ |

### High Learning Rate ($\eta = 0.2$)

| Method | Behavior | Final Loss |
|--------|----------|------------|
| Vanilla GD | **DIVERGES!** | $10^{96}$ 💥 |
| RMSProp | Converges | ≈ 0 ✓ |

---

## Visual Comparison

### Contour Plot (x vs y parameters)

```
     Vanilla GD (η=0.1)         RMSProp (η=0.1)
         ↓                           ↓
    ─────────────               ─────────────
   ╱             ╲             ╱             ╲
  │  ↗ ↙ ↗ ↙ ↗   │           │    ↘         │
  │    oscillating│           │      ↘  →●  │
   ╲             ╱             ╲             ╱
    ─────────────               ─────────────
   (y oscillates)              (smooth path)
```

### Loss vs Epochs

```
Loss
 ↑
 │ ╭────────────────  Vanilla GD (stuck at 22.5)
 │ │
 │ ╰╮
 │   ╲
 │    ╲______→ 0     RMSProp (converges)
 └─────────────────→ Epochs
```

---

## Key Insights

### When Vanilla GD Fails, RMSProp Succeeds

| Scenario | Vanilla GD | RMSProp |
|----------|-----------|---------|
| Flat regions | Stuck (tiny steps) | Takes larger steps |
| Steep regions | Overshoots/oscillates | Dampens steps |
| High LR | Diverges | Stays stable |
| Uneven gradients | Dominated by large ones | Normalizes all |

---

## Summary

| Property | RMSProp Behavior |
|----------|------------------|
| Learning rate | Adaptive per-parameter |
| Small gradients | Amplified (larger effective LR) |
| Large gradients | Dampened (smaller effective LR) |
| Oscillations | Prevented by scaling |
| Momentum | ❌ Not included |

---

## Key Takeaways

1. **Adaptive learning rate** — each parameter gets its own effective learning rate
2. **Normalizes by magnitude** — divides by √(moving average of squared gradients)
3. **No momentum** — only adjusts step size, not direction
4. **Handles uneven gradients** — parameters with large gradients don't dominate
5. **Enables higher learning rates** — stable even with aggressive LR
6. **Foundation for Adam** — combined with momentum in the next lecture

---

## Comparison Table: All Methods So Far

| Method | LR Type | Gradient Modification | Best For |
|--------|---------|----------------------|----------|
| Vanilla GD | Fixed | None | Small, smooth problems |
| SGD | Fixed | Single random point | Large datasets |
| Momentum | Fixed | Velocity accumulation | Flat regions, inertia |
| RMSProp | **Adaptive** | Magnitude normalization | Uneven gradients, stability |

---

## Next: Adam Optimizer

The final lecture covers **Adam (Adaptive Moment Estimation)**, which combines:
- ✅ Momentum (velocity/direction)
- ✅ RMSProp (adaptive learning rate)

→ Best of both worlds!
