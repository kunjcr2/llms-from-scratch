# Momentum Gradient Descent

> Part 2 of 4 in the optimization series: SGD → **Momentum** → RMSProp → Adam

---

## Recap: Vanilla Gradient Descent

**Gradient Descent** minimizes the loss function $J(\theta)$ where $\theta$ represents model parameters.

### Algorithm Steps
1. **Initialize** parameters $\theta$ with random values
2. **Calculate gradient**: $\nabla_\theta J(\theta)$
3. **Update parameters**: $\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta J(\theta)$
4. **Repeat** until convergence

### Problems with Vanilla GD
| Issue | Description |
|-------|-------------|
| **Computational cost** | Uses all $n$ data points for each gradient calculation |
| **Slow in flat regions** | Small gradients = tiny parameter updates |
| **Oscillations** | High learning rates cause divergence in steep regions |
| **Stuck at saddle points** | Gradient ≈ 0 halts progress |

---

## Momentum: The Physics-Inspired Solution

### Core Idea
> Add an **inertia term** to gradient descent, inspired by physical momentum.

Think of a ball rolling down a hill:
- It **accelerates** on steep slopes
- It **maintains velocity** through flat regions
- It **resists sudden direction changes**

### Goals of Momentum
1. **Dampen oscillations** in steep regions
2. **Maintain direction** in flat regions
3. **Escape local minima** through accumulated velocity

---

## Mathematical Formulation

### Velocity Update
$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \nabla_\theta J(\theta)$$

Where:
- $v_t$ = current velocity
- $v_{t-1}$ = previous velocity
- $\beta$ = momentum coefficient (typically 0.9 or 0.99)
- $\nabla_\theta J(\theta)$ = current gradient

### Parameter Update
$$\theta_{new} = \theta_{old} - \eta \cdot v_t$$

### Interpretation
| Component | Contribution | Purpose |
|-----------|--------------|---------|
| $\beta \cdot v_{t-1}$ | ~90-99% of update | Carries forward previous direction |
| $(1-\beta) \cdot \nabla$ | ~1-10% of update | Adjusts based on current gradient |

---

## Why Momentum Works

### 1. Flat Regions (Gradient ≈ 0)

**Vanilla GD Problem:**
```
If gradient = 0 → θ update = 0 → STUCK!
```

**Momentum Solution:**
```
v_t = 0.99 × v_{t-1} + 0.01 × 0
    = 0.99 × v_{t-1}  ← Still has velocity!
```
Even with zero gradient, the accumulated velocity keeps parameters moving.

### 2. Steep Regions (Accelerating Descent)

**Physical Analogy:** Ball rolling downhill gains speed

| Time Step | Velocity Accumulation |
|-----------|----------------------|
| t=1 | $v_1 = 0.9 \times 0 + 0.1 \times \nabla_1$ |
| t=2 | $v_2 = 0.9 \times v_1 + 0.1 \times \nabla_2$ |
| t=3 | $v_3 = 0.9 \times v_2 + 0.1 \times \nabla_3$ |

Velocity keeps building → Faster parameter updates → Quicker convergence!

**Vanilla GD (Contrast):**
- Each step only uses current gradient
- Updates get smaller as slope decreases
- Like a ball that slows down going downhill (unphysical!)

### 3. Preventing Oscillations

**Vanilla GD with High Learning Rate:**
```
Position: A → B → C → D (diverging oscillations)
          ↘  ↗  ↘  ↗
            Loss increases!
```

**Momentum GD:**
```
Even if gradient flips direction:
v_t = 0.99 × (previous direction) + 0.01 × (flipped gradient)
    ≈ Previous direction (mostly unchanged)
```
Inertia resists sudden direction changes → Smooth convergence

---

## Comparison: Vanilla GD vs Momentum

### Contour Plot Visualization

For loss function $J = x^2 + 10y^2$ (elliptical contours):

| Method | Path Behavior |
|--------|--------------|
| **Vanilla GD** | Zigzag path, can diverge with high LR |
| **Momentum GD** | Smooth curve rolling toward minimum |

```
     Vanilla GD              Momentum GD
         ↓                       ↓
    ─────────────           ─────────────
   ╱ ╲ ╱ ╲ ╱ ╲ ╱           ╱    ╲    ╱
  │   ╳   ╳   │           │  ╲    ╲   │
  │  ╱ ╲ ╱ ╲  │           │   ╲    ╲  │
   ╲    ╳    ╱             ╲   •~~~~• ╱
    ─────────────           ─────────────
   (oscillating)            (smooth curve)
```

---

## Experimental Results

### Test Case: $J = x^2 + 10y^2$

Starting point: $(1.5, 1.5)$  
Target minimum: $(0, 0)$

### Low Learning Rate ($\eta = 0.01$)

| Method | Convergence | Final Loss |
|--------|-------------|------------|
| Vanilla GD | Slow but stable | Low |
| Momentum GD | Fast, smooth | ≈ 0 |

### High Learning Rate ($\eta = 0.2$)

| Method | Behavior | Final Loss |
|--------|----------|------------|
| Vanilla GD | **DIVERGES!** | $10^{49}$ 💥 |
| Momentum GD | Converges smoothly | ≈ 0.016 ✓ |

> **Key Insight:** Momentum enables higher learning rates without divergence!

---

## Implementation

### Vanilla Gradient Descent
```python
def batch_gradient_descent(grad_fn, eta, epochs, start):
    x, y = start
    for _ in range(epochs):
        grad = grad_fn(x, y)
        x -= eta * grad[0]  # Update with gradient directly
        y -= eta * grad[1]
    return x, y
```

### Momentum Gradient Descent
```python
def momentum_gradient_descent(grad_fn, eta, beta, epochs, start):
    x, y = start
    v = np.array([0.0, 0.0])  # Initialize velocity to zero
    
    for _ in range(epochs):
        grad = grad_fn(x, y)
        v = beta * v + (1 - beta) * grad  # Accumulate velocity
        x -= eta * v[0]  # Update with velocity (not gradient!)
        y -= eta * v[1]
    return x, y
```

### Key Difference
```diff
- x -= eta * grad[0]        # Vanilla: use gradient directly
+ v = beta * v + (1-beta) * grad
+ x -= eta * v[0]           # Momentum: use accumulated velocity
```

---

## Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| $\eta$ (learning rate) | 0.01 - 0.1 | Step size |
| $\beta$ (momentum) | 0.9 - 0.99 | How much past velocity is retained |
| Epochs | 50 - 1000 | Number of iterations |

### Beta ($\beta$) Effects
- **Higher β (0.99)**: More inertia, smoother but slower to change direction
- **Lower β (0.9)**: Less inertia, more responsive to current gradient

---

## Physical Intuition Summary

| Physics Concept | GD Analog |
|-----------------|-----------|
| Ball | Parameter values $\theta$ |
| Hill surface | Loss function $J(\theta)$ |
| Velocity | Accumulated gradient $v$ |
| Gravity | Negative gradient direction |
| Friction | Decay factor $(1-\beta)$ |

---

## Key Takeaways

1. **Momentum mimics physics** - Parameters move like a ball rolling down a loss landscape
2. **Accumulates velocity** - 90-99% of update comes from previous direction
3. **Solves flat regions** - Maintains speed even when gradient ≈ 0
4. **Prevents divergence** - Inertia resists oscillations from high learning rates
5. **Enables faster training** - Can use higher learning rates safely
6. **Uses all data points** - Unlike SGD, considers full batch (can combine with SGD too!)

---

## Comparison Table

| Aspect | Vanilla GD | Momentum GD |
|--------|-----------|-------------|
| Update rule | $\theta - \eta \nabla J$ | $\theta - \eta v$ |
| Flat regions | Gets stuck | Maintains velocity |
| High LR | Diverges/oscillates | Stays stable |
| Steep regions | Constant speed | Accelerates |
| Convergence | Can be slow | Usually faster |

---

## Next: RMSProp

The next lecture covers **RMSProp (Root Mean Square Propagation)**, which addresses a different problem: adapting learning rates per-parameter based on gradient magnitudes.
