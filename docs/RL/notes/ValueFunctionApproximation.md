# Value Function Approximation

In classical RL (Dynamic Programming, Monte Carlo, Temporal Difference), we use **tabular methods**—storing a value for each state in a table. This works for small state spaces but becomes impractical for real-world problems.

---

## The Problem with Tabular Methods

| Aspect | Description |
|--------|-------------|
| **Memory** | Must store values for every possible state |
| **Computation** | Must update and visit every state |
| **Scale** | Chess has ~$10^{46}$ states—impossible to enumerate |

**Solution**: Instead of storing values in a table, we **approximate** the value function using a parameterized function that can generalize from visited states to unvisited ones.

---

## The Core Idea

In tabular methods, we write $V(s)$—a lookup table.

In function approximation, we write $\hat{V}(s, \mathbf{w})$—a function parameterized by weights $\mathbf{w}$.

| Notation | Meaning |
|----------|---------|
| $\hat{V}$ | Approximate value function (hat indicates approximation) |
| $s$ | State |
| $\mathbf{w}$ | Weight vector (learnable parameters) |

**Goal**: Find weights $\mathbf{w}$ such that $\hat{V}(s, \mathbf{w}) \approx V_\pi(s)$ for all states.

---

## The Objective: Mean Squared Value Error

We want to minimize the error between our approximation and the true value function:

$$\overline{VE}(\mathbf{w}) = \sum_s \mu(s) \left[ V_\pi(s) - \hat{V}(s, \mathbf{w}) \right]^2$$

Where:
- $V_\pi(s)$ = true value function
- $\hat{V}(s, \mathbf{w})$ = our approximation
- $\mu(s)$ = **state distribution** (how often state $s$ is visited)

### Why Weight by $\mu(s)$?

Not all states are equally important. If the agent visits state A 100 times but state B only once, errors at state A matter more.

```
State Distribution Example:

Probability
    │     ●
    │    ╱ ╲
    │   ╱   ╲
    │  ╱     ╲
    │ ╱       ╲
    └──────────────► State
       -1   0   1

State 0 is visited most frequently → prioritize accurate estimates here
```

By multiplying by $\mu(s)$, we focus the optimization on frequently-visited states.

---

## Gradient Descent: Finding Optimal Weights

To minimize $\overline{VE}(\mathbf{w})$, we use **gradient descent**:

### Intuition

Imagine standing on a hillside and wanting to reach the valley (minimum). Gradient descent says:
1. Look around to find the steepest downhill direction
2. Take a small step in that direction
3. Repeat until you reach the bottom

### The Update Rule

Starting from the objective function and applying gradient descent:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

Where:
- $\alpha$ = learning rate (step size)
- $V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}_t)$ = error (how far off we are)
- $\nabla \hat{V}(S_t, \mathbf{w}_t)$ = gradient (direction to move)

**Intuition**: Move the weights in the direction that reduces the error, scaled by how large the error is.

---

## Problem 1: The Target is Unknown

In supervised learning, we have labeled data: $(x, y)$ pairs where $y$ is known.

In RL, we don't know $V_\pi(s)$—that's precisely what we're trying to learn!

**Solution**: Replace the true value with an **approximation target**.

### Monte Carlo Target

Use the actual return $G_t$ from experience:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ G_t - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

This works because $\mathbb{E}[G_t] = V_\pi(S_t)$—the expected return equals the true value.

### TD Target

Use the TD estimate $R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w})$:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w}_t) - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

---

## Problem 2: Semi-Gradient Methods

When using the TD target, there's a mathematical subtlety.

The gradient of the squared error should include the gradient of both terms:

$$\nabla \left[ V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}) \right]^2$$

With the TD target, $\hat{V}(S_{t+1}, \mathbf{w})$ also depends on $\mathbf{w}$, so the full gradient would be:

$$\nabla \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w}) - \hat{V}(S_t, \mathbf{w}) \right]^2$$

This requires computing $\nabla \hat{V}(S_{t+1}, \mathbf{w})$ as well.

**In practice**: We ignore the gradient through the target (treat it as a constant).

This gives us **semi-gradient methods**—"semi" because we only compute part of the true gradient.

---

## Why RL is Different from Supervised Learning

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|---------------------|------------------------|
| **Target** | Known beforehand $(x, y)$ pairs | Unknown until interaction |
| **Target stability** | Fixed | Can change as learning progresses |
| **Data** | IID samples | Correlated sequential data |

These differences make RL harder but still tractable with the right approximations.

---

## Summary

1. **Tabular methods don't scale**: Real problems have enormous state spaces ($10^{46}$ for chess)

2. **Function approximation**: Replace $V(s)$ lookup table with parameterized $\hat{V}(s, \mathbf{w})$

3. **Objective**: Minimize weighted mean squared error $\overline{VE}(\mathbf{w})$

4. **State distribution $\mu(s)$**: Focus learning on frequently-visited states

5. **Gradient descent**: Update weights in direction that reduces error

6. **Target approximations**: Use Monte Carlo returns ($G_t$) or TD targets ($R + \gamma \hat{V}$) since true values unknown

7. **Semi-gradient methods**: Ignore gradient through TD target for practical computation

> **Next**: What functions can represent $\hat{V}(s, \mathbf{w})$? Linear function approximation, neural networks, and more.
