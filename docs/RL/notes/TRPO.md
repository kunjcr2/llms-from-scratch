# Trust Region Policy Optimization (TRPO)

This document covers **Trust Region Policy Optimization (TRPO)**, a foundational algorithm that introduced the concept of **trust regions** for stable policy updates. TRPO (2015) laid the groundwork for modern algorithms like **PPO** and **GRPO** (used in DeepSeek).

---

## Prerequisites: Advantage Function

The **advantage function** measures how much better or worse an action is compared to the policy's default behavior:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

Where:
- $Q^\pi(s, a)$ is the action-value function (expected return from taking action $a$ in state $s$)
- $V^\pi(s)$ is the state-value function (expected return from state $s$ following $\pi$)

### Intuition

Consider state $s$ with possible actions having these expected values:

| Action | Expected Return | Advantage |
|--------|-----------------|-----------|
| $a_0$ (current policy) | 50 | 0 |
| $a_1$ | 30 | -20 |
| $a_2$ | 40 | -10 |
| $a_3$ | 60 | **+10** |

A **positive advantage** indicates room for improvement — action $a_3$ is better than the current policy's choice.

---

## The Core Question

Given two policies $\pi$ (old) and $\pi'$ (new), can we express the new policy's performance in terms of the old policy's performance?

$$\eta(\pi') = \eta(\pi) + \text{(some difference term)}$$

Where $\eta(\pi) = V^\pi(s_0)$ is the expected cumulative reward from the initial state.

### Intuition

If the new policy has positive advantage at every state, it **must** be better than the old policy. The performance difference should be related to the **sum of advantages** across all states.

---

## Deriving the Performance Difference

### Step 1: Express Performance as Value Function

$$\eta(\pi') = \mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

### Step 2: Add and Subtract Baseline

By adding and subtracting $V^\pi(s_0)$, we can show:

$$\eta(\pi') - \eta(\pi) = \mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty} \gamma^t r_t - V^\pi(s_0)\right]$$

### Step 3: Telescoping Sum

Using the advantage function and a telescoping sum technique, this simplifies to:

$$\boxed{\eta(\pi') = \eta(\pi) + \mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right]}$$

> **Key Result**: The new policy's performance equals the old policy's performance plus the expected sum of discounted advantages.

---

## The State Distribution Problem

The formula above requires knowing which states the agent visits under $\pi'$ (the new policy). But we only know the state distribution under $\pi$ (the old policy)!

### Visualization

```
Old Policy (known):      s₀ → s₁ → s₂ → s₃ → ...
New Policy (unknown):    s₀ → s₁' → s₂' → s₃' → ...
```

We can compute the old trajectory but don't know the new trajectory's state distribution.

### Solution: Substitute Old Distribution

Approximate by using the old policy's state distribution:

$$\tilde{L}_\pi(\pi') = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \pi'(a|s) A^\pi(s, a)$$

Where $\rho_\pi(s)$ is the **state visitation frequency** under policy $\pi$.

> **Approximation**: States are visited according to the old policy, but actions are taken according to the new policy.

---

## Constructing a Surrogate Objective

### The Lower Bound Theorem

The true performance has a guaranteed lower bound:

$$\eta(\pi') \geq L_\pi(\pi') - C \cdot D_{KL}^{max}(\pi || \pi')$$

Where:
- $L_\pi(\pi')$ is the surrogate objective using the old state distribution
- $C$ is a constant coefficient
- $D_{KL}^{max}$ is the maximum KL divergence between policies

### Why This Matters

We can't maximize $\eta(\pi')$ directly (unknown state distribution), but we **can** maximize this lower bound!

---

## Surrogate Objective Visualization

Consider optimizing an unknown function $f(\theta)$:

```
                    Peak
                     *
                    / \
       True f(θ)   /   \
                  /     \
                 /       \
Surrogate g(θ)  *---*---*
                   ↑
              Current θ
```

**Strategy**:
1. Construct a surrogate function $g(\theta)$ that is always below $f(\theta)$
2. Both functions touch at the current parameter value
3. Maximize the surrogate to move toward better parameters
4. Repeat

This **guaranteed improvement** approach avoids overshooting and instability.

---

## The Trust Region Constraint

### From Penalty to Constraint

Instead of maximizing:

$$L_\pi(\pi') - C \cdot D_{KL}(\pi || \pi')$$

We reformulate as a **constrained optimization problem**:

$$\max_{\pi'} L_\pi(\pi') \quad \text{subject to} \quad D_{KL}^{max}(\pi || \pi') \leq \delta$$

Where $\delta$ is the **trust region** size.

### Why "Trust Region"?

The constraint creates a bounded region around the current policy where updates are "trusted" to improve performance:

```
                 Policy Space
        ┌─────────────────────────────┐
        │                             │
        │     ┌─────────┐             │
        │     │  Trust  │             │
        │     │ Region  │             │
        │     │  π ──→  │ ← π' must   │
        │     │    δ    │   stay here │
        │     └─────────┘             │
        │                             │
        └─────────────────────────────┘
```

---

## TRPO vs Vanilla Policy Gradient

| Aspect | Vanilla Policy Gradient | TRPO |
|--------|------------------------|------|
| Update Size | Unconstrained | Bounded by trust region |
| Stability | Can diverge | Guaranteed improvement |
| Step Size | Fixed learning rate | Adaptive (within δ) |
| Variance | High | Lower |

### Why Constrained Updates Matter for LLMs

For **LLM alignment** (RLHF), the base model is already very good. Unconstrained updates could:
- Cause catastrophic forgetting
- Lead to mode collapse
- Break capabilities on many prompts

The trust region ensures the aligned model stays close to the base model while improving on the reward signal.

---

## TRPO Algorithm

```
Initialize policy parameters θ₀

For each iteration:
    1. Collect trajectories using current policy πθ
    
    2. Estimate advantages Â(s, a) using GAE or similar
    
    3. Compute surrogate objective:
       L(θ) = E[π(a|s, θ) / π(a|s, θ_old) · Â(s, a)]
    
    4. Solve constrained optimization:
       θ_new = argmax_θ L(θ)
       subject to: D_KL(π_θ_old || π_θ) ≤ δ
    
    5. Update: θ ← θ_new
```

---

## Key Equations Summary

### Performance Difference Identity

$$\eta(\pi') = \eta(\pi) + \mathbb{E}_{\pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right]$$

### Surrogate Objective

$$L_\pi(\pi') = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \pi'(a|s) A^\pi(s, a)$$

### Lower Bound (Minorization)

$$\eta(\pi') \geq L_\pi(\pi') - C \cdot D_{KL}^{max}(\pi || \pi')$$

### Trust Region Constraint

$$D_{KL}^{max}(\pi_{old} || \pi_{new}) \leq \delta$$

---

## Historical Context

| Year | Development |
|------|-------------|
| 2002 | Kakade introduces surrogate objective concepts |
| 2015 | **TRPO** formalizes trust region for policy optimization |
| 2017 | PPO simplifies TRPO with clipping |
| 2024 | GRPO (DeepSeek) extends these ideas for LLMs |

---

## Key Takeaways

1. **Performance Difference = Sum of Advantages**: The improvement of a new policy over the old is the expected sum of discounted advantages

2. **Surrogate Objective**: When we can't compute the true objective, construct a lower bound that we *can* optimize

3. **Trust Region = Stability**: Bounding KL divergence prevents destructive large updates

4. **Foundation for Modern RL**: TRPO's ideas directly underpin PPO and GRPO

5. **Critical for LLM Alignment**: Trust regions ensure fine-tuned models don't deviate too far from capable base models

---

## Connection to Other Topics

- **Advantage Function**: See [AdvantageFunction.md](./AdvantageFunction.md) for GAE and estimation methods
- **REINFORCE**: See [REINFORCE.md](./REINFORCE.md) for the foundational policy gradient update
- **PPO**: The practical successor that approximates TRPO's constraint with clipping (next topic)
