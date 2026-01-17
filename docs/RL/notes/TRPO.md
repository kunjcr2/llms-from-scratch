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

## Part 2: Solving the Constraint Optimization Problem

Now that we have our constrained optimization problem:

$$\max_{\pi'} L_\pi(\pi') \quad \text{subject to} \quad D_{KL}^{max}(\pi || \pi') \leq \delta$$

The question is: **How do we actually solve this?**

---

## Simplifying the Objective Function

### Removing Constants

The surrogate objective is:

$$L_\pi(\pi') = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \pi'(a|s) A^\pi(s, a)$$

Since $\eta(\pi)$ is **constant** (it's the known performance of the old policy), we only need to maximize:

$$\sum_s \rho_\pi(s) \sum_a \pi'(a|s) A^\pi(s, a)$$

> **Intuition**: Find a new policy $\pi'$ that, on average, prefers actions with higher advantages. This makes sense because higher advantages mean more room for improvement!

---

## Rewriting as Expectations

### Expected Value Review

For a random variable $X$ with probability $P(X)$ and function $f(X)$:

$$\mathbb{E}[f(X)] = \sum_x P(x) \cdot f(x)$$

**Example**: Expected value of a dice roll:
$$\mathbb{E}[X] = \frac{1}{6}(1) + \frac{1}{6}(2) + \frac{1}{6}(3) + \frac{1}{6}(4) + \frac{1}{6}(5) + \frac{1}{6}(6) = 3.5$$

### Applying to Our Objective

The state visitation frequency $\rho_\pi(s)$ is a probability distribution over states. So:

$$\sum_s \rho_\pi(s) \cdot Y(s) = \mathbb{E}_{s \sim \rho_\pi}[Y(s)]$$

Where $Y(s) = \sum_a \pi'(a|s) A^\pi(s, a)$

Our objective becomes:

$$\mathbb{E}_{s \sim \rho_\pi}\left[\sum_a \pi'(a|s) A^\pi(s, a)\right]$$

---

## The Advantage Decomposition

Recall that the advantage can be written as:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

Since $V^\pi(s)$ doesn't depend on the action (it's a constant for each state), and $\sum_a \pi'(a|s) = 1$, we have:

$$\sum_a \pi'(a|s) V^\pi(s) = V^\pi(s) \quad \text{(constant)}$$

So we only need to maximize:

$$\mathbb{E}_{s \sim \rho_\pi}\left[\sum_a \pi'(a|s) Q^\pi(s, a)\right]$$

---

## Important Sampling

### The Problem

We want to compute expectations over actions from $\pi'$ (new policy), but we can only **sample actions from $\pi$** (old policy)!

### The Solution: Importance Sampling

If we can't sample from distribution $P$, but can sample from distribution $Q$:

$$\mathbb{E}_{x \sim P}[f(x)] = \mathbb{E}_{x \sim Q}\left[\frac{P(x)}{Q(x)} \cdot f(x)\right]$$

**Intuition**: Re-weight samples by how much more/less likely they are under the target distribution.

### Visual Example

```
Distribution P (target, mean ≈ 3):     Distribution Q (accessible, mean ≈ 6):

    ╭──╮                                             ╭──╮
   ╱    ╲                                           ╱    ╲
  ╱      ╲                                         ╱      ╲
 ╱        ╲                                       ╱        ╲
╱          ╲                                     ╱          ╲
──────────────                              ──────────────
  0  3  6  9                                  0  3  6  9
```

By multiplying samples from Q by $\frac{P(x)}{Q(x)}$:
- Samples where P > Q get **upweighted**
- Samples where P < Q get **downweighted**
- Result: Correct expected value despite sampling from wrong distribution!

### Applying to Our Objective

We want: $\mathbb{E}_{a \sim \pi'}[Q^\pi(s, a)]$

But we can only sample from $\pi$ (old policy). Using importance sampling:

$$\mathbb{E}_{a \sim \pi}\left[\frac{\pi'(a|s)}{\pi(a|s)} \cdot Q^\pi(s, a)\right]$$

---

## The Final Objective Function

Combining everything, our objective becomes:

$$\boxed{\max_{\theta} \mathbb{E}_{s \sim \rho_{\pi_{old}}, a \sim \pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \cdot Q^{\pi_{old}}(s, a)\right]}$$

Or equivalently using advantages:

$$\max_{\theta} \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \cdot A^{\pi_{old}}(s, a)\right]$$

> **Note**: Q-values and advantages are interchangeable here since they differ by a constant (the value function).

### Connection to Vanilla Policy Gradient

The gradient of this objective looks remarkably similar to the vanilla policy gradient!

| Vanilla Policy Gradient | TRPO Objective Gradient |
|------------------------|------------------------|
| $\mathbb{E}\left[\frac{\nabla \pi_\theta}{\pi_\theta} Q^\pi(s,a)\right]$ | $\mathbb{E}\left[\frac{\nabla \pi_\theta}{\pi_{old}} Q^{\pi_{old}}(s,a)\right]$ |

The key difference: TRPO explicitly uses $\pi_{old}$ and $\pi_{new}$ (iteration), while vanilla PG has a single $\pi$.

---

## Taylor Series Approximation

Since $\pi_\theta$ stays close to $\pi_{old}$ (within the trust region), we can use **Taylor series** to approximate both the objective and constraint.

### Taylor Series Review

For a function $f(\theta)$ near point $\theta_0$:

$$f(\theta) \approx f(\theta_0) + (\theta - \theta_0)^T \nabla f(\theta_0) + \frac{1}{2}(\theta - \theta_0)^T H (\theta - \theta_0)$$

**Example**: Approximating $\sin(x)$ near $x = 0$:
- $\sin(0) = 0$
- $\cos(0) = 1$ (first derivative at 0)
- $\sin(x) \approx x$ (first-order approximation)

```
        True sin(x)     First-order approximation (y = x)
              ╱╲                    ╱
             ╱  ╲                  ╱
            ╱    ╲                ╱
           ╱      ╲              ╱
──────────●────────          ───●───────
          0                     0
```

Works great near 0, breaks down as x increases!

### Approximating the Objective

Let $f(\theta)$ be our objective. Using first-order Taylor expansion around $\theta_{old}$:

$$f(\theta) \approx f(\theta_{old}) + (\theta - \theta_{old})^T \cdot g$$

Where $g = \nabla f(\theta_{old})$ is the **policy gradient** at the old parameters.

Since $f(\theta_{old})$ is constant, we maximize:

$$(\theta - \theta_{old})^T \cdot g$$

### Approximating the Constraint

The KL divergence constraint uses a **second-order** (quadratic) approximation:

$$D_{KL}(\pi_{old} || \pi_\theta) \approx \frac{1}{2}(\theta - \theta_{old})^T F (\theta - \theta_{old})$$

Where:
- $F$ is the **Fisher Information Matrix**: $F = \nabla^2 D_{KL}|_{\theta_{old}}$
- First-order term is zero (KL divergence is zero at $\theta = \theta_{old}$)

---

## The Final Optimization Problem

After Taylor approximation, TRPO becomes:

$$\max_\theta \quad g^T(\theta - \theta_{old})$$

$$\text{subject to} \quad \frac{1}{2}(\theta - \theta_{old})^T F (\theta - \theta_{old}) \leq \delta$$

### Closed-Form Solution

This constrained quadratic problem has an analytical solution:

$$\boxed{\theta_{new} = \theta_{old} + \sqrt{\frac{2\delta}{g^T F^{-1} g}} \cdot F^{-1} g}$$

Or with step size factor $\alpha$:

$$\theta_{new} = \theta_{old} + \alpha \cdot F^{-1} g$$

---

## The Fisher Matrix Challenge

### The Problem

Computing $F^{-1}$ (the inverse of the Fisher Information Matrix) is **very expensive**:
- For a neural network with $n$ parameters, $F$ is an $n \times n$ matrix
- Storing and inverting this is often infeasible for large networks

### The Solution: Conjugate Gradient Method

Instead of computing $F^{-1}$ directly, TRPO uses the **Conjugate Gradient (CG)** algorithm to compute $F^{-1}g$ iteratively without forming $F$ explicitly.

> **This is a major computational bottleneck of TRPO** — and one of the main reasons PPO was developed as a simpler alternative!

---

## TRPO Summary: Four Steps

### Step 1: Surrogate Objective

Construct the surrogate objective that:
- Is always ≤ true objective
- Matches true objective at current policy

$$L_\pi(\pi') - C \cdot D_{KL}^{max}(\pi || \pi')$$

### Step 2: Trust Region Reformulation

Convert the penalty term into a constraint:

$$\max_{\pi'} L_\pi(\pi') \quad \text{s.t.} \quad D_{KL}^{max}(\pi || \pi') \leq \delta$$

### Step 3: Simplify with Importance Sampling

Apply probability manipulations to get:

$$\mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s, a)\right]$$

This connects to vanilla policy gradient!

### Step 4: Solve with Taylor Approximation

Use Taylor series to get a tractable optimization:

$$\max_\theta \quad g^T(\theta - \theta_{old}) \quad \text{s.t.} \quad \frac{1}{2}(\theta - \theta_{old})^T F (\theta - \theta_{old}) \leq \delta$$

Solution: $\theta_{new} = \theta_{old} + \alpha \cdot F^{-1}g$

---

## TRPO: Strengths and Limitations

### Strengths

| Strength | Description |
|----------|-------------|
| **Guaranteed Improvement** | Each update provably improves or maintains performance |
| **Stable Training** | Trust region prevents catastrophic policy changes |
| **Theoretical Foundation** | Strong mathematical guarantees |

### Limitations

| Limitation | Description |
|------------|-------------|
| **Computational Cost** | Fisher matrix inversion via CG is expensive |
| **Implementation Complexity** | Conjugate gradient + line search adds complexity |
| **Hard Constraint** | KL constraint can be too restrictive or too loose |

> **This led to PPO (2017)**: Replaces the hard KL constraint with a simple clipping mechanism, achieving similar stability with much simpler implementation.

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
