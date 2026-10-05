# REINFORCE Algorithm

This document covers the **REINFORCE** algorithm, our first policy gradient algorithm, along with its variants: **REINFORCE with Baseline** and **Actor-Critic Methods**.

---

## Gradient Ascent for Policies

The foundation of REINFORCE is gradient ascent on the policy parameters. Starting from a non-optimal policy (point A), we iteratively move toward a better policy (point B) by following the gradient direction:

$$\theta_{t+1} = \theta_t + \alpha \cdot \nabla J(\theta)$$

Where:
- $\theta_t$ is the policy parameter at time step $t$
- $\alpha$ is the learning rate
- $\nabla J(\theta)$ is the gradient of the performance measure

From the **Policy Gradient Theorem**, we know:

$$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a Q^\pi(s, a) \nabla \pi(a|s, \theta)$$

---

## Deriving the Update Rule

### Expected Value Formulation

The summation over states can be converted to an expected value using probability concepts.

If we have a random variable $X$ with probability $P(X)$, the expected value of a function $f(X)$ is:

$$\mathbb{E}[f(X)] = \sum_x P(x) \cdot f(x)$$

Since $\mu(s)$ represents the probability of being in state $s$ (state distribution), we can write:

$$\sum_s \mu(s) \cdot Y(s) = \mathbb{E}[Y(s)]$$

This allows us to rewrite the gradient as:

$$\nabla J(\theta) \propto \mathbb{E}_s\left[\sum_a Q^\pi(s, a) \nabla \pi(a|s, \theta)\right]$$

### Absorbing the Action Summation

To eliminate the summation over actions, we use a **multiply-and-divide trick**:

$$\sum_a Q^\pi(s, a) \nabla \pi(a|s, \theta) = \sum_a \pi(a|s, \theta) \cdot Q^\pi(s, a) \cdot \frac{\nabla \pi(a|s, \theta)}{\pi(a|s, \theta)}$$

Now, since $\pi(a|s, \theta)$ is the probability of selecting action $a$, this becomes:

$$\mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a) \cdot \frac{\nabla \pi(a|s, \theta)}{\pi(a|s, \theta)}\right]$$

### Final Form

Since $Q^\pi(s, a)$ is the expected return starting from state $s$ and taking action $a$, we can replace it with the actual Monte Carlo return $G_t$:

$$\boxed{\theta_{t+1} = \theta_t + \alpha \cdot G_t \cdot \frac{\nabla \pi(a_t|s_t, \theta)}{\pi(a_t|s_t, \theta)}}$$

> The term $\frac{\nabla \pi}{\pi}$ appears frequently in policy gradient methods, including PPO and GRPO. Whenever you see this term, recognize it as stemming from the policy gradient theorem derivation.

---

## Intuition Behind REINFORCE

Consider an agent in state $S$ taking action $A_t$ and receiving total return $G_t$. The update moves the policy parameters in a specific direction:

### The Gradient Direction ($\nabla \pi / \pi$)

This vector points in the direction that **maximizes the probability of selecting action $A_t$ in state $S$**.

Among all possible directions from the current policy point $O$, moving along $\nabla \pi / \pi$ specifically increases the likelihood of choosing the action we just took.

### The Return Multiplier ($G_t$)

- **High $G_t$**: Move more in the gradient direction (reinforce good actions)
- **Low $G_t$**: Move less (don't reinforce poor actions)

### The Division by $\pi$ (Normalization)

Without division by $\pi$, frequently visited states would dominate the updates simply because the agent visits them more often, not because they're more valuable.

Dividing by $\pi$ **penalizes over-visited state-action pairs**, ensuring updates reflect actual value rather than visit frequency.

---

## REINFORCE Algorithm Summary

```
Initialize policy parameters θ
For each episode:
    Generate trajectory: s₀, a₀, r₁, s₁, a₁, r₂, ...
    For each step t of the episode:
        Calculate return Gₜ = Σᵢ γⁱ rₜ₊ᵢ₊₁
        Update: θ ← θ + α · Gₜ · ∇log π(aₜ|sₜ, θ)
```

> **Key Point**: We must wait until the end of the episode to calculate $G_t$ before updating policy parameters.

---

## REINFORCE with Baseline

### The Variance Problem

REINFORCE suffers from **high variance** because the return $G_t$ can vary wildly between episodes, leading to unstable gradient estimates.

### Solution: Subtract a Baseline

We modify the update rule by subtracting a baseline $b(s)$ from the return:

$$\theta_{t+1} = \theta_t + \alpha \cdot (G_t - b(s)) \cdot \frac{\nabla \pi(a_t|s_t, \theta)}{\pi(a_t|s_t, \theta)}$$

### Why Does This Work?

The baseline must be a **function of states only** (not actions). This is crucial because:
- If $b(s)$ depends on actions, it would interfere with the gradient calculation
- State-only dependence allows it to be "factored out" without affecting the expected gradient

### Common Choice: Value Function Estimate

The natural choice for baseline is the **approximate value function**:

$$b(s) = \hat{V}(s, w)$$

This represents the expected return from state $s$, making $G_t - \hat{V}(s)$ the **advantage** of the action taken over the average.

---

## Actor-Critic Methods

### From Monte Carlo to Temporal Difference

The problem with both REINFORCE variants: we must wait until episode end to compute $G_t$.

This is the same limitation we encountered moving from Monte Carlo to TD learning!

### The TD Target Replacement

Replace the Monte Carlo return $G_t$ with the **TD target**:

$$G_t \approx r_{t+1} + \gamma \hat{V}(s_{t+1}, w)$$

### Actor-Critic Update Rule

$$\theta_{t+1} = \theta_t + \alpha \cdot \underbrace{(r_{t+1} + \gamma \hat{V}(s_{t+1}, w) - \hat{V}(s_t, w))}_{\text{TD Error } \delta} \cdot \frac{\nabla \pi(a_t|s_t, \theta)}{\pi(a_t|s_t, \theta)}$$

### Why "Actor-Critic"?

| Component | Role | Formula |
|-----------|------|---------|
| **Actor** | Takes actions, receives rewards, moves to next state | $r_{t+1} + \gamma \hat{V}(s_{t+1})$ |
| **Critic** | Evaluates the current state | $\hat{V}(s_t)$ |

The actor proposes actions while the critic evaluates whether those actions improved upon the expected value.

---

## Comparison of Methods

| Method | Target | Update Timing | Variance | Bias |
|--------|--------|---------------|----------|------|
| REINFORCE | $G_t$ | End of episode | High | None |
| REINFORCE + Baseline | $G_t - \hat{V}(s)$ | End of episode | Lower | None |
| Actor-Critic | $r + \gamma\hat{V}(s') - \hat{V}(s)$ | Every step | Lowest | Some |

---

## Key Takeaways

1. **Policy Gradient Core**: The gradient direction $\nabla \pi / \pi$ maximizes the probability of selecting the action taken

2. **Return Weighting**: Multiply by $G_t$ to reinforce actions proportional to their rewards

3. **Baseline for Variance Reduction**: Subtracting a state-dependent baseline (typically $\hat{V}(s)$) reduces variance without introducing bias

4. **TD for Online Updates**: Replacing Monte Carlo returns with TD targets enables step-by-step updates (Actor-Critic)

5. **Foundation for Modern RL**: These concepts directly extend to TRPO, PPO, and GRPO algorithms

---

## Connection to Modern Algorithms

The $\frac{\nabla \pi}{\pi}$ term appears in the objective functions of:
- **TRPO** (Trust Region Policy Optimization)
- **PPO** (Proximal Policy Optimization)
- **GRPO** (Group Relative Policy Optimization)

Understanding REINFORCE provides the mathematical foundation for these advanced algorithms.
