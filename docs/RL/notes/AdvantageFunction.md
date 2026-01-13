# Generalized Advantage Estimation (GAE)

This document covers the **Advantage Function** and **Generalized Advantage Estimation (GAE)**, which are essential components for modern policy gradient algorithms like PPO and GRPO used in RLHF.

---

## Recap: Policy Gradient Methods

Before diving into GAE, let's recall the key concepts from policy gradient methods:

### Step 1: Parameterizing the Policy

We express the policy in terms of a policy parameter θ:

$$\pi_\theta(a | s)$$

This gives the probability of taking action $a$ while being in state $s$, parameterized by $\theta$.

### Step 2: Defining the Performance Measure

The performance measure we want to maximize is:

$$J(\theta) = V_{\pi_\theta}(s_0)$$

This is the value function for the initial state - the expected sum of all rewards the agent receives from the start of the episode.

### Step 3: Gradient Ascent

To maximize $J(\theta)$, we use gradient ascent:

$$\theta_{t+1} = \theta_t + \alpha \cdot \nabla J(\theta)$$

### Step 4: The Policy Gradient Theorem

The gradient involves two components:
1. **State distribution μ(s)** - how often states are visited
2. **Gradient of the policy** - $∇π(a|s, θ)$

The **Policy Gradient Theorem** tells us that we don't need to worry about how the state distribution changes with θ. The gradient is:

$$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a Q^\pi(s, a) \nabla_\theta \pi(a|s, \theta)$$

---

## Three Methods for Expressing the Gradient

### Method 1: Using Q-values

$$\nabla J(\theta) = \mathbb{E}_\pi \left[ Q_\pi(s, a_t) \cdot \frac{\nabla \pi(a_t|s_t, \theta)}{\pi(a_t|s_t, \theta)} \right]$$

### Method 2: REINFORCE (Using Returns)

Since $Q^\pi(s, a)$ is the expected value of returns, we can use actual Monte Carlo returns:

$$\nabla J(\theta) = \mathbb{E}_\pi \left[ G_t \cdot \frac{\nabla \pi(a_t|s_t, \theta)}{\pi(a_t|s_t, \theta)} \right]$$

This is more practical because we don't need to average over all possible trajectories.

### Method 3: With Baseline

To reduce variance, we subtract a baseline (function of states only):

$$\nabla J(\theta) = \mathbb{E}_\pi \left[ (G_t - b(s)) \cdot \frac{\nabla \pi(a_t|s_t, \theta)}{\pi(a_t|s_t, \theta)} \right]$$

---

## The Advantage Function

### Definition

The advantage function measures whether an action is better or worse compared to the policy's default behavior:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

Where:
- $Q^\pi(s, a)$ = expected cumulative reward after taking action $a$ in state $s$
- $V^\pi(s)$ = expected cumulative reward from state $s$ under the current policy (default behavior)

### Intuition

Consider a state $s_t$ with multiple action options:

| Action | Expected Reward | Advantage |
|--------|-----------------|-----------|
| A1     | 80              | 80 - 50 = **+30** |
| A2 (default) | 50        | 50 - 50 = 0 |
| A5     | 20              | 20 - 50 = **-30** |

- **Positive advantage**: Action is better than the policy's default → increase its probability
- **Negative advantage**: Action is worse than the policy's default → decrease its probability
- **Zero advantage**: Action is same as default → no change needed

### Connection to Policy Improvement

This is the same intuition from **policy iteration** in dynamic programming:
1. Fix a policy, calculate its value function
2. For each state, ask: "Is there any action that leads to better expected rewards than the current value function?"
3. If yes, there's scope for improvement → update the policy
4. If advantages are negative for all alternative actions, the policy is optimal

---

## Gradient with Advantage Function

Using the value function $V^\pi(s)$ as the baseline (which is valid since it only depends on states, not actions):

$$\nabla J(\theta) = \mathbb{E}_\pi \left[ A^\pi(s_t, a_t) \cdot \nabla \log \pi(a_t|s_t, \theta) \right]$$

This is the most common formulation used in modern algorithms.

### Intuition (3D Visualization)

Imagine a 3D surface where:
- X-axis: $\theta_1$
- Y-axis: $\theta_2$
- Z-axis: $\pi(a|s)$ (probability of choosing action $a$ in state $s$)

The gradient $\nabla \log \pi$ gives the direction that maximizes the probability of choosing action $a$.

- **If A > 0**: Move UP the surface → increase probability of this action (it's better than average)
- **If A < 0**: Move DOWN the surface → decrease probability of this action (it's worse than average)

---

## Estimating the Advantage Function

### The Challenge

We need estimates for:
- $Q^\pi(s_t, a_t)$ - action value function
- $V^\pi(s_t)$ - state value function

### One-Step Estimate (TD Target)

Using the temporal difference target as a proxy for Q:

$$Q^\pi(s_t, a_t) \approx r_t + \gamma V^\pi(s_{t+1})$$

Therefore:

$$\hat{A}^{(1)}_t = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$$

This is called a **one-step backup** - we only look one step ahead.

### Two-Step Estimate

Looking two steps ahead:

$$\hat{A}^{(2)}_t = r_t + \gamma r_{t+1} + \gamma^2 V^\pi(s_{t+2}) - V^\pi(s_t)$$

### N-Step Estimate

More generally, for an n-step backup:

$$\hat{A}^{(n)}_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V^\pi(s_{t+n}) - V^\pi(s_t)$$

### Monte Carlo Estimate (Infinite Steps)

When n → ∞ (full episode):

$$\hat{A}^{(\infty)}_t = G_t - V^\pi(s_t)$$

Where $G_t$ is the Monte Carlo return.

---

## Bias-Variance Tradeoff

| Method | Bias | Variance |
|--------|------|----------|
| One-step (TD) | **High** - only uses next state | **Low** - single estimate |
| Monte Carlo | **Low** - uses all actual rewards | **High** - many estimates summed |

---

## Generalized Advantage Estimation (GAE)

### The Key Idea

Instead of choosing ONE n-step return, combine ALL n-step returns with exponentially decaying weights!

### Weighting Scheme

| N-Step Return | Weight |
|---------------|--------|
| 1-step | $(1 - \lambda)$ |
| 2-step | $(1 - \lambda) \cdot \lambda$ |
| 3-step | $(1 - \lambda) \cdot \lambda^2$ |
| n-step | $(1 - \lambda) \cdot \lambda^{n-1}$ |

### GAE Formula

$$\hat{A}^{GAE}_t = (1-\lambda) \hat{A}^{(1)}_t + (1-\lambda)\lambda \hat{A}^{(2)}_t + (1-\lambda)\lambda^2 \hat{A}^{(3)}_t + ...$$

### The λ Parameter

The parameter $\lambda \in [0, 1]$ controls the balance between bias and variance:

| λ Value | Behavior |
|---------|----------|
| λ = 0 | Pure one-step TD (high bias, low variance) |
| λ = 1 | Pure Monte Carlo (low bias, high variance) |
| 0 < λ < 1 | Weighted combination (balanced) |

### Practical Implementation

In practice, GAE can be computed recursively:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}^{GAE}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + ...$$

Where $\delta_t$ is the TD error at time $t$.

---

## Summary: The Complete Picture

The gradient of the performance measure using GAE:

$$\nabla J(\theta) = \mathbb{E}_\pi \left[ \hat{A}^{GAE}_t \cdot \nabla \log \pi(a_t|s_t, \theta) \right]$$

This is often denoted simply as:

$$\nabla J(\theta) = G$$

### Why GAE is Important

1. **Practical implementation**: Provides an efficient way to estimate advantages
2. **Flexible bias-variance control**: λ parameter tunes the tradeoff
3. **Modern algorithm foundation**: Used in PPO, GRPO, and other RLHF algorithms

---

## Looking Ahead

With GAE providing advantage estimates, the next topics are:

1. **Trust Region Policy Optimization (TRPO)** - Constraining policy updates
2. **Proximal Policy Optimization (PPO)** - Simplified trust region method
3. **RL for LLMs** - How the agent-environment interface applies to language models
4. **Group Relative Policy Optimization (GRPO)** - Used in modern reasoning models
