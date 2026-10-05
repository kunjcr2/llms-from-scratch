# Policy Gradient Methods

This chapter is fundamental for understanding how reinforcement learning is used to impart reasoning capabilities to large language models. All RL applications in LLMs involve policy gradient methods - they are at the heart of algorithms used in DeepSeek, OpenAI's reasoning models (o1, o3), and every reasoning LLM that uses reinforcement learning.

---

## Why Policy Gradient Methods?

### The Traditional Approach (Action-Value Methods)

The objective of reinforcement learning is to find the optimal policy. Traditional methods work as follows:

1. Look at a particular state `s`
2. Enumerate all possible actions at that state (e.g., A1, A2, A3, A4)
3. Calculate the expected return for each action trajectory
4. Build a table mapping actions to expected returns (Q-values)
5. Select the action with the highest expected return

**Methods for building this table:**
- **Tabular Methods:** Monte Carlo, Temporal Difference, Dynamic Programming
- **Function Approximation:** When state space is huge (e.g., chess has 10^46 states), use a neural network to approximate Q(s, a)

### The Problem with Action-Value Methods

These methods are **not directly evaluating the policy**. Instead:
1. Calculate a number for each action
2. Check which number has the highest value
3. Use that to derive the policy

### The Policy Gradient Approach

**Can we directly optimize for the policy instead of using a proxy?**

Policy gradient methods learn the optimal policy **without considering the action value function**. We learn the policy directly.

---

## Policy Parameterization

### Notation

The policy is denoted as:

```
π(a | s, θ)
```

This indicates the probability of the agent taking action `a` while in state `s`, parameterized by `θ`.

Where:
- `θ` is the **policy parameter** (a d-dimensional vector)
- `θ ∈ ℝ^d'` (e.g., if d'=2, then θ = [θ₁, θ₂])

The goal is to find the value of `θ` that gives the best policy.

### Softmax of Numerical Preferences

**Problem:** How do we parameterize the policy in the first place?

**Step 1: Learn Action Preferences**

For a state `s` with actions A1, A2, A3, A4, A5, we learn a function that maps actions to preferences:

```
h(s, a, θ) → preference value
```

This function can be a polynomial or any differentiable function parameterized by θ.

**Step 2: Convert Preferences to Probabilities**

Preferences can have arbitrary values (e.g., 1-5), but probabilities must lie in [0, 1].

**Solution: Softmax Function**

The softmax function converts arbitrary values to probabilities:

```
π(a | s, θ) = exp(h(s, a, θ)) / Σ_b exp(h(s, b, θ))
```

**Example:**

| Action | Preference | Softmax Probability |
|--------|-----------|---------------------|
| A1     | 3         | e³/(e³+e²+e⁴+e⁵+e¹) = 0.086 |
| A2     | 2         | e²/(e³+e²+e⁴+e⁵+e¹) = 0.032 |
| A3     | 4         | e⁴/(e³+e²+e⁴+e⁵+e¹) = 0.234 |
| A4     | 5         | e⁵/(e³+e²+e⁴+e⁵+e¹) = 0.636 |
| A5     | 1         | e¹/(e³+e²+e⁴+e⁵+e¹) = 0.012 |

**Why Softmax Works:**
- Exponential function is always positive → no negative probabilities
- Denominator is sum of all numerators → values sum to 1
- Preserves relative ordering of preferences

**In Practice:**
- `h(s, a, θ)` is not given a priori - the agent must learn it through experience
- In modern applications (AlphaGo, RLHF, PPO, GRPO), the entire policy `π(a | s, θ)` is parameterized as a deep neural network

---

## Performance Measure

### Defining the Objective

To find the optimal policy, we need to define what "optimal" means.

**Performance Measure:**
```
J(θ) = V_π(s₀)
```

Where `V_π(s₀)` is the value function of the policy at the initial state - the expected return the agent gets from the start state to the end of the episode.

**Goal:** Find θ that maximizes J(θ)

### Gradient Ascent

To maximize J(θ), we use gradient ascent (opposite of gradient descent):

$$θ_{t+1} = θ_t + α · ∇J(θ)$$

Where:
- $θ_t$ = policy parameter at current time step
- $θ_{t+1}$ = policy parameter at next time step
- $α$ = step size (learning rate)
- $∇J(θ)$ = gradient of performance measure

**Intuition:** At each step, move in the direction that increases the performance measure until reaching a maximum.

---

## The Challenge of Computing ∇J(θ)

### The Dependency Problem

When computing the gradient of the performance measure, two things change:

1. **State distribution changes:** Different policies visit different states
2. **Action selection changes:** Different policies take different actions at each state

**Example:**

With policy π₁:
```
s₀ → (action a) → s₁ → (action a') → s₂ → ...
```

With different policy π₂:
```
s₀ → (action a'') → s₁' → (action a''') → s₂' → ...
```

**Intuition suggests:**
```
∇J(θ) depends on:
  - ∇_θ μ(s)  [gradient of state distribution]
  - ∇_θ π(a|s,θ)  [gradient of policy]
```

Computing the gradient of the state distribution with respect to θ seems extremely complex...

---

## The Policy Gradient Theorem

### Historical Context

In 2000, two independent research groups discovered this theorem:
- One group had a 12-page proof
- Rich Sutton's group had a 1-page proof

This theorem enabled rapid progress in RL and is the foundation for modern reasoning LLMs. Without it, we wouldn't have ChatGPT (trained via RLHF which uses policy gradients internally).

### The Theorem

The gradient of the performance measure is:

$$
∇J(θ) ∝ Σ_s μ(s) · Σ_a Q_π(s,a) · ∇_θ π(a|s,θ)
$$

**Key insight:** The gradient does NOT involve `∇_θ μ(s)` (gradient of state distribution), even though the state distribution depends on θ!

### Components Explained

**1. State Distribution μ(s)**

> How often each state is visited during an episode.

Example with 5 states and visit counts:
| State | Visits | μ(s) |
|-------|--------|------|
| s₁    | 10     | 10/71 |
| s₂    | 3      | 3/71 |
| s₃    | 50     | 50/71 |
| s₄    | 5      | 5/71 |
| s₅    | 3      | 3/71 |

**2. Action Value Function $Q_π(s,a)$**

> The expected return after taking action `a` in state `s` and following policy π thereafter.

**3. Policy Gradient $∇_θ π(a|s,θ)$**

> How the probability of taking action `a` in state `s` changes with respect to θ.

### Why This is Remarkable

The theorem shows that:
```
∇J(θ) = f(μ(s), Q(s,a), ∇π)   [NOT ∇μ!]
```

Analogously, if we had `f(u(x))`, normally:
```
df/dx = (df/du) · (du/dx)
```

But the policy gradient theorem gives us a result where `du/dx` (the gradient of state distribution) doesn't appear - making computation tractable!

---

## Summary

1. **Why Policy Gradients?** Direct policy optimization rather than learning value functions as a proxy

2. **Policy Parameterization:** Use softmax over numerical preferences to convert action preferences into probabilities

3. **Performance Measure:** J(θ) = V_π(s₀) - expected return from start state

4. **Gradient Ascent:** Iteratively update θ to maximize J(θ)

5. **Policy Gradient Theorem:** Provides a tractable formula for ∇J(θ) that doesn't require computing the gradient of the state distribution

---

## Looking Ahead

The next lectures will cover three policy gradient algorithms that build on this foundation:

1. **REINFORCE** - The fundamental policy gradient algorithm
2. **REINFORCE with Baseline** - Reduces variance using a baseline
3. **Actor-Critic Methods** - Combines policy gradient with value function approximation

These methods form the basis for modern approaches like:
- **PPO (Proximal Policy Optimization)** - Used by OpenAI
- **GRPO** - Used by DeepSeek
- **RLHF (Reinforcement Learning from Human Feedback)** - Used to train ChatGPT
