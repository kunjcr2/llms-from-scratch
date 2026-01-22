# Proximal Policy Optimization (PPO)

PPO is the algorithm at the heart of **Reinforcement Learning with Human Feedback (RLHF)**. It combines the simplicity of vanilla policy gradient methods with the trust region concept from TRPO, giving us the best of both worlds.

---

## Motivation: From Vanilla PG to PPO

### Vanilla Policy Gradient Recap

For vanilla policy gradient methods, the gradient of the performance measure is:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s,a)\right]
$$

Where:
- $J(\theta)$ = performance measure (the "mountain" we want to climb)
- $\hat{A}(s,a)$ = advantage estimate — how much better/worse the action is compared to the policy's default behavior

**Intuition**: If advantage is positive → reinforce the action (climb up). If negative → penalize it (go down).

The objective function whose gradient gives us the above:

$$
L(\theta) = \mathbb{E}\left[\log \pi_\theta(a|s) \cdot \hat{A}(s,a)\right]
$$

### The Trust Region Problem (TRPO)

After 2015, **trust region methods** emerged to address instability in policy updates. TRPO maximizes a surrogate objective with a constraint:

$$
\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot \hat{A}(s,a)\right] \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) \leq \delta
$$

The actual theory suggests a **penalty** instead of a constraint:

$$
L(\theta) = \mathbb{E}\left[\frac{\pi_\theta}{\pi_{\theta_{old}}} \cdot \hat{A}\right] - \beta \cdot D_{KL}^{max}(\pi_{\theta_{old}} \| \pi_\theta)
$$

**Problem**: TRPO requires computing the inverse of the Fisher information matrix — very computationally expensive!

### PPO: Best of Both Worlds

Can we get:
- **Simplicity** of vanilla policy gradients
- **Trust region concept** (penalizing large updates) from TRPO

**Answer**: Yes! This is PPO (2017).

---

## The Clipped Surrogate Objective

### Define the Probability Ratio

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

- If $r_t(\theta) \gg 1$ → new policy deviates significantly from old policy
- If $r_t(\theta) = 1$ → policies are identical

### The Core Insight: Different Clipping for Different Advantages

#### Case 1: Advantage > 0 (Good Action)

When the action is better than the policy's default behavior:
- We want to reinforce this action
- But we **don't want to be overly aggressive** with updates
- **Solution**: Clip at $1 + \epsilon$ to prevent over-optimization

```
r_t(θ)
  |             ___________
  |            /
  |           /
  |          /(1+ε)
  |         /
  |        /
  |_______/________________
  0       1      1+ε      →
```

#### Case 2: Advantage < 0 (Bad Action)

When the action is worse than the policy's default:
- Large $r_t(\theta)$ values → large negative penalty (good! we want this)
- Small $r_t(\theta)$ values → penalty reduces (bad! we don't want to let it off easy)
- **Solution**: Clip at $1 - \epsilon$ to maintain high penalty

```
r_t(θ)
  |
  |       _______________
(1-ε)____|
  |
  |
  |________________________
  0      1-ε     1        →
```

### Combined Clipping Function

The clip function bounds $r_t(\theta)$ between $1-\epsilon$ and $1+\epsilon$:

$$
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

### The PPO Objective

$$
L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \cdot \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t\right)\right]
$$

**Why the minimum?**
- For **positive advantages**: min selects the clipped version when $r_t > 1+\epsilon$ (prevents over-optimization)
- For **negative advantages**: min selects the clipped version when $r_t < 1-\epsilon$ (maintains penalty)

The sign of the advantage automatically determines which constraint is active!

---

## Complete PPO Objective

The full objective includes three components:

$$
L(\theta) = \mathbb{E}\left[L^{CLIP}(\theta) - c_1 \cdot L^{VF}(\theta) + c_2 \cdot S[\pi_\theta]\right]
$$

### Component 1: Clipped Surrogate ($L^{CLIP}$)
The main policy optimization term as described above.

### Component 2: Value Function Loss ($L^{VF}$)
Since advantages are computed using the value function:

$$
\hat{A}_t \approx r_t + \gamma V(s_{t+1}) - V(s_t)
$$

We need accurate value estimates, so we minimize:

$$
L^{VF} = (V_\theta(s_t) - V_t^{target})^2
$$

> **Note**: PPO requires a separate model/head to estimate the value function.

### Component 3: Entropy Bonus ($S[\pi_\theta]$)
Encourages exploration by maximizing policy entropy:

$$
S[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

This prevents premature convergence to a deterministic policy.

### Gradient Ascent Update

$$
\theta \leftarrow \theta + \alpha \nabla_\theta L(\theta)
$$

---

## Core Idea of PPO in RLHF

1. **Prompt pool** → pick a batch of prompts
2. **Policy model** → generate answers (sampled)
3. **Reward model** → score each answer
4. **PPO update** → adjust the policy so:
   - High-reward answers become **more likely**
   - Low-reward answers become **less likely**
   - Changes are **clipped** to prevent drift

PPO replaces the "plain policy gradient step" in RLHF with a **safe update**.

---

## PPO in RLHF — Workflow

```
[Prompts] → [Policy Model] → [Generated Answers]
                          ↓
                [Reward Model scores]
                          ↓
          [PPO objective: reward + clip]
                          ↓
                 [Optimizer updates policy]
```

---

## PPO with Hugging Face TRL

Hugging Face's [**trl**](https://huggingface.co/docs/trl/index) library simplifies RLHF. In practice, use a **trained reward model**.

---

## PPO vs TRPO Summary

| Aspect | TRPO | PPO |
|--------|------|-----|
| Trust Region | KL divergence constraint | Clipping mechanism |
| Computation | Requires Fisher matrix inverse | Simple gradient updates |
| Implementation | Complex | Straightforward |
| Performance | Strong | Comparable or better |

---

## Resources

- [OpenAI PPO paper (2017)](https://arxiv.org/abs/1707.06347)
- [Hugging Face TRL docs](https://huggingface.co/docs/trl/index)
- [Illustrated PPO blog](https://huggingface.co/blog/ppo)
- [RLHF with TRL course](https://huggingface.co/learn/rl-course/unit2/ppo)

---

## Key Takeaways

1. **PPO bridges vanilla PG and TRPO** — simple implementation with trust region benefits
2. **Clipping prevents destructive updates** — different clipping behavior for positive vs negative advantages
3. **Full objective has 3 parts**: clipped surrogate + value function loss + entropy bonus
4. **PPO was crucial for ChatGPT** — the algorithm that aligned LLMs with human preferences
5. **No matrix inverse needed** — unlike TRPO, making it computationally efficient
