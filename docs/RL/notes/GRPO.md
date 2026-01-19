# Group Relative Policy Optimization (GRPO)

GRPO is the reinforcement learning algorithm used to train **DeepSeek** and other reasoning language models. It builds upon PPO but eliminates the need for a separate value function model.

---

## Prerequisites

- **RL Fundamentals** (states, actions, rewards, policies): See [RLFundamentals.md](./RLFundamentals.md)
- **Policy Gradient Methods**: See [PolicyGradientTheorem.md](./PolicyGradientTheorem.md)
- **REINFORCE & Baseline**: See [REINFORCE.md](./REINFORCE.md)
- **Advantage Function**: See [AdvantageFunction.md](./AdvantageFunction.md)
- **Trust Regions & PPO**: See [TRPO.md](./TRPO.md)

---

## RL for Large Language Models

### Agent-Environment Interface for LLMs

| Component | In LLMs |
|-----------|---------|
| **Agent** | The LLM itself (DeepSeek, GPT, LLaMA, etc.) |
| **Environment** | Python interpreter, dataset, or human evaluators |
| **Policy** | $\pi_\theta$ — the model's probability distribution over tokens |

### States in LLMs

The **state** is the concatenation of the prompt and all previously generated tokens:

```
State S₀: "Roger has 5 tennis balls..."  (prompt)
State S₁: S₀ + "Roger"
State S₂: S₁ + "has"
State S₃: S₂ + "11"
State S₄: S₃ + "tennis"
State S₅: S₄ + "balls"
```

### Actions in LLMs

Each **action** is selecting the next token from the vocabulary based on the probability distribution:

$$\pi_\theta(a|s) = P(\text{next token} | \text{current context})$$

The model outputs probabilities for all tokens in the vocabulary, then samples or selects the highest probability token.

### Rewards in LLMs (Sparse Reward)

Unlike classical RL where rewards come after every action, LLMs receive rewards **only at the end of the response**:

```
Step 1: "Roger"  → Reward = 0
Step 2: "has"    → Reward = 0
Step 3: "11"     → Reward = 0
Step 4: "tennis" → Reward = 0
Step 5: "balls"  → Reward = 1 (if answer is correct) or 0 (if wrong)
```

> This sparse reward structure is unique to LLMs and creates challenges for credit assignment.

---

## Policy Updates Based on Rewards

Given a state, the LLM produces a probability distribution over all vocabulary tokens. The goal is to adjust this distribution:

- **Positive reward** → Increase probability of the chosen action
- **Negative reward** → Decrease probability of the chosen action

### Policy Gradient Update (Recap)

$$\theta_{t+1} = \theta_t + \alpha \cdot G_t \cdot \nabla \log \pi_\theta(a_t|s_t)$$

Where $G_t$ is the cumulative return. For detailed derivation, see [REINFORCE.md](./REINFORCE.md).

---

## The Value Function Problem

Traditional algorithms (Actor-Critic, PPO) use a **value function** $V_\phi(s)$ to estimate expected future rewards:

$$A_t = G_t - V_\phi(s_t)$$

### Why Value Functions are Problematic for LLMs

1. **Memory overhead**: Requires a separate neural network (critic) alongside the LLM
2. **Training complexity**: Must jointly train the critic and the policy
3. **Sparse rewards**: With rewards only at episode end, value estimation is difficult

---

## GRPO's Key Innovation: Group-Based Advantage

Instead of using a learned value function, GRPO estimates advantages by **generating multiple responses** to the same prompt.

### The Process

1. Given prompt, generate **G responses** (e.g., 10 different completions)
2. Evaluate each response and collect rewards: $r_1, r_2, ..., r_G$
3. **Normalize rewards** to compute advantages:

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

### Example

For a math problem with 4 generated responses:

| Response | Reward | Normalized Advantage |
|----------|--------|---------------------|
| Response 1 (correct, elegant) | 1.0 | +1.5 |
| Response 2 (correct) | 0.8 | +0.5 |
| Response 3 (wrong) | 0.0 | -1.0 |
| Response 4 (wrong, nonsense) | 0.0 | -1.0 |

> Responses better than average get positive advantages; worse ones get negative.

### Why This Works

- **No critic needed**: Advantage is computed from the group, not a learned model
- **Relative comparison**: Model learns which responses are relatively better
- **Efficient**: Uses the same LLM to generate all responses

---

## PPO Clipping (Background)

GRPO builds on PPO's clipping mechanism. For detailed derivation, see [TRPO.md](./TRPO.md).

### The Ratio

$$r_\theta = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$$

This ratio measures how much the new policy differs from the old policy for a given action.

### Clipping Mechanism

PPO clips this ratio to prevent large policy updates:

$$L^{CLIP} = \min\left( r_\theta \cdot A_t, \; \text{clip}(r_\theta, 1-\epsilon, 1+\epsilon) \cdot A_t \right)$$

Where $\epsilon$ is typically 0.1 or 0.2.

### Intuition

| Advantage | Ratio | Effect |
|-----------|-------|--------|
| $A > 0$ (good action) | $r_\theta > 1 + \epsilon$ | Clip — don't get too greedy |
| $A > 0$ (good action) | $r_\theta < 1 + \epsilon$ | Allow increase |
| $A < 0$ (bad action) | $r_\theta < 1 - \epsilon$ | Clip — still penalize sufficiently |
| $A < 0$ (bad action) | $r_\theta > 1 - \epsilon$ | Allow decrease |

---

## GRPO Loss Function

The complete GRPO loss combines PPO clipping with a KL divergence penalty:

$$L^{GRPO} = L^{PPO} - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

Where:
- $L^{PPO}$ is the clipped surrogate objective
- $\pi_{ref}$ is the **reference policy** (the base SFT model before RL)
- $\beta$ is a hyperparameter controlling the KL penalty strength

### Why the KL Term?

The KL divergence ensures the fine-tuned model doesn't deviate too far from the base model:

- Prevents **mode collapse**
- Preserves **general capabilities**
- Ensures **stable training**

---

## GRPO Training Loop

```
For each batch of prompts:
    1. Sample G responses per prompt using current policy π_θ
    
    2. Compute rewards for each response (e.g., correctness check)
    
    3. Normalize rewards within each group:
       A_i = (r_i - mean(r)) / std(r)
    
    4. Compute PPO clipped loss with group advantages
    
    5. Add KL penalty: L = L_PPO - β · D_KL(π_θ || π_ref)
    
    6. Update policy: θ ← θ - α · ∇L
```

---

## Comparison: PPO vs GRPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Value function** | Required (critic network) | Not needed |
| **Advantage estimation** | GAE using learned V(s) | Normalized group rewards |
| **Memory** | Higher (policy + critic) | Lower (policy only) |
| **Reward structure** | Works with dense rewards | Designed for sparse rewards |
| **Use case** | General RL | LLM training |

---

## Key Takeaways

1. **LLMs as RL agents**: State = context, Action = next token, Reward = end-of-response score

2. **Sparse rewards**: LLMs only receive feedback after complete responses

3. **Group-based advantage**: Generate multiple responses, normalize rewards within the group

4. **No critic needed**: Eliminates the value function model entirely

5. **KL regularization**: Keeps the policy close to the reference (base) model

6. **Foundation**: GRPO combines insights from REINFORCE, PPO, and TRPO but adapted for LLM training

---

## Why GRPO for Reasoning Models

GRPO became popular for training reasoning models like DeepSeek-R1 because:

1. **Verifiable rewards**: Math/code problems have ground-truth answers
2. **Multiple solution paths**: Same problem can be solved different ways
3. **Efficient**: No need for reward models or value functions for correctness checking
4. **Scalable**: Group sampling parallelizes well on modern hardware
