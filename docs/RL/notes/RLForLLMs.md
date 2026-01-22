# Reinforcement Learning for Large Language Models

This lecture bridges classical RL concepts with their application to LLMs. We cast the language model prediction problem into the agent-environment interface, enabling all previously covered RL techniques to be applied.

---

## Prerequisites

- **RL Fundamentals** (states, actions, rewards, policies): See [RLFundamentals.md](./RLFundamentals.md)
- **Policy Gradient Methods**: See [PolicyGradientTheorem.md](./PolicyGradientTheorem.md)
- **TRPO & PPO**: See [TRPO.md](./TRPO.md) and [ProximalPolicyOptimization.md](./ProximalPolicyOptimization.md)

---

## Course Progression Recap

Before diving into RL for LLMs, here's the path we've covered:

1. **Tabular Methods** - Value tables for small state spaces
2. **Function Approximation** - Neural networks for large state spaces
3. **Policy Gradient Methods** - Direct policy optimization
4. **REINFORCE** - First policy gradient algorithm
5. **Actor-Critic** - Baseline subtraction using value functions
6. **TRPO** - Trust region constraints for stable updates
7. **PPO** - Simplified clipping mechanism (best of both worlds)

Now we apply these methods to **Large Language Models**.

---

## Why RL for LLMs?

At first glance, casting LLMs as RL problems isn't obvious. In traditional RL:
- An **agent** interacts with an **environment**
- The agent receives **rewards** and improves

But how does this map to text generation?

> The key insight: Next-token prediction can be viewed as a sequential decision-making process.

---

## The Agent-Environment Interface for LLMs

### Mapping RL Components to LLMs

| RL Component | LLM Equivalent |
|--------------|----------------|
| **Agent** | The LLM itself |
| **State** | Prompt + previously generated tokens |
| **Action** | Next token prediction |
| **Policy** | The LLM (probability distribution over vocabulary) |
| **Environment** | User/Python interpreter providing feedback |

---

## States in LLMs

The **state** is the information available to the agent before taking an action. For LLMs, this is the concatenation of:
1. The original prompt
2. All tokens generated so far

### Example: Math Problem

**Prompt**: "Roger has 5 tennis balls and he buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does Roger have?"

| State | Content |
|-------|---------|
| $S_0$ | Prompt only |
| $S_1$ | $S_0$ + "Roger" |
| $S_2$ | $S_1$ + "has" |
| $S_3$ | $S_2$ + "11" |
| $S_4$ | $S_3$ + "tennis" |
| $S_5$ | $S_4$ + "balls" |

Each action (token) is appended to the state for the next iteration.

---

## Actions in LLMs

The **action** is the next token the model predicts. The LLM outputs a probability distribution over its entire vocabulary:

```
P("Roger" | prompt) = 0.15
P("He" | prompt) = 0.12
P("The" | prompt) = 0.08
...
```

The token with the highest probability (or sampled from the distribution) becomes the action.

> The LLM **is** the policy because it outputs $P(\text{action} | \text{state})$ for all possible actions.

---

## The Policy is the LLM

In most RL problems, the agent and policy are conceptually separate. But for LLMs:

$$\pi_\theta(a|s) = P(\text{next token} | \text{current context})$$

This is exactly what the LLM computes. The model outputs probabilities over all vocabulary tokens, making it a perfect fit for the policy definition.

> **Unique to LLMs**: The agent and the policy are the same entity.

---

## Rewards in LLMs

Rewards are received **only at the end** of the response (sparse rewards):

| Step | Token | Reward |
|------|-------|--------|
| 1 | "Roger" | 0 |
| 2 | "has" | 0 |
| 3 | "11" | 0 |
| 4 | "tennis" | 0 |
| 5 | "balls" | **+1** (if correct) or **0** (if wrong) |

This is analogous to chess: you don't get rewards for individual moves, only after the game ends (win/lose).

---

## Mathematical Notation

### Prompt and Completion

- **Prompt** $X = (x_1, x_2, ..., x_T)$ - sequence of input tokens
- **Completion** $Y = (y_1, y_2, ..., y_n)$ - sequence of generated tokens
- **Policy** $\pi_\theta$ - the LLM parameterized by $\theta$

### Autoregressive Generation

LLMs predict one token at a time based on all previous tokens:

$$P(Y|X) = P(y_1|X) \cdot P(y_2|X, y_1) \cdot P(y_3|X, y_1, y_2) \cdots$$

Using product notation:

$$P(Y|X; \pi_\theta) = \prod_{t=1}^{n} P(y_t | X, y_{<t}; \pi_\theta)$$

Where $y_{<t}$ denotes all tokens before position $t$: $(y_1, y_2, ..., y_{t-1})$.

### Shorthand Notation

Instead of writing:
- $P(y_2 | X, y_1)$
- $P(y_3 | X, y_1, y_2)$

We write:
- $P(y_2 | X, y_{<2})$
- $P(y_3 | X, y_{<3})$

---

## The Reward Model Challenge

Unlike chess (clear win/lose) or games (score), LLM rewards are often **subjective**:

- "Write a poem" - How do you rate a poem?
- "Explain quantum physics" - Quality is subjective
- Different users have different preferences

### Solution: Reward Models

AI labs (OpenAI, Google, Anthropic) collect **human preference data**:

1. Generate multiple responses
2. Ask humans: "Which response do you prefer?"
3. Preferred response gets higher reward
4. Train a **reward model** on this preference data

### Example: ChatGPT Preference Collection

```
Response 1: [Generated text A]
Response 2: [Generated text B]

"Which response do you prefer?" → User selects Response 2

Result: Response 2 receives higher reward than Response 1
```

The reward model is trained via supervised learning on these preferences:
- **Input**: Prompt + Response
- **Output**: Reward score

---

## Training the Reward Model

The reward model $R_\phi$ is a classifier trained to predict human preferences:

1. Collect many (prompt, response_A, response_B, preference) tuples
2. Train $R_\phi$ to assign higher scores to preferred responses
3. Use $R_\phi$ in RL training to provide rewards

This is crucial because:
- Ground truth rewards don't exist for open-ended tasks
- Human feedback is expensive and slow
- The reward model provides cheap, fast approximations

> For verifiable tasks (math, code), rewards can be computed directly without a reward model.

---

## The Complete Picture

```
┌─────────────────────────────────────────────────────────┐
│                    RL FOR LLMs                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Agent (LLM) ←──────────── Policy π_θ                  │
│       │                                                 │
│       ▼                                                 │
│   State S_t = Prompt + Previous Tokens                  │
│       │                                                 │
│       ▼                                                 │
│   Action a_t = Next Token                               │
│       │                                                 │
│       ▼                                                 │
│   S_{t+1} = S_t + a_t                                   │
│       │                                                 │
│       ▼ (repeat until EOS)                              │
│                                                         │
│   Reward = RewardModel(prompt, completion)              │
│       │                                                 │
│       ▼                                                 │
│   Update π_θ using PPO/GRPO/etc.                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **LLMs as RL Agents**: Despite initial non-intuitiveness, LLMs fit the agent-environment interface perfectly.

2. **State = Context**: The state is the prompt plus all previously generated tokens.

3. **Action = Next Token**: Each action is a token prediction from the vocabulary.

4. **Policy = LLM**: The model itself is the policy, outputting $P(a|s)$ over all tokens.

5. **Sparse Rewards**: Rewards come only at the end of generation.

6. **Reward Models**: For subjective tasks, train a model on human preference data.

7. **All RL Methods Apply**: Once cast in this framework, REINFORCE, PPO, GRPO, etc. all become applicable.

---

## What's Next

With LLMs cast as RL problems, we can now apply:

- **GRPO** (Group Relative Policy Optimization) - See [GRPO.md](./GRPO.md)
- **DPO** (Direct Preference Optimization) - See [DirectPreferenceOptimization.md](./DirectPreferenceOptimization.md)
- **Reward Modeling** - See [RewardModel.md](./RewardModel.md)

These methods build directly on the foundation established here.
