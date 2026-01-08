# TD Control: SARSA and Q-Learning

TD Control methods extend temporal difference learning from the **prediction problem** (estimating value functions) to the **control problem** (finding optimal policies).

---

## From Prediction to Control

### Prediction Problem (Review)
Given a policy, estimate the value function:

$$V(S) \leftarrow V(S) + \alpha \left[ R_{t+1} + \gamma V(S') - V(S) \right]$$

### Control Problem
Find the optimal policy. This requires:
1. Estimating **action values** Q(s, a) instead of state values V(s)
2. Using the Q-values to improve the policy
3. Balancing exploration and exploitation

---

## Why Action Values?

State value $V(s)$ tells us: "How good is this state?"

But for control, we need: "Which action should I take?"

**Action value function** $Q(s, a)$ answers: "What is the expected return if I take action $a$ in state $s$ and follow the policy thereafter?"

**Key Insight**: If we know Q-values, the optimal policy is simply:
$$\pi^*(s) = \arg\max_a Q(s, a)$$

---

## TD Update for Action Values

Replace V with Q in the TD update:

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R_{t+1} + \gamma Q(S', A') - Q(S, A) \right]$$

Where:
- $S, A$ = current state and action
- $R_{t+1}$ = reward received
- $S', A'$ = next state and next action
- $\alpha$ = learning rate
- $\gamma$ = discount factor

---

## Policy Iteration in TD Methods

Similar to Dynamic Programming and Monte Carlo, TD control uses an iterative process:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Start with arbitrary Q-values                           │
│                          ↓                                  │
│  2. Generate experience using current policy                │
│                          ↓                                  │
│  3. Update Q-values after each step (not episode)           │
│                          ↓                                  │
│  4. Improve policy based on updated Q-values                │
│                          ↓                                  │
│  5. Repeat until convergence                                │
└─────────────────────────────────────────────────────────────┘
```

**Key Difference from MC**: Updates happen after each step, not after each episode.

---

## Epsilon-Greedy Policy (Review)

To balance exploration and exploitation:

- With probability $\epsilon$: take a **random** action (explore)
- With probability $1 - \epsilon$: take the **greedy** action (exploit)

**Example**:
| State S | Action | Q(S, A) |
|---------|--------|---------|
| S | A1 | 32 |
| S | A2 | 28 |
| S | A3 | 45 |

- Greedy policy always chooses A3 (max Q-value)
- Epsilon-greedy sometimes chooses A1 or A2 randomly
- This ensures all actions get explored

---

## SARSA: On-Policy TD Control

### The Name
SARSA uses the tuple $(S, A, R, S', A')$ for each update:
- **S**: Current state
- **A**: Current action
- **R**: Reward received
- **S'**: Next state
- **A'**: Next action

### Update Rule

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$

### Algorithm

```
Initialize Q(s, a) arbitrarily for all state-action pairs

For each episode:
    Initialize S
    Choose A from S using epsilon-greedy policy derived from Q
    
    For each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using epsilon-greedy policy derived from Q
        
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        
        S <- S'
        A <- A'
    Until S is terminal
```

### On-Policy Meaning

**Behavior policy** = **Target policy** = Epsilon-greedy

The agent:
- Follows epsilon-greedy policy to select actions
- Learns about that same epsilon-greedy policy

---

## On-Policy vs Off-Policy Learning

### On-Policy Learning

**Analogy**: Learning to ride a bicycle by practicing yourself.

- You follow your current riding style (your policy)
- You make mistakes and learn from them
- You update your style based on your own experience
- The policy you learn = the policy you follow

### Off-Policy Learning

**Analogy**: Learning to ride a bicycle while watching a professional cyclist.

- You still ride around and make mistakes
- But when updating, you ask: "What would the expert do?"
- You learn the expert's policy while following your own
- The policy you learn differs from the policy you follow

### Formal Definition

| Type | Behavior Policy | Target Policy |
|------|-----------------|---------------|
| On-Policy | Policy A | Same Policy A |
| Off-Policy | Policy B | Different Policy A |

---

## Q-Learning: Off-Policy TD Control

### Key Insight

Q-Learning uses:
- **Behavior policy**: Epsilon-greedy (for exploration)
- **Target policy**: Greedy (always picks max Q-value)

### Update Rule

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \right]$$

**Critical Difference from SARSA**: Uses $\max_a Q(S', a)$ instead of $Q(S', A')$

### Algorithm

```
Initialize Q(s, a) arbitrarily for all state-action pairs

For each episode:
    Initialize S
    
    For each step of episode:
        Choose A from S using epsilon-greedy policy derived from Q
        Take action A, observe R, S'
        
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        
        S <- S'
    Until S is terminal
```

### Off-Policy Explanation

The agent:
- **Behaves** using epsilon-greedy (sometimes explores randomly)
- **Learns** about the greedy policy (always assumes optimal action next)

The "expert" it learns from is the greedy policy that always picks the best action.

---

## SARSA vs Q-Learning Comparison

| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| **Policy Type** | On-policy | Off-policy |
| **Behavior Policy** | Epsilon-greedy | Epsilon-greedy |
| **Target Policy** | Epsilon-greedy | Greedy |
| **Update Uses** | $Q(S', A')$ | $\max_a Q(S', a)$ |
| **Learning Style** | Learns what it does | Learns the optimal |

### The Key Formula Difference

**SARSA**:
$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$

**Q-Learning**:
$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \right]$$

---

## Practical Example: Cliff Walking

### The Environment

```
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ S │ C │ C │ C │ C │ C │ C │ C │ C │ C │ C │ G │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

S = Start, G = Goal, C = Cliff
```

**Rules**:
- All transitions: reward = -1 (encourages shortest path)
- Falling into cliff: reward = -100, return to start
- Goal: Reach G from S while avoiding the cliff

### SARSA Result: Safe Path

```
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ ↑ │ → │ → │ → │ → │ → │ → │ → │ → │ → │ → │ ↓ │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ ↑ │   │   │   │   │   │   │   │   │   │   │ ↓ │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ ↑ │   │   │   │   │   │   │   │   │   │   │ ↓ │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ S │ C │ C │ C │ C │ C │ C │ C │ C │ C │ C │ G │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

SARSA takes the **safe path** — far from the cliff edge.

### Q-Learning Result: Optimal Path

```
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ ↑ │ → │ → │ → │ → │ → │ → │ → │ → │ → │ → │ ↓ │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ S │ C │ C │ C │ C │ C │ C │ C │ C │ C │ C │ G │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

Q-Learning takes the **optimal path** — shortest route along the cliff edge.

### Why the Difference?

**SARSA** (on-policy):
- Learns about the epsilon-greedy policy it follows
- Since epsilon-greedy sometimes takes random actions, walking near the cliff means occasionally falling in
- Learns to avoid the cliff edge because its policy sometimes makes mistakes

**Q-Learning** (off-policy):
- Learns about the greedy policy (always optimal)
- Assumes it will always take the best action
- Therefore, walking near the cliff is fine if the best action is to stay on the edge
- Finds the shortest path because it ignores exploration mistakes

### Practical Implication

| Algorithm | Path | Behavior |
|-----------|------|----------|
| SARSA | Safe but longer | Conservative — accounts for exploration mistakes |
| Q-Learning | Optimal but risky | Aggressive — assumes perfect execution |

---

## When to Use Which?

### Use SARSA When:
- Safety matters (e.g., robotics, autonomous vehicles)
- You want the learned policy to match actual behavior
- Exploration mistakes have high costs

### Use Q-Learning When:
- You want to learn the optimal policy
- The cost of exploration is acceptable
- You can reduce epsilon over time

---

## Implicit Policy Iteration

In both SARSA and Q-Learning, policy iteration happens implicitly:

1. **Q-values are updated after each step** (not after each episode like MC)
2. **Policy is derived from Q-values** (epsilon-greedy based on current Q)
3. As Q-values change, the policy automatically changes
4. No explicit "policy improvement" step needed

This is why TD control is efficient — both evaluation and improvement happen simultaneously.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Control Problem** | Finding the optimal policy (vs prediction = estimating values) |
| **Action Values** | Q(s, a) tells us expected return for taking action a in state s |
| **SARSA** | On-policy; uses Q(S', A') for update |
| **Q-Learning** | Off-policy; uses max Q(S', a) for update |
| **On-Policy** | Learns about the policy it follows |
| **Off-Policy** | Learns about a different (better) policy |
| **Epsilon-Greedy** | Balances exploration (random) and exploitation (greedy) |
| **Implicit Iteration** | Q-values and policy update together after each step |

---

## Key Takeaways

1. **TD Control extends TD Prediction**: Replace V(s) with Q(s, a)

2. **SARSA is on-policy**: Learns about the epsilon-greedy policy it follows

3. **Q-Learning is off-policy**: Learns about the greedy policy while following epsilon-greedy

4. **The max makes the difference**: Q-Learning uses $\max_a Q(S', a)$, SARSA uses $Q(S', A')$

5. **Safety vs Optimality**: SARSA is safer, Q-Learning finds optimal but riskier paths

6. **Updates after each step**: Unlike MC, TD methods update Q-values after every step

7. **Implicit policy iteration**: No separate evaluation and improvement phases

> **Next**: These are **tabular methods** where each state-action pair has its own value. What if we have too many states? Function approximation allows us to generalize — and neural networks are universal function approximators.
