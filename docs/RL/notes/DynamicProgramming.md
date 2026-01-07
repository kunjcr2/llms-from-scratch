# Dynamic Programming

Dynamic Programming (DP) is a collection of algorithms that compute optimal policies given a **perfect model of the environment** (i.e., known transition probabilities). DP methods are not learning methods—they assume complete knowledge of the MDP.

---

## Step 1: Policy Evaluation (The Prediction Problem)

**Goal:** Given a policy $\pi$, estimate the value function $V_\pi(s)$ for all states.

**The Challenge:**  
Using the Bellman equation, we can express $V(s)$ in terms of $V(s')$:

$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_\pi(s') \right]$$

This creates a **recursive dependency**—every equation contains unknown value functions. For $n$ states, we get $n$ equations with $n$ unknowns.

**The Solution: Iterative Policy Evaluation**

Instead of solving the system directly (computationally expensive), we use an **iterative approach**:

1. **Initialize:** Assign $V_0(s) = 0$ for all states
2. **Iterate:** For each iteration $k$, update all states using:
   $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$
3. **Converge:** As $k \to \infty$, the sequence $V_0, V_1, V_2, \ldots$ converges to the true $V_\pi$

Each iteration performs a **full state sweep**—updating the value estimate for every state before moving to the next iteration.

---

## Step 2: Policy Improvement

**Goal:** Given an estimated value function, improve the current policy.

**The Core Idea:**  
For each state, check if the current policy's action is actually the best:

1. Calculate the **expected return** for each possible action:
   $$Q_\pi(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_\pi(s') \right]$$

2. If a different action yields a higher expected return than the policy's action, **update the policy**:
   $$\pi'(s) = \arg\max_a Q_\pi(s, a)$$

**Intuition:** If taking action $A_3$ from state $S$ gives expected return 8, but the current policy recommends $A_1$ which only gives 5, then the policy is not optimal—we should switch to $A_3$.

---

## Policy Iteration: Putting It Together

Policy iteration alternates between evaluation and improvement until convergence:

```
┌─────────────────────────────────────────────────────────┐
│  1. Start with an arbitrary policy π₀                   │
│                          ↓                              │
│  2. Policy Evaluation: Compute V_π for current policy   │
│                          ↓                              │
│  3. Policy Improvement: Update policy using V_π         │
│                          ↓                              │
│  4. If policy changed → Go to step 2                    │
│     If policy stable → Done! (π* and V* found)          │
└─────────────────────────────────────────────────────────┘
```

The process converges to the **optimal policy** $\pi^*$ and **optimal value function** $V^*$.

---

## Value Iteration: A Shortcut

Value iteration combines policy evaluation and improvement into a single update step using the **Bellman Optimality Equation**:

$$V_{k+1}(s) = \max_a \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$

This directly computes the optimal value function without explicitly maintaining a policy during iteration.

---

## Practical Example: Car Rental Problem

| Component | Description |
|-----------|-------------|
| **States** | Number of cars at each of two locations (max 20 each) |
| **Actions** | Cars moved between locations overnight (-5 to +5) |
| **Rewards** | +$10 per car rented, -$2 per car moved |
| **Dynamics** | Requests and returns follow Poisson distributions |

Policy iteration finds the optimal policy: a matrix indicating how many cars to move for each possible state configuration.

---

## Limitations of Dynamic Programming

| Limitation | Description |
|------------|-------------|
| **Requires Model** | Must know exact transition probabilities $P(s', r \| s, a)$ |
| **Not Learning** | Does not learn from experience—assumes perfect knowledge |
| **Computational Cost** | Full state sweeps required; expensive for large state spaces |
| **Impractical for Real Problems** | Transition probabilities often impossible to obtain (e.g., chess, robotics) |

> Despite these limitations, DP provides the **theoretical foundation** for understanding Monte Carlo and Temporal Difference methods, which overcome these drawbacks by learning from experience.
