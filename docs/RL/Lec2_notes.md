## 1. The Concept of Value

Value is one of the four foundational elements of reinforcement learning, alongside policy, rewards, and the environment model. It provides a critical measure for decision-making by considering the future impact of a state or action.

### Intuition: Long-Term Desirability

The primary distinction is that **Value is fundamentally different from immediate rewards**. While rewards concern only short-term desirability, **Value focuses on the long-term desirability of the state**.

Instead of just looking at single rewards, the Value calculation involves **adding all the rewards together** until the end of the episode to evaluate the resulting **return ($G_t$)** received by the agent. Later rewards are discounted by a factor ($\gamma$) because **immediate rewards are more valuable** than those received later in the trajectory.

### Types of Value Functions

The sources identify two essential value functions, both aiming to estimate the expected return:

#### A. State Value Function ($V_{\pi}(s)$)

The state value function is formally defined as the **expected return** the agent receives, starting in a specific state ($s$) and following a particular policy ($\pi$) thereafter until the end of the episode.

- **Notation:** It is consistently denoted by the symbol $V_{\pi}(s)$ across RL literature.
- **Estimation:** To calculate the value of a state, the agent may run multiple episodes (e.g., 10 episodes), determine the return for each episode ($G_t$), sum them up, and then divide by the total number of episodes to calculate the **mean (expected value)** of the returns.

#### B. Action Value Function ($Q(s, a)$)

The action value function quantifies the value associated with a specific action taken in a specific state.

- **Notation:** It is denoted by the famous and frequently repeated symbol $Q(s, a)$.
- **Definition:** It is formally defined as the **expected return starting in a state, taking an action, and then following a policy thereafter**.
- **Utility:** The value of states and actions (the Q values) are highly useful for implementation because they provide direct information: an agent can look at the Q table and see which action has the highest Q value for a given state. An agent's policy can be formed by telling it to **always choose the action with the maximum Q value**.
- **Relation to $V(s)$:** The relationship between $Q(s, a)$ and $V(s)$ depends entirely on the policy chosen by the agent. If the policy dictates choosing action $A_1$, then $Q(s, A_1)$ will exactly match $V(s)$.

### Value Estimation via Bellman Equations

Estimating the value function is often referred to as the **prediction problem** in RL [23, 15:57]. This process was significantly simplified by Richard Bellman's key finding:

1.  **The Recursive Nature (Bellman Equation):** Bellman stated that the value of being in a state ($V(s)$) can be expressed recursively in terms of the value of the next state ($V(s')$).
    - **Intuition:** The value of a state is equal to the expected immediate reward ($R$) plus the **discounted expected value of the next state** ($\gamma \cdot V(s')$). This recursive relationship is powerful and is at the heart of algorithms used in complex applications.
2.  **The Optimal Choice (Bellman Optimality Equation):** This equation extends the basic Bellman equation to define the best possible value by selecting the action that maximizes the expected return.
    - The optimal action is chosen by selecting the **maximum** of the immediate reward plus the discounted value of the next state: $\max_a (R + \gamma \cdot V(s'))$ [41, 28:28].
    - This is equivalent to finding the **maximum of the action value functions** for all possible actions: $\max (Q(s, a))$. Solving for this maximum allows the agent to confidently choose the optimal action for every state.

---

## 2. The Role of the Model

The Model of the environment is defined by the necessary information to determine the dynamics of the system, primarily requiring knowledge of the **transition probabilities** from one state ($S$) to the next state ($S'$) [77, 78, 55:49].

RL methods are categorized based on whether they use or require a model: Model-Based or Model-Free.

### A. Model-Based Methods (Dynamic Programming)

Dynamic Programming (DP) is a collection of algorithms that compute optimal policies given a **perfect model of the environment** (i.e., known transition probabilities). DP methods are not learning methods—they assume complete knowledge of the MDP.

---

#### Step 1: Policy Evaluation (The Prediction Problem)

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

#### Step 2: Policy Improvement

**Goal:** Given an estimated value function, improve the current policy.

**The Core Idea:**  
For each state, check if the current policy's action is actually the best:

1. Calculate the **expected return** for each possible action:
   $$Q_\pi(s, a) = \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_\pi(s') \right]$$

2. If a different action yields a higher expected return than the policy's action, **update the policy**:
   $$\pi'(s) = \arg\max_a Q_\pi(s, a)$$

**Intuition:** If taking action $A_3$ from state $S$ gives expected return 8, but the current policy recommends $A_1$ which only gives 5, then the policy is not optimal—we should switch to $A_3$.

---

#### Policy Iteration: Putting It Together

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

#### Value Iteration: A Shortcut

Value iteration combines policy evaluation and improvement into a single update step using the **Bellman Optimality Equation**:

$$V_{k+1}(s) = \max_a \sum_{s', r} P(s', r | s, a) \left[ r + \gamma V_k(s') \right]$$

This directly computes the optimal value function without explicitly maintaining a policy during iteration.

---

#### Practical Example: Car Rental Problem

| Component | Description |
|-----------|-------------|
| **States** | Number of cars at each of two locations (max 20 each) |
| **Actions** | Cars moved between locations overnight (-5 to +5) |
| **Rewards** | +$10 per car rented, -$2 per car moved |
| **Dynamics** | Requests and returns follow Poisson distributions |

Policy iteration finds the optimal policy: a matrix indicating how many cars to move for each possible state configuration.

---

#### Limitations of Dynamic Programming

| Limitation | Description |
|------------|-------------|
| **Requires Model** | Must know exact transition probabilities $P(s', r \| s, a)$ |
| **Not Learning** | Does not learn from experience—assumes perfect knowledge |
| **Computational Cost** | Full state sweeps required; expensive for large state spaces |
| **Impractical for Real Problems** | Transition probabilities often impossible to obtain (e.g., chess, robotics) |

> Despite these limitations, DP provides the **theoretical foundation** for understanding Monte Carlo and Temporal Difference methods, which overcome these drawbacks by learning from experience.

### B. Model-Free Methods

Model-free methods are preferred because they **do not require a model of the environment**. They learn directly through **raw experience**.

#### 1. Monte Carlo (MC) Methods

MC methods learn by **simulating a large number of episodes** and collecting raw experience.

- **Process:** The agent calculates the return (sum of discounted rewards) for all visited states and actions. The final estimate of the action value function ($Q(s, a)$) is the **average of all the returns** received for that state-action pair across all episodes [85, 135:59].
- **Intuition:** MC is like updating your knowledge only after the _entire_ episode is complete. You must wait to receive all future rewards before updating your value function estimate.
- **Action Selection:** MC uses the estimated $Q(s, a)$ values to determine the policy, typically employing an **epsilon greedy policy** to balance exploitation (choosing the action with maximum $Q$ value) and exploration (taking random actions). Exploration is necessary to ensure the agent encounters all states and actions, preventing it from missing an optimal path.

#### 2. Temporal Difference (TD) Methods

Temporal Difference methods are considered the **"best of both worlds"**. They combine learning from raw experience (like MC) with updating estimates based on other learned estimates (like DP). TD methods are highly useful in practice due to their efficiency.

- **Efficiency Intuition:** Unlike MC, which waits for the entire episode to complete, TD makes **incremental updates after each step**. This is similar to how humans think and make updates—incrementally, rather than waiting until the final outcome.
- **Mechanism:** TD solves the challenge of waiting for the final return by **approximating the return** using the immediate reward plus the discounted estimated value of the next state. This uses the structure of the Bellman equation to define the update target (the "TD target").
- **Algorithms:** Q-learning and SARSA are prominent examples of TD methods. These algorithms efficiently update the action value function ($Q(s, a)$). Q-learning, for instance, uses a maximum operation derived from the Bellman optimality equation when calculating the update target.
