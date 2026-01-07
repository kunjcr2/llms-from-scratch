# Value Functions

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
