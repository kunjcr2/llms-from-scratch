# Value Function Approximation

In classical RL (Dynamic Programming, Monte Carlo, Temporal Difference), we use **tabular methods**—storing a value for each state in a table. This works for small state spaces but becomes impractical for real-world problems.

---

## The Problem with Tabular Methods

| Aspect | Description |
|--------|-------------|
| **Memory** | Must store values for every possible state |
| **Computation** | Must update and visit every state |
| **Scale** | Chess has ~$10^{46}$ states—impossible to enumerate |

**Solution**: Instead of storing values in a table, we **approximate** the value function using a parameterized function that can generalize from visited states to unvisited ones.

---

## The Core Idea

In tabular methods, we write $V(s)$—a lookup table.

In function approximation, we write $\hat{V}(s, \mathbf{w})$—a function parameterized by weights $\mathbf{w}$.

| Notation | Meaning |
|----------|---------|
| $\hat{V}$ | Approximate value function (hat indicates approximation) |
| $s$ | State |
| $\mathbf{w}$ | Weight vector (learnable parameters) |

**Goal**: Find weights $\mathbf{w}$ such that $\hat{V}(s, \mathbf{w}) \approx V_\pi(s)$ for all states.

---

## The Objective: Mean Squared Value Error

We want to minimize the error between our approximation and the true value function:

$$\overline{VE}(\mathbf{w}) = \sum_s \mu(s) \left[ V_\pi(s) - \hat{V}(s, \mathbf{w}) \right]^2$$

Where:
- $V_\pi(s)$ = true value function
- $\hat{V}(s, \mathbf{w})$ = our approximation
- $\mu(s)$ = **state distribution** (how often state $s$ is visited)

### Why Weight by $\mu(s)$?

Not all states are equally important. If the agent visits state A 100 times but state B only once, errors at state A matter more.

```
State Distribution Example:

Probability
    │     ●
    │    ╱ ╲
    │   ╱   ╲
    │  ╱     ╲
    │ ╱       ╲
    └──────────────► State
       -1   0   1

State 0 is visited most frequently → prioritize accurate estimates here
```

By multiplying by $\mu(s)$, we focus the optimization on frequently-visited states.

---

## Gradient Descent: Finding Optimal Weights

To minimize $\overline{VE}(\mathbf{w})$, we use **gradient descent**:

### Intuition

Imagine standing on a hillside and wanting to reach the valley (minimum). Gradient descent says:
1. Look around to find the steepest downhill direction
2. Take a small step in that direction
3. Repeat until you reach the bottom

### The Update Rule

Starting from the objective function and applying gradient descent:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

Where:
- $\alpha$ = learning rate (step size)
- $V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}_t)$ = error (how far off we are)
- $\nabla \hat{V}(S_t, \mathbf{w}_t)$ = gradient (direction to move)

**Intuition**: Move the weights in the direction that reduces the error, scaled by how large the error is.

---

## Problem 1: The Target is Unknown

In supervised learning, we have labeled data: $(x, y)$ pairs where $y$ is known.

In RL, we don't know $V_\pi(s)$—that's precisely what we're trying to learn!

**Solution**: Replace the true value with an **approximation target**.

### Monte Carlo Target

Use the actual return $G_t$ from experience:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ G_t - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

This works because $\mathbb{E}[G_t] = V_\pi(S_t)$—the expected return equals the true value.

### TD Target

Use the TD estimate $R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w})$:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w}_t) - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

---

## Problem 2: Semi-Gradient Methods

When using the TD target, there's a mathematical subtlety.

The gradient of the squared error should include the gradient of both terms:

$$\nabla \left[ V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}) \right]^2$$

With the TD target, $\hat{V}(S_{t+1}, \mathbf{w})$ also depends on $\mathbf{w}$, so the full gradient would be:

$$\nabla \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w}) - \hat{V}(S_t, \mathbf{w}) \right]^2$$

This requires computing $\nabla \hat{V}(S_{t+1}, \mathbf{w})$ as well.

**In practice**: We ignore the gradient through the target (treat it as a constant).

This gives us **semi-gradient methods**—"semi" because we only compute part of the true gradient.

---

## Why RL is Different from Supervised Learning

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|---------------------|------------------------|
| **Target** | Known beforehand $(x, y)$ pairs | Unknown until interaction |
| **Target stability** | Fixed | Can change as learning progresses |
| **Data** | IID samples | Correlated sequential data |

These differences make RL harder but still tractable with the right approximations.

---

## Summary

1. **Tabular methods don't scale**: Real problems have enormous state spaces ($10^{46}$ for chess)

2. **Function approximation**: Replace $V(s)$ lookup table with parameterized $\hat{V}(s, \mathbf{w})$

3. **Objective**: Minimize weighted mean squared error $\overline{VE}(\mathbf{w})$

4. **State distribution $\mu(s)$**: Focus learning on frequently-visited states

5. **Gradient descent**: Update weights in direction that reduces error

6. **Target approximations**: Use Monte Carlo returns ($G_t$) or TD targets ($R + \gamma \hat{V}$) since true values unknown

7. **Semi-gradient methods**: Ignore gradient through TD target for practical computation

---

## Linear Methods

The simplest approach to function approximation uses a **linear combination** of features.

### The Linear Value Function

$$\hat{V}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^{d} w_i x_i(s)$$

Where:
- $\mathbf{w} = [w_1, w_2, \ldots, w_d]^\top$ = weight vector
- $\mathbf{x}(s) = [x_1(s), x_2(s), \ldots, x_d(s)]^\top$ = feature vector for state $s$

### Example with Two Features

If $\mathbf{w} = [w_1, w_2]$ and $\mathbf{x}(s) = [x_1(s), x_2(s)]$:

$$\hat{V}(s, \mathbf{w}) = w_1 \cdot x_1(s) + w_2 \cdot x_2(s)$$

The feature functions $x_1(s), x_2(s)$ are defined beforehand based on domain knowledge. Our job is to learn the weights $w_1, w_2$.

### Gradient for Linear Methods

The gradient is remarkably simple:

$$\nabla \hat{V}(s, \mathbf{w}) = \mathbf{x}(s)$$

Taking the derivative with respect to each weight:
- $\frac{\partial}{\partial w_1}(w_1 x_1 + w_2 x_2) = x_1(s)$
- $\frac{\partial}{\partial w_2}(w_1 x_1 + w_2 x_2) = x_2(s)$

### Update Rules for Linear Methods

**Monte Carlo (Linear)**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ G_t - \hat{V}(S_t, \mathbf{w}_t) \right] \mathbf{x}(S_t)$$

**Temporal Difference (Linear)**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w}_t) - \hat{V}(S_t, \mathbf{w}_t) \right] \mathbf{x}(S_t)$$

### Feature Vector Design

Different problems require different feature representations:

| Problem Type | Feature Choice |
|-------------|---------------|
| Periodic functions | Fourier basis (sin, cos) |
| Localized states | Tile coding / Coarse coding |
| Spatial problems | Radial basis functions |

For deeper treatment of feature engineering, see Sutton & Barto, Chapter 9.5.

### Advantages of Linear Methods

- Simple mathematical analysis
- Gradient computation is straightforward ($\nabla \hat{V} = \mathbf{x}(s)$)
- Convergence guarantees under certain conditions
- Computationally efficient

---

## Nonlinear Methods: Neural Networks

For many complex problems, linear methods are insufficient. The function we want to approximate may be highly nonlinear.

### Why Neural Networks?

Neural networks can approximate arbitrarily complex functions by stacking layers of nonlinear transformations.

```
Architecture Overview:

Input Layer     Hidden Layers      Output
(States)                          (Value)

  [s₁] ─────┐
            ├──→ [h₁] ──┐
  [s₂] ─────┤           ├──→ [h₄] ──→ V̂(s,w)
            ├──→ [h₂] ──┤
  [s₃] ─────┤           ├──→ [h₅]
            ├──→ [h₃] ──┘
  [s₄] ─────┘

     w₁,w₂,...      w₇,w₈,...     w₁₃,...
```

- **Input**: State representation (could be raw pixels, features, etc.)
- **Hidden Layers**: Nonlinear transformations with learnable weights
- **Output**: Estimated value $\hat{V}(s, \mathbf{w})$

### Deep Reinforcement Learning

When neural networks are used to approximate value functions, we call it **Deep Reinforcement Learning** (Deep RL).

| System | Application |
|--------|-------------|
| AlphaGo | Beat human Go champion using neural network value/policy approximation |
| RLHF | Reinforcement Learning from Human Feedback for LLM alignment |
| DeepSeek | Uses GRPO algorithm with neural network approximators |

### Gradient Computation

For neural networks, we use **backpropagation** to compute $\nabla \hat{V}(S_t, \mathbf{w})$. Modern frameworks (PyTorch, TensorFlow) handle this automatically.

The same update rules apply:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ \text{Target} - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

The only difference is that $\nabla \hat{V}$ is computed via backpropagation rather than a simple feature vector.

---

## Control: Finding the Optimal Policy

So far, we've discussed **prediction**—estimating $V_\pi$ for a given policy. Now we address **control**—finding the optimal policy $\pi^*$.

### Using Action-Value Functions

For control, we work with action-value functions $Q(s, a)$ rather than state-value functions $V(s)$.

$$\hat{Q}(s, a, \mathbf{w}) \approx Q_\pi(s, a)$$

**Why?** To select the best action in state $s$, we compute $\hat{Q}(s, a, \mathbf{w})$ for all actions and pick the one with the highest value.

### Greedy Policy

The **greedy policy** selects the action with the maximum Q-value:

$$\pi(s) = \arg\max_a \hat{Q}(s, a, \mathbf{w})$$

### Epsilon-Greedy Policy

Pure greedy selection doesn't explore. The **epsilon-greedy** policy balances exploration and exploitation:

$$\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} \hat{Q}(s, a', \mathbf{w}) \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

In practice:
- With probability $1 - \epsilon$: Choose the greedy action (exploit)
- With probability $\epsilon$: Choose a random action (explore)

### Example

Suppose $\epsilon = 0.05$ and we have 4 actions with estimated Q-values:

| Action | $\hat{Q}(s, a, \mathbf{w})$ |
|--------|---------------------------|
| $a_1$  | 20 |
| $a_2$  | 30 (maximum) |
| $a_3$  | 10 |
| $a_4$  | 15 |

- **95% of the time**: Select $a_2$ (greedy)
- **5% of the time**: Select randomly among $a_1, a_2, a_3, a_4$

This ensures the agent occasionally tries suboptimal actions, discovering potentially better strategies.

---

## SARSA with Function Approximation

The control algorithm combines:
1. Epsilon-greedy action selection
2. TD weight updates after each transition

### The Update Process

```
State-Action Transition:

    S ──(A)──→ R ──→ S' ──(A')──→ ...
    
    At each step, update weights using:
    - Current state-action: (S, A)
    - Reward: R
    - Next state-action: (S', A')
```

### Weight Update Rule

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{Q}(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat{Q}(S_t, A_t, \mathbf{w}_t) \right] \nabla \hat{Q}(S_t, A_t, \mathbf{w}_t)$$

### Algorithm: Episodic Semi-Gradient SARSA

```
Initialize weights w arbitrarily
For each episode:
    Initialize S
    Choose A from S using ε-greedy based on Q̂(s, ·, w)
    
    For each step of episode (until S is terminal):
        Take action A, observe R, S'
        Choose A' from S' using ε-greedy based on Q̂(s', ·, w)
        
        # Update weights
        δ ← R + γ Q̂(S', A', w) - Q̂(S, A, w)
        w ← w + α δ ∇Q̂(S, A, w)
        
        S ← S'
        A ← A'
```

### Key Points

1. **Immediate updates**: Weights update after every transition (not waiting for episode end)
2. **Bootstrapping**: Uses estimated Q-values for the next state-action pair
3. **On-policy**: The same ε-greedy policy generates behavior and is being improved
4. **Semi-gradient**: Ignores gradient through the target $\hat{Q}(S', A', \mathbf{w})$

---

## Summary

1. **Linear Methods**: Use $\hat{V}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$ with gradient $\nabla \hat{V} = \mathbf{x}(s)$

2. **Feature vectors**: Domain-specific choices (Fourier, tile coding, radial basis functions)

3. **Nonlinear Methods**: Neural networks for complex function approximation → **Deep RL**

4. **Control problem**: Find optimal policy using action-value function $\hat{Q}(s, a, \mathbf{w})$

5. **Epsilon-greedy policy**: Explore with probability $\epsilon$, exploit with probability $1 - \epsilon$

6. **SARSA with function approximation**: Update weights after each $(S, A, R, S', A')$ transition

7. **Iterative improvement**: As weights improve → Q-values improve → policy improves → better actions → better experience → better weight updates

> **Next**: Deep Q-Networks (DQN) and techniques for stable deep reinforcement learning.
