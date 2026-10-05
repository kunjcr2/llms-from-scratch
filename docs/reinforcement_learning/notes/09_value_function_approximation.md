# Value Function Approximation

In classical RL (Dynamic Programming, Monte Carlo, Temporal Difference), we use **tabular methods**—storing a value for each state in a table. This works for small state spaces but becomes impractical for real-world problems.

---

## The Problem with Tabular Methods

**The Intuition:** Imagine you are trying to estimate the price of houses in a city.
If there are only 5 houses, you could just memorize the exact price of each one. 
**This is like Tabular Methods.** You have a lookup table:
- House 1: $300,000
- House 2: $450,000

But what if the city has 10 million houses? Or what if a house is newly built and you've *never seen it before*?
You can't use a table anymore. It's too big, and it can't guess the price of new houses.

| Aspect | Description / The Tabular Problem | The VFA Solution |
|--------|-----------------------------------|------------------|
| **Memory** | Must store values for every possible state | Store a small set of "rules" (weights) instead of states. |
| **Computation** | Must update and visit every state | Update generalized rules that apply across many states. |
| **Scale** | Chess has ~$10^{46}$ states—impossible to enumerate | **Generalization!** If you learn that "hanging your Queen is bad," you apply that rule to *any* new board state. |

**Solution**: Instead of storing values in a table, we **approximate** the value function using a parameterized function (a "rule of thumb") that can generalize from visited states to unvisited ones.

---

## The Core Idea: Weights and Features

In tabular methods, we write $V(s)$—a lookup table.

In function approximation, we write $\hat{V}(s, \mathbf{w})$—a function parameterized by weights $\mathbf{w}$.

| Notation | Meaning |
|----------|---------|
| $\hat{V}$ | Approximate value function (hat indicates approximation or 'prediction') |
| $s$ | State (e.g., a specific car) |
| $\mathbf{w}$ | Weight vector (learnable rules of thumb, like how much age depreciates a car) |

**Goal**: Find weights $\mathbf{w}$ such that our prediction $\hat{V}(s, \mathbf{w}) \approx V_\pi(s)$ (the true value) for all states.

---

## The Objective: Mean Squared Value Error

We want to minimize the error between our approximation and the true value function:

$$\overline{VE}(\mathbf{w}) = \sum_s \mu(s) \left[ V_\pi(s) - \hat{V}(s, \mathbf{w}) \right]^2$$

Where:
- $V_\pi(s)$ = true value function
- $\hat{V}(s, \mathbf{w})$ = our approximation
- $\mu(s)$ = **state distribution** (how often state $s$ is visited)

### Why Weight by $\mu(s)$?

**The Intuition:** Imagine you are learning how to drive. Is it more important to accurately judge how to brake at a stop sign (happens 100 times a day) or how to reverse on ice (happens once a year)?

Not all states are equally important. By multiplying by $\mu(s)$, we focus the optimization on frequently-visited states. If our rule gets the value slightly wrong on a rare edge-case state we never see, it doesn't really matter.

```text
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

---

## Gradient Descent: Finding Optimal Weights

To minimize $\overline{VE}(\mathbf{w})$, we use **gradient descent**.

### Intuition

Imagine you are blindfolded on a bumpy mountain, and you want to get to the very bottom of the valley (the minimum error).
1. You feel the ground around you to find which direction slopes downwards the steepest. (This is the **Gradient**).
2. You take a small step in that direction. (The size of the step is your **Learning Rate, $\alpha$**).
3. Repeat until you are at the bottom.

### The Update Rule

Starting from the objective function and applying gradient descent:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

Where:
- $\alpha$ = learning rate (step size from our mountain analogy)
- $V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}_t)$ = error (how far off we are from the true value). If our guess is perfect, we don't move! The worse the guess, the bigger the step.
- $\nabla \hat{V}(S_t, \mathbf{w}_t)$ = gradient (Which specific weight caused the mistake? Steer heavily backwards in that direction).

---

## Problem 1: The Target is Unknown

In supervised learning (like predicting house prices on Kaggle), we have labeled data: $(x, y)$ pairs where the final $y$ is known.

In RL, we are navigating an unknown world. **We don't know $V_\pi(s)$**—that's precisely what we're trying to learn!
**Solution**: We fake it. We replace the true value with an **approximation target**.

### Monte Carlo Target (The "Hindsight" Target)

Play out the entire game until the end. Use the actual return $G_t$ from experience as the target:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ G_t - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

This works because $\mathbb{E}[G_t] = V_\pi(S_t)$—the expected return equals the true value.

### TD Target (The "Guess of a Guess")

Take just ONE step, get a reward $R$, and look at where you landed. Use the TD estimate $R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w})$ as the target:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w}_t) - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

**Wait, we update our guess using our own guess?** Yes! This is **bootstrapping**. It feels crazy, like pulling yourself up by your own shoelaces, but it works practically because the immediate, real-world reward $R$ grounds the future guess in reality.

---

## Problem 2: Semi-Gradient Methods

When using the TD target, there's a mathematical subtlety.

The gradient of the squared error should technically include the gradient of both terms:
$$\nabla \left[ V_\pi(S_t) - \hat{V}(S_t, \mathbf{w}) \right]^2$$

With the TD target, $\hat{V}(S_{t+1}, \mathbf{w})$ also depends on $\mathbf{w}$, so the full gradient would absolutely have to be:
$$\nabla \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \mathbf{w}) - \hat{V}(S_t, \mathbf{w}) \right]^2$$

This requires computing $\nabla \hat{V}(S_{t+1}, \mathbf{w})$ as well.

**The Intuition:** If we are standing on the mountain (Gradient Descent) trying to walk toward a target, but the target itself moves every time we take a step (because the target uses $\mathbf{w}$ too), the math breaks down.

**The Solution:** We cheat. 
When calculating the downhill slope, we pretend the Target is a fixed, painted mark on the ground. We ignore the gradient through the target entirely (treat it as a constant).

This gives us **semi-gradient methods**—"semi" because we only compute part of the true gradient. It's mathematically unpure, but it works wonderfully in practice.

---

## Why RL is Different from Supervised Learning

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|---------------------|------------------------|
| **Target** | Known beforehand $(x, y)$ pairs | Unknown until interaction |
| **Target stability** | Fixed | Can change as learning progresses (Bootstrapping) |
| **Data** | IID samples | Correlated sequential data |

These differences make RL harder but still tractable with the right approximations.

---

## Linear Methods

The simplest approach to function approximation uses a **linear combination** of features, exactly like our rule-of-thumb house pricing.

### The Linear Value Function

$$\hat{V}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) = \sum_{i=1}^{d} w_i x_i(s)$$

Where:
- $\mathbf{w} = [w_1, w_2, \ldots, w_d]^\top$ = weight vector
- $\mathbf{x}(s) = [x_1(s), x_2(s), \ldots, x_d(s)]^\top$ = feature vector for state $s$

### Example with Two Features

If $\mathbf{w} = [w_1, w_2]$ and $\mathbf{x}(s) = [x_1(s), x_2(s)]$ where $x_1$ is Age and $x_2$ is Miles:

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
Different problems require different feature representation architectures:
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

## Nonlinear Methods: Enter Neural Networks

A linear equation is great, but what if you're trying to play a video game from raw pixels? Pixels don't have a simple linear relationship to "winning" or "losing". The function we want to approximate may be highly nonlinear.

### Why Neural Networks?

Neural networks can approximate arbitrarily complex functions by stacking layers of nonlinear transformations. Instead of us manually telling the agent what features to look for (like "Age" or "Miles"), the hidden layers *learn the features themselves* directly from the raw state.

```text
The Intuition of a Neural Net in RL:

Input Layer     Hidden Layers                        Output
(Raw Pixels)    (Learned Features)                   (Value)

  [s₁] ──┐      [Layer 1: Finds Edges] ──┐
         ├──→                            ├──→ [h₄] ──→ V̂(s,w)
  [s₂] ──┤      [Layer 2: Finds Shapes] ──┤
         ├──→                           ├──→ [h₅]
  [s₃] ──┤      [Layer 3: Finds "Alien"]──┘
         ├──→                           
  [s₄] ──┘
```

When neural networks are used to approximate value functions, we call it **Deep Reinforcement Learning** (Deep RL). Examples include AlphaGo, ChatGPT's RLHF, and DeepSeek.

### Gradient Computation

For neural networks, we use **backpropagation** to compute $\nabla \hat{V}(S_t, \mathbf{w})$. Modern frameworks (PyTorch, TensorFlow) handle this automatically.

The exact same update rules apply:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ \text{Target} - \hat{V}(S_t, \mathbf{w}_t) \right] \nabla \hat{V}(S_t, \mathbf{w}_t)$$

The only difference is that $\nabla \hat{V}$ is computed via backpropagation rather than pulling a simple feature vector.

---

## Control: Finding the Optimal Policy

So far, we've discussed **prediction**—estimating $V_\pi$ for a given policy. Now we address **control**—finding the optimal policy $\pi^*$.

### Using Action-Value Functions

For control, predicting the value of a state $V(s)$ isn't enough. We need to know the value of taking specific actions. We work with action-value functions $Q(s, a)$.

$$\hat{Q}(s, a, \mathbf{w}) \approx Q_\pi(s, a)$$

**Why?** To select the best action in state $s$, we compute our neural network's $\hat{Q}(s, a, \mathbf{w})$ for every possible action and pick the one with the highest estimated value.

### Greedy Policy

The **greedy policy** selects the action with the maximum Q-value:
$$\pi(s) = \arg\max_a \hat{Q}(s, a, \mathbf{w})$$

### Epsilon-Greedy Policy

If we always pick what we *think* is best right now, we might miss out on a massive reward down a path we just haven't explored yet. The **epsilon-greedy** policy balances exploration and exploitation:

$$\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} \hat{Q}(s, a', \mathbf{w}) \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

In practice:
- With probability $1 - \epsilon$: Choose the greedy action (exploit our knowledge).
- With probability $\epsilon$: Choose a random action (explore the environment).

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

```text
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

```text
Initialize weights w arbitrarily
For each episode:
    Initialize S
    Choose A from S using ε-greedy based on Q̂(s, ·, w)
    
    For each step of episode (until S is terminal):
        Take action A, observe R, S'
        Choose A' from S' using ε-greedy based on Q̂(s', ·, w)
        
        # Calculate Target and Error
        δ ← R + γ Q̂(S', A', w) - Q̂(S, A, w)
        
        # Update weights (backpropagation or gradient descent)
        w ← w + α δ ∇Q̂(S, A, w)
        
        S ← S'
        A ← A'
```

### Key Points

1. **Immediate updates**: Weights update after every transition (not waiting for episode end).
2. **Bootstrapping**: Uses estimated Q-values for the next state-action pair to update the current state-action pair.
3. **On-policy**: The same $\epsilon$-greedy policy generates behavior and is being improved.
4. **Semi-gradient**: Ignores gradient through the target $\hat{Q}(S', A', \mathbf{w})$.

---

## Summary

1. **Tabular methods don't scale**: Real problems have enormous state spaces.
2. **Function approximation**: Replace $V(s)$ lookup table with parameterized "rule of thumb" $\hat{V}(s, \mathbf{w})$.
3. **Linear Methods**: Use $\hat{V}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$ with simple gradient $\nabla \hat{V} = \mathbf{x}(s)$.
4. **Nonlinear Methods**: Neural networks for complex function approximation → **Deep RL**. Learns features automatically.
5. **State distribution $\mu(s)$**: Focus learning on frequently-visited states.
6. **Gradient descent**: Update weights iteratively in the direction that reduces error.
7. **Target approximations**: Use Monte Carlo returns ($G_t$) or TD targets ($R + \gamma \hat{V}$) since true values are unknown.
8. **Semi-gradient methods**: Ignore gradient through the TD target for practical computation.
9. **Control problem**: Find optimal policy using action-value function $\hat{Q}(s, a, \mathbf{w})$.
10. **Epsilon-greedy policy**: Explore with probability $\epsilon$, exploit with probability $1 - \epsilon$.
11. **SARSA with function approximation**: Update weights iteratively as weights improve → policy improves → better experience → better updates.

> **Next**: Deep Q-Networks (DQN) and techniques for stable deep reinforcement learning.
