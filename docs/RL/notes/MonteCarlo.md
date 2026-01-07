# Monte Carlo Methods

Monte Carlo (MC) methods are our first **learning-based** approach to reinforcement learning. Unlike Dynamic Programming, they **do not require a model of the environment**.

---

## Why Monte Carlo?

### Dynamic Programming Limitation
In DP, we need to know the transition probabilities $P(s', r | s, a)$ — the complete model of the environment.

**Problem**: In most real-world cases, you don't have this model!
- Teaching an agent to play chess: How do you know exact probabilities before playing?
- Mars Rover: You don't know the Martian environment until you experience it

### Monte Carlo Solution
Instead of requiring a model, MC methods **learn from experience**:
- Interact with the environment
- Observe what happens
- Update estimates based on actual outcomes

> **Key Insight**: Monte Carlo methods require only **experience** — no prior knowledge of environment dynamics.

---

## MC vs Dynamic Programming

| Aspect | Dynamic Programming | Monte Carlo |
|--------|---------------------|-------------|
| **Model Required?** | ✅ Yes (transition probabilities) | ❌ No |
| **Learning Type** | Mathematical computation | Learning from experience |
| **When to Use** | Known environment model | Unknown environment |
| **Bootstrapping** | ✅ Yes | ❌ No |
| **Update Timing** | Iterative sweeps | After episode completion |

---

## How Monte Carlo Works

### The Core Idea

1. **Generate episodes** by interacting with the environment
2. For each state visited, calculate the **actual return** (sum of rewards until episode ends)
3. **Average** these returns across many episodes
4. The average converges to the **true value function**

### Example: Grid World

```
┌───┬───┬───┬───┐
│ S │   │   │   │    S = Start
├───┼───┼───┼───┤    G = Goal
│   │ X │   │   │    X = Obstacle
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │ G │
└───┴───┴───┴───┘
```

**Episode 1**: S → right → down → hit X (negative reward) → ... → G
**Episode 2**: S → down → down → right → ... → G

For each state visited, calculate the return from that state to the goal, then average across episodes.

---

## Monte Carlo Prediction

**Goal**: Given a policy π, estimate the value function $V_\pi(s)$ for all states.

### Algorithm

```
Initialize:
    Returns(s) ← empty list, for all states s
    V(s) ← 0, for all states s

For each episode:
    Generate episode following policy π
    
    For each state s in episode:
        G ← return from state s to end of episode
        Append G to Returns(s)
        V(s) ← average(Returns(s))
```

### Update Rule

After each episode, for each visited state:

$$V(S) = \frac{1}{N} \sum_{i=1}^{N} G_i$$

Where $N$ is the number of times state $S$ was visited across all episodes.

---

## Action Value Functions (Q-values)

### Why Q-values?

State value $V(s)$ tells us: "How good is this state?"

But we need: "Which action should I take?"

**Action value function** $Q(s, a)$ answers: "How good is taking action $a$ in state $s$?"

### Estimating Q-values

Same as V(s), but track state-action pairs:

```
For each episode:
    For each (state, action) pair in episode:
        G ← return from that point to end
        Append G to Returns(s, a)
        Q(s, a) ← average(Returns(s, a))
```

---

## Monte Carlo Control

**Goal**: Find the optimal policy.

### Policy Iteration (MC Version)

```
┌─────────────────────────────────────────────────────────┐
│  1. Start with arbitrary policy π₀                      │
│                          ↓                               │
│  2. Policy Evaluation: Estimate Q_π using MC             │
│                          ↓                               │
│  3. Policy Improvement: π(s) = argmax_a Q(s,a)          │
│                          ↓                               │
│  4. Repeat until convergence                             │
└─────────────────────────────────────────────────────────┘
```

---

## The Exploration Problem

### The Issue
If we always take the "best" action (greedy), we might never explore other options!

**Example**: 
- State S has actions: Up, Down, Left, Right
- First episode: Up gives reward +5
- Agent always picks Up, never tries other actions
- But maybe Right gives +10!

### Solution: Epsilon-Greedy Policy

With probability $\epsilon$: take **random** action (explore)
With probability $1-\epsilon$: take **best** action (exploit)

```python
def epsilon_greedy(Q, state, epsilon):
    if random() < epsilon:
        return random_action()      # Explore
    else:
        return argmax(Q[state])     # Exploit
```

### Why It Works

From multi-armed bandit experiments:

| Policy | Long-term Performance |
|--------|----------------------|
| Greedy (ε=0) | Poor — misses better options |
| ε=0.1 | Best — balances exploration/exploitation |
| ε=0.5 | Moderate — too much random exploration |

---

## On-Policy vs Off-Policy

### On-Policy
- The policy we're **optimizing** = the policy **generating data**
- Agent learns from its own experience
- Example: ε-greedy MC Control

### Off-Policy
- **Behavior policy** (μ): generates the data
- **Target policy** (π): the one we want to optimize
- Agent learns from someone else's experience

### Importance Sampling

When using off-policy, we need to adjust for the probability difference:

$$\text{Importance Sampling Ratio} = \frac{\pi(a|s)}{\mu(a|s)}$$

If target policy is 10x more likely to take action $a$, then the return should be weighted 10x more.

---

## Connection to Multi-Armed Bandits

Monte Carlo is like having **multiple bandits** — one for each state!

| Multi-Armed Bandit | Monte Carlo |
|-------------------|-------------|
| 1 state, K actions | N states, K actions each |
| Pull lever → get reward | Take action → observe return |
| Average rewards per lever | Average returns per state-action |
| ε-greedy for exploration | ε-greedy for exploration |

---

## Interactive Simulation Example

For a 4×4 grid with goal (+reward) and obstacle (-reward):

1. **Initialize** Q-table with zeros (16 states × 4 actions = 64 values)
2. **Generate episode** following ε-greedy policy
3. **Update Q-values** for each (state, action) visited:
   - Calculate return from that point
   - Average with previous returns
4. **Update policy**: For each state, pick action with max Q-value
5. **Repeat** — colors stabilize as Q-values converge to true values

---

## Key Takeaways

1. **No model required**: MC learns from experience, not environment dynamics

2. **Episode-based updates**: Must wait until episode ends to calculate returns

3. **Averaging returns**: Value estimates = average of observed returns

4. **Exploration is critical**: Use ε-greedy to ensure all actions are tried

5. **Q-values for control**: Action values let us determine optimal policy

6. **On-policy vs Off-policy**: Whether learning from own experience or others'

---

## Limitations

| Limitation | Description |
|------------|-------------|
| **Episodic only** | Must have episodes that terminate |
| **High variance** | Returns can vary wildly between episodes |
| **Slow convergence** | Need many episodes to get accurate estimates |
| **Wait for episode end** | Cannot update mid-episode |

> **Next**: Temporal Difference methods combine MC (learning from experience) with DP (bootstrapping) — best of both worlds!
