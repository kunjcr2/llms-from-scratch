# Monte Carlo Methods

Monte Carlo (MC) methods are the first **learning-based** approach in RL. The key shift from Dynamic Programming: instead of needing a perfect model of the environment, the agent **learns by doing**—running episodes, observing what happens, and averaging up the results.

> **Analogy:** DP is like a chess grandmaster who has memorized every possible board position. MC is like a player who has never read a book but has played 10,000 games—they learn what positions tend to win by *experiencing* wins and losses, not from a formula.

---

## Why MC? (The Gap DP Leaves)

DP requires knowing $P(s', r \mid s, a)$—the exact probability of every transition. That's fine on paper but almost impossible in practice:

- **Chess:** You can't pre-compute every outcome before playing a single game.
- **Robotics:** The robot must learn what "slippery floor" means by slipping.
- **Finance:** Market dynamics aren't given to you by a textbook.

MC's answer: **skip the model entirely**. Run episodes, measure the actual return from each state, and average. The Law of Large Numbers guarantees these averages converge to the true $V_\pi(s)$.

---

## The Core Idea: Average Actual Returns

For any state $s$, the true value $V_\pi(s)$ is the *expected* return starting from $s$ under policy $\pi$. MC estimates this by taking the *empirical* mean:

$$V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i^{(s)}$$

where $G_i^{(s)}$ is the actual discounted return observed from state $s$ in episode $i$.

More episodes → better estimate. Simple, but powerful.

---

## MC Prediction (Estimating $V_\pi$)

**Goal:** Evaluate a fixed policy $\pi$ — how good is each state under this policy?

**Algorithm — First-Visit MC:**

```
Initialize:
    Returns(s) ← empty list, for all s
    V(s) ← 0, for all s

Loop (for each episode):
    Generate a full episode: S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ
    G ← 0

    For t = T-1 downto 0:
        G ← γ·G + Rₜ₊₁                    ← accumulate discounted return (backwards)
        If Sₜ not visited earlier in episode:
            Append G to Returns(Sₜ)
            V(Sₜ) ← mean(Returns(Sₜ))
```

**Why backwards?** Walking the episode in reverse lets us compute each state's return in one pass — no need to re-sum rewards from scratch for every state.

**First-visit vs Every-visit:**
- **First-visit MC**: Only counts $s$ the first time it appears in an episode. Unbiased, widely used.
- **Every-visit MC**: Counts every visit. Biased for finite samples, but often lower variance in practice.

> **Key constraint:** MC requires **complete episodes**. You must reach a terminal state before updating anything. This makes it unsuitable for continuous (non-episodic) tasks.

---

## Why Q-values, Not Just V?

State values $V(s)$ answer: *"How good is this state?"*

But to **improve a policy**, you need: *"Which action should I take from this state?"*

Without a model, you can't derive the best action from $V(s)$ alone—you'd need $P(s', r \mid s, a)$ to compare actions. So MC focuses on **action-values**:

$$Q(s, a) = \text{expected return starting from } s, \text{ taking action } a, \text{ then following } \pi$$

Same algorithm, but track $(s, a)$ pairs instead of just $s$:

```
For each (state, action) pair in episode:
    G ← return from that point to episode end
    Append G to Returns(s, a)
    Q(s, a) ← mean(Returns(s, a))
```

Once you have $Q$, the greedy policy is trivially: $\pi(s) = \arg\max_a Q(s, a)$ — no model needed.

---

## MC Control (Finding the Optimal Policy)

Combine prediction and improvement in a loop, just like policy iteration in DP:

```
Start with any policy π₀
        │
        ▼
┌─────────────────────────────────┐
│  MC Prediction                  │  ← Run episodes, estimate Q_π
│  (many episodes)                │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Policy Improvement             │  ← π(s) = argmax_a Q(s, a)
└───────────────┬─────────────────┘
                │
        Policy stable? ──── No ──→ (loop back)
                │
               Yes
                ▼
           π* found ✓
```

The guarantee: each improvement step makes the policy at least as good (Policy Improvement Theorem holds here too). With enough episodes, this converges to $\pi^*$.

---

## The Exploration Problem

There's a catch: if the policy always picks the greediest action, some state-action pairs will **never be visited** and their Q-values will never be updated.

**Example:** State $S$ has actions Up, Down, Left, Right. First episode: Up gives +5. The greedy policy locks onto Up forever. But Right might give +10—we'll never know.

### Solution 1: Exploring Starts
Randomly initialize the starting $(s_0, a_0)$ pair each episode so every pair eventually gets sampled. Works in theory, impractical in real environments where you can't force arbitrary starts.

### Solution 2: ε-Greedy (Practical)

```python
def epsilon_greedy(Q, state, epsilon):
    if random() < epsilon:
        return random_action()      # Explore — try something new
    else:
        return argmax(Q[state])     # Exploit — use current best knowledge
```

With probability $\epsilon$ the agent tries a random action; otherwise it acts greedily. This ensures every action gets visited infinitely often (in the limit), satisfying the **requirement for convergence**.

**Tuning $\epsilon$:**

| ε value | Behavior |
|---------|----------|
| 0 (pure greedy) | Converges fast but gets stuck in local optima |
| 0.1 | Good balance — standard starting point |
| 0.5+ | Too much noise, slow convergence |

> A common trick: **decay ε over time** — explore heavily early on, exploit more as Q-values stabilize.

---

## On-Policy vs Off-Policy

| | On-Policy | Off-Policy |
|---|---|---|
| **Data source** | The policy being improved | A separate *behavior* policy μ |
| **Example** | ε-greedy MC control | Learning from human demos |
| **Pros** | Simple, stable | Can reuse old data; can learn optimal while exploring |
| **Cons** | Must balance explore/exploit in the same policy | Requires importance sampling correction |

### Off-Policy: Importance Sampling

When the behavior policy $\mu$ and target policy $\pi$ disagree on probabilities, raw returns are biased. We correct with the **importance sampling ratio**:

$$\rho_t = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{\mu(A_k \mid S_k)}$$

Intuitively: if $\pi$ would have taken the same actions as $\mu$, $\rho = 1$ (no correction needed). If $\pi$ is much more likely to take those actions, the return counts more. If much less likely, it counts less.

**Two flavors:**
- **Ordinary IS:** Average weighted returns. Unbiased, but can have very high variance.
- **Weighted IS:** Normalize by sum of weights. Biased but drastically lower variance — usually preferred in practice.

---

## Connection to Multi-Armed Bandits

MC is the natural extension of multi-armed bandits to sequential decision-making:

| Multi-Armed Bandit | Monte Carlo |
|-------------------|-------------|
| Single state, $K$ levers | $N$ states, $K$ actions each |
| Pull lever → immediate reward | Take action → episode return |
| Average rewards per lever | Average returns per $(s, a)$ |
| ε-greedy exploration | ε-greedy exploration |

The difference: in bandits, one pull = one data point. In MC, one episode gives data points for every state-action pair visited along the way.

---

## Limitations

| Limitation | Why It Matters |
|------------|----------------|
| **Episodic only** | Can't apply to continuous tasks (e.g., ongoing stock trading) |
| **High variance** | A single unlucky episode can swing estimates wildly |
| **Slow to propagate** | State A's value can't improve until an episode *starting from A* completes |
| **Waits for episode end** | No mid-episode updates — wastes information available earlier |

> **Up next:** Temporal Difference methods fix the last two problems. They **bootstrap**—updating value estimates mid-episode using other estimates, like DP—while still learning from experience, like MC. Best of both worlds.
