# Dynamic Programming

Dynamic Programming (DP) is a family of algorithms for solving MDPs when you have **complete knowledge of the environment**—meaning you know exactly how actions affect state transitions and rewards. It is not a *learning* method; it's a *planning* method.

> **Analogy:** Imagine you have a perfect map of a city and know every road's travel time. DP lets you compute the fastest route from any starting point—without ever actually driving. RL methods (Monte Carlo, TD) are for when you *don't* have the map and must figure it out by driving around.

---

## The Core Problem: Why "Iterative"?

We want to find $V_\pi(s)$—the expected return from state $s$ under policy $\pi$. The Bellman equation gives us:

$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma V_\pi(s') \right]$$

The catch: **the value of $s$ depends on the values of its neighbors $s'$**, which are also unknown. For $n$ states you get $n$ equations with $n$ unknowns. You *could* solve this linear system exactly, but that's $O(n^3)$—too expensive for large state spaces.

**The fix:** Start with a guess ($V_0 = 0$ everywhere) and iteratively refine it. Each pass makes the estimates more accurate until they converge.

---

## Step 1: Policy Evaluation (Prediction)

**Question:** How good is this policy?

**Algorithm — Iterative Policy Evaluation:**

1. Initialize $V_0(s) = 0$ for all states $s$
2. Repeat until convergence:
   $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma V_k(s') \right]$$
3. Stop when the maximum change across all states is below a threshold $\theta$

**What's happening intuitively:**  
Each sweep propagates "reward information" backward through the state space. States near high-reward regions start getting higher values. States far away gradually pick up that signal over successive sweeps—like ripples spreading outward from a stone dropped in water.

> **Key detail:** Each iteration does a **full state sweep**—every state is updated once per iteration using the *previous* iteration's values. This is called *synchronous* updates.

---

## Step 2: Policy Improvement

**Question:** Given we know how good each state is, can we act better?

Once we have $V_\pi$, we can compute the **action-value** for each state:

$$Q_\pi(s, a) = \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma V_\pi(s') \right]$$

Then we greedily update the policy:

$$\pi'(s) = \arg\max_a \; Q_\pi(s, a)$$

**Intuition:** Suppose the current policy says "go left" from state $S$, giving expected return 5. But we compute that "go right" gives return 8. Then we update the policy to say "go right." This is guaranteed to be at least as good as the old policy—a result called the **Policy Improvement Theorem**.

> If $\pi' = \pi$ (no state changed its action), the policy is already optimal.

---

## Policy Iteration: The Full Loop

Policy iteration alternates between the two steps above until the policy stops changing:

```
Start with any policy π₀
        │
        ▼
┌───────────────────────────────┐
│  Policy Evaluation            │  ← Compute V_π (how good is π?)
│  (iterate until convergence)  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Policy Improvement           │  ← Update π greedily using V_π
└───────────────┬───────────────┘
                │
        Policy changed? ──── Yes ──→ (loop back)
                │
               No
                │
                ▼
         π* and V* found ✓
```

**Why does this converge?**  
- There are finitely many deterministic policies (at most $|A|^{|S|}$)  
- Each improvement step produces a strictly better policy (or stays the same)  
- So the loop must terminate

---

## Value Iteration: The Shortcut

Policy iteration requires policy evaluation to *fully converge* before each improvement step. That can be wasteful—what if we just do **one sweep** of evaluation, then one improvement step, repeat?

Value iteration merges both steps into a single update using the **Bellman Optimality Equation**:

$$V_{k+1}(s) = \max_a \sum_{s', r} P(s', r \mid s, a) \left[ r + \gamma V_k(s') \right]$$

Notice the $\max$ instead of $\sum_a \pi(a|s)$—we're simultaneously improving *and* evaluating. No explicit policy is maintained until the very end (then we extract $\pi^*$ greedily from $V^*$).

| | Policy Iteration | Value Iteration |
|---|---|---|
| **Evaluation** | Full convergence each step | One sweep per step |
| **Convergence** | Fewer iterations | More iterations |
| **Cost per iter** | Higher | Lower |
| **Best for** | Smaller state spaces | Larger state spaces |

---

## Concrete Example: Jack's Car Rental

| Component | Description |
|-----------|-------------|
| **States** | # of cars at 2 locations (0–20 each) → 441 states |
| **Actions** | Move 0–5 cars between locations overnight |
| **Reward** | +$10 per rental, −$2 per car moved |
| **Dynamics** | Requests/returns are Poisson-distributed |

Policy iteration finds a threshold policy: *"If location A has $x$ more cars than location B, move $k$ cars overnight."* The optimal policy naturally emerges from the value function—no hand-engineering needed.

---

## Limitations

| Limitation | Why It Matters |
|------------|----------------|
| **Requires a model** | Need exact $P(s', r \mid s, a)$—often impossible (e.g., Go, robotics) |
| **Full state sweeps** | Scales poorly: $O(\|S\|^2 \cdot \|A\|)$ per iteration |
| **No learning from data** | Can't improve from real experience—pure planning |

> **Why study DP at all then?** DP establishes the *theoretical baseline*. Monte Carlo and Temporal Difference methods can be understood as approximate DP—they sample from the environment instead of using exact transition probabilities. The Bellman equations remain central to all of them.
