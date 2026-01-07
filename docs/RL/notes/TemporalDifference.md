# Temporal Difference (TD) Methods

Temporal Difference methods combine the best of both **Dynamic Programming** and **Monte Carlo** methods.

---

## Why TD Methods?

| Method | Requires Model? | Learns from Experience? | Waits for Episode End? | Uses Bootstrapping? |
|--------|-----------------|------------------------|------------------------|---------------------|
| Dynamic Programming | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| Monte Carlo | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Temporal Difference** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |

**Key Insight**: TD methods learn from raw experience (like MC) but update estimates based on other learned estimates without waiting for the episode to end (like DP).

---

## The Monte Carlo Update (Review)

In Monte Carlo, we update value functions only after the episode completes:

$$V(S) \leftarrow V(S) + \alpha \left[ G_t - V(S) \right]$$

Where:
- $G_t$ = actual return (sum of all rewards until episode ends)
- $\alpha$ = step size parameter (learning rate)
- $G_t - V(S)$ = difference between actual return and current estimate

**Problem**: We must wait until the episode ends to know $G_t$.

**Intuition**: 
- New estimate = Old estimate + α × (Return − Old estimate)
- We're "nudging" the old estimate towards the actual return
- If α = 1, new estimate would equal the return completely

---

## The TD Update

Instead of waiting for the full return $G_t$, TD methods approximate it:

$$V(S) \leftarrow V(S) + \alpha \left[ R_{t+1} + \gamma V(S') - V(S) \right]$$

Where:
- $R_{t+1}$ = immediate reward
- $\gamma V(S')$ = discounted estimate of next state's value
- $R_{t+1} + \gamma V(S')$ = **TD Target**
- $R_{t+1} + \gamma V(S') - V(S)$ = **TD Error**

**Key Difference**:
- MC uses actual return: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$
- TD approximates it: $R_{t+1} + \gamma V(S')$

The value function $V(S')$ serves as a proxy for all future rewards!

---

## Why This Works

```
Full Return (MC):     G_t = R₁ + γR₂ + γ²R₃ + γ³R₄ + ...
                           ↓
TD Approximation:     R₁ + γ × V(S')
                           └── Estimate of (R₂ + γR₃ + γ²R₄ + ...)
```

Since $V(S')$ is defined as the expected value of all future returns from state $S'$, it's a reasonable approximation of what we'd get if we waited until the end.

---

## TD Prediction Algorithm (TD(0))

Given a policy π, estimate $V_\pi$ for all states:

```
Initialize V(s) = 0 for all states

For each episode:
    Initialize S (starting state)
    
    For each step of episode:
        A ← action given by π for S
        Take action A, observe R, S'
        
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        
        S ← S'
    Until S is terminal
```

**Bootstrapping**: We use the old estimate $V(S')$ to update $V(S)$ — this is similar to DP where we use estimates to update other estimates.

---

## Intuitive Example: Driving Home

Imagine you're driving home from work through junctions 1 → 2 → 3 → ... → Home.

**Question**: How long does it take from junction 3 to home?

### Monte Carlo Approach:
- Wait until you actually reach home
- Calculate total time from junction 3 to home
- Update your estimate

**Problem**: If there's a traffic jam between junction 3 and 4, you can't update your estimate until you get home!

### TD Approach:
- As soon as you reach junction 4, update immediately
- Use: `Time(3→4) + γ × Estimated_time(4→Home)`

**Advantage**: If your friend asks "how long from junction 3?", you can answer immediately without waiting to reach home!

---

## Random Walk Problem

A classic problem to compare MC vs TD:

```
     ←  A  ←→  B  ←→  C  ←→  D  ←→  E  →
   [0]                              [+1]
```

- Start at C (center)
- Move left or right with equal probability
- Left terminal state: reward = 0
- Right terminal state: reward = +1
- All other transitions: reward = 0

### True Values:
| State | A | B | C | D | E |
|-------|---|---|---|---|---|
| True V | 1/6 | 2/6 | 3/6 | 4/6 | 5/6 |

**Intuition**:
- E is closest to +1 reward → highest value
- A is farthest from +1 reward → lowest value
- C is equidistant → value = 0.5

### Results:
- **TD(0) converges faster** and more accurately
- TD methods better approximate the true value line
- Lower RMS error with fewer episodes

---

## TD vs MC: Key Differences

| Aspect | Monte Carlo | Temporal Difference |
|--------|-------------|---------------------|
| **Update timing** | After episode ends | After each step |
| **Target** | Actual return $G_t$ | $R + \gamma V(S')$ |
| **Bootstrapping** | No | Yes |
| **Variance** | High (depends on entire trajectory) | Lower |
| **Bias** | Unbiased | Biased (uses estimates) |
| **Works for** | Episodic tasks only | Episodic and continuing tasks |

---

## The Step Size Parameter (α)

The parameter α controls how much we update:

- **α = 0**: No learning (ignore new information)
- **α = 1**: Completely replace old estimate with new target
- **0 < α < 1**: Blend old estimate with new information

**Typical values**: 0.01 to 0.5

**Effect of α**:
- Higher α → Faster learning, but more noisy
- Lower α → Slower learning, but more stable

---

## Summary

1. **TD methods combine MC and DP**:
   - Like MC: Learn from experience, no model required
   - Like DP: Bootstrap using estimates, don't wait for episode end

2. **TD Target**: $R_{t+1} + \gamma V(S')$ approximates the full return

3. **Bootstrapping**: Using $V(S')$ to update $V(S)$

4. **Advantages over MC**:
   - Update after each step (no waiting)
   - Works for continuing tasks
   - Often faster convergence

5. **Key parameters**:
   - α (step size): How much to update
   - γ (discount): How much to value future rewards

> **Next**: Control problem — using TD methods to find optimal policies (Q-Learning, SARSA)
