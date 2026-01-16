# KL Divergence (Kullback-Leibler Divergence)

KL Divergence measures **how different one probability distribution is from another**. It's central to TRPO, PPO, and LLM alignment (RLHF).

---

## The Simplest Explanation

Imagine you have two bags of colored balls:

**Bag P (True distribution):**
- 50% Red, 30% Blue, 20% Green

**Bag Q (Approximate distribution):**
- 40% Red, 40% Blue, 20% Green

KL Divergence answers: **"How surprised would I be if I thought the balls came from Q, but they actually came from P?"**

If Q is very different from P → **High KL divergence** → Lots of surprise!
If Q is similar to P → **Low KL divergence** → Little surprise

---

## The Formula

$$D_{KL}(P \| Q) = \sum_x P(x) \cdot \log\frac{P(x)}{Q(x)}$$

Breaking it down:
- **P(x)**: Probability of event x in the TRUE distribution
- **Q(x)**: Probability of event x in the APPROXIMATE distribution
- **log(P/Q)**: How "surprised" we are (log of the ratio)
- **Weighted by P(x)**: Events that happen more often matter more

---

# 🚨🚨🚨 CROSS-ENTROPY LOSS IS KL DIVERGENCE!!! 🚨🚨🚨

## YES, THEY ARE THE SAME THING

When you train a neural network with **Cross-Entropy Loss**, you are **MINIMIZING KL DIVERGENCE**.

Every single time you call `nn.CrossEntropyLoss()` in PyTorch, you're doing KL minimization!

---

## The Math: Why They're The Same

### Cross-Entropy Definition

$$H(P, Q) = -\sum_x P(x) \cdot \log Q(x)$$

### KL Divergence Definition

$$D_{KL}(P \| Q) = \sum_x P(x) \cdot \log\frac{P(x)}{Q(x)}$$

### Let's Expand KL Divergence

$$D_{KL}(P \| Q) = \sum_x P(x) \cdot \log P(x) - \sum_x P(x) \cdot \log Q(x)$$

$$= -H(P) + H(P, Q)$$

### Rearranging:

$$\boxed{H(P, Q) = H(P) + D_{KL}(P \| Q)}$$

Where:
- **H(P, Q)** = Cross-Entropy (what you minimize in training!)
- **H(P)** = Entropy of true distribution (CONSTANT for fixed labels!)
- **D_KL(P || Q)** = KL Divergence

### The Punchline

Since **H(P) is constant** (your labels don't change during training):

$$\text{Minimizing } H(P, Q) = \text{Minimizing } D_{KL}(P \| Q)$$

## 🎯 MINIMIZING CROSS-ENTROPY = MINIMIZING KL DIVERGENCE 🎯

---

## In Classification (Every Neural Network Ever)

```
True label P:     [0, 0, 1, 0, 0]  ← One-hot encoded "cat"
Model output Q:   [0.1, 0.2, 0.6, 0.05, 0.05]  ← Softmax probabilities

Cross-Entropy Loss = -log(0.6) = 0.51
                   ↓
        This is minimizing KL(P || Q)!
        Making Q look more like P!
```

After training:
```
Model output Q:   [0.01, 0.01, 0.95, 0.01, 0.02]  ← Much closer to P!

Cross-Entropy Loss = -log(0.95) = 0.05  ← Lower loss!
                   ↓
        KL divergence is now smaller!
```

---

## In LLM Training (Next Token Prediction)

```python
# What you write:
loss = F.cross_entropy(logits, target_tokens)

# What's actually happening:
# P = one-hot of actual next token
# Q = model's softmax distribution over vocab
# loss = H(P, Q) = D_KL(P || Q) + constant
```

**Every time GPT predicts the next token, it's minimizing KL divergence between its predictions and the true distribution!**

---

## Why We Need Softmax (IMPORTANT!)

KL Divergence and Cross-Entropy only work on **PROBABILITY DISTRIBUTIONS**.

A probability distribution must:
- Have all values between 0 and 1
- Sum to exactly 1.0

**Raw logits from a neural network are GARBAGE for this:**

```
Raw logits from model: [2.5, -1.3, 8.7, 0.2]
                        ↓
        These are just random numbers!
        They don't sum to 1!
        They can be negative!
        
        KL Divergence would be MEANINGLESS here!
```

**Softmax converts logits → probability distribution:**

```
Softmax(logits):      [0.002, 0.00004, 0.995, 0.002]
                        ↓
        Now they sum to 1.0! ✓
        All positive! ✓
        This is a valid probability distribution!
        
        NOW we can compute KL Divergence!
```

### The Softmax Formula

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Why exponential?**
- Makes all values positive (e^x > 0 always)
- Dividing by sum makes them sum to 1
- Preserves ordering (bigger logit = bigger probability)

### In PyTorch

```python
# Raw logits from your model
logits = model(x)  # e.g., [2.5, -1.3, 8.7, 0.2]

# Option 1: Explicit softmax then cross-entropy
probs = F.softmax(logits, dim=-1)
loss = -torch.sum(targets * torch.log(probs))

# Option 2: PyTorch combines them (more numerically stable!)
loss = F.cross_entropy(logits, targets)  # Takes raw logits, does softmax internally
```

> **PyTorch's `cross_entropy` takes RAW LOGITS, not probabilities!** It applies softmax internally for numerical stability.

---

## Python Proof

```python
import numpy as np

def entropy(p):
    """H(P) - entropy of distribution P"""
    p = np.array(p) + 1e-10
    return -np.sum(p * np.log(p))

def cross_entropy(p, q):
    """H(P, Q) - cross entropy"""
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    return -np.sum(p * np.log(q))

def kl_divergence(p, q):
    """D_KL(P || Q)"""
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    return np.sum(p * np.log(p / q))

# True distribution (one-hot for classification)
P = [0, 0, 1, 0]  # True class is index 2
# Model predictions
Q = [0.1, 0.2, 0.6, 0.1]

print(f"Entropy H(P):        {entropy(P):.4f}")
print(f"Cross-Entropy H(P,Q): {cross_entropy(P, Q):.4f}")
print(f"KL Divergence:        {kl_divergence(P, Q):.4f}")
print(f"H(P) + KL:            {entropy(P) + kl_divergence(P, Q):.4f}")
print()
print("✓ Cross-Entropy = H(P) + KL Divergence!")

# Output:
# Entropy H(P):        0.0000  ← One-hot has zero entropy!
# Cross-Entropy H(P,Q): 0.5108
# KL Divergence:        0.5108
# H(P) + KL:            0.5108
# 
# ✓ Cross-Entropy = H(P) + KL Divergence!
```

---

## Worked Example: Coin Flips

### Fair coin (P) vs Biased coin (Q)

**P (fair coin):** Heads = 0.5, Tails = 0.5
**Q (biased coin):** Heads = 0.9, Tails = 0.1

$$D_{KL}(P \| Q) = 0.5 \cdot \log\frac{0.5}{0.9} + 0.5 \cdot \log\frac{0.5}{0.1}$$

$$= 0.5 \cdot \log(0.556) + 0.5 \cdot \log(5)$$

$$= 0.5 \cdot (-0.588) + 0.5 \cdot (1.609)$$

$$= -0.294 + 0.805 = 0.511 \text{ nats}$$

### Same coin (P = Q)

**P:** Heads = 0.5, Tails = 0.5
**Q:** Heads = 0.5, Tails = 0.5

$$D_{KL}(P \| Q) = 0.5 \cdot \log\frac{0.5}{0.5} + 0.5 \cdot \log\frac{0.5}{0.5}$$

$$= 0.5 \cdot \log(1) + 0.5 \cdot \log(1) = 0$$

> **Key insight**: KL divergence is 0 when distributions are identical!

---

## Visual Intuition

```
Distribution P (True)          Distribution Q (Approximation)
    ████                           ██████
    ████                           ██████
    ████  ███                      ██████  ██
    ████  ███  ██                  ██████  ██  ██
    ────────────                   ────────────────
     A    B    C                     A     B    C
    50%  30%  20%                   60%   25%  15%

                    KL Divergence = 0.12
                    (Moderate difference)

Distribution P                 Distribution Q (Very Different!)
    ████████                       ██
    ████████                       ██
    ████████                       ██  ████████
    ────────                       ────────────
      A  B                          A    B
    80% 20%                        10%  90%

                    KL Divergence = 1.47
                    (BIG difference!)
```

---

## Why It's NOT Symmetric

**Important!** KL divergence is NOT a distance metric!

$$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$

### Example:
- P: Common event (99%), Rare event (1%)
- Q: Common event (50%), Rare event (50%)

**KL(P || Q):** "How surprised is Q about P's data?"
- P's common event (99%) happens all the time
- Q thinks it's only 50% → Q is VERY surprised!
- **Result: HIGH**

**KL(Q || P):** "How surprised is P about Q's data?"
- Q's rare event (50%) happens often
- P thinks it's only 1% → P is surprised, but events are rarer
- **Result: DIFFERENT**

---

## Why KL Divergence in RL?

### In TRPO / PPO

When updating a policy from π_old to π_new:

```
Old Policy π_old:           New Policy π_new:
Action A: 70%               Action A: 40%  ← Big change!
Action B: 20%               Action B: 50%  ← Big change!
Action C: 10%               Action C: 10%

KL(π_old || π_new) = HIGH → Policy changed too much!
```

**The Trust Region Constraint:**
$$D_{KL}(\pi_{old} \| \pi_{new}) \leq \delta$$

This says: "Don't let the new policy be TOO different from the old one!"

### Why limit policy changes?

1. **Stability**: Big jumps can cause training to diverge
2. **Sample efficiency**: Old data becomes useless if policy changes too much
3. **LLM safety**: Don't let aligned model drift too far from base model

---

## Real-World Analogy: GPS Navigation

Imagine you're driving with a GPS:

**Old Route (π_old):** Highway → Exit 5 → Main Street → Destination

**If GPS suddenly says (π_new):** Take a U-turn → Go 50 miles north → etc.

You'd be confused! The new route is VERY different from what you expected.
**KL divergence = HIGH**

**Better (π_new with constraint):** Highway → Exit 6 → Oak Street → Destination

Small change, similar route. You can adapt easily.
**KL divergence = LOW** ✓

---

## KL Divergence in LLM Fine-Tuning (RLHF)

```
Base Model (before RLHF):
  "The capital of France is Paris."  → 95% confident
  
Fine-tuned Model (after RLHF):
  "The capital of France is Paris."  → 85% confident  ← Slight change OK
  
Over-tuned Model (bad):
  "The capital of France is Paris."  → 10% confident  ← BROKEN!
  "The capital of France is Cheese." → 60% confident  ← WTF
```

KL divergence constraint prevents the model from forgetting what it knew!

---

## Python Code Example

```python
import numpy as np

def kl_divergence(p, q):
    """
    Calculate KL divergence D_KL(P || Q)
    p, q: probability distributions (arrays that sum to 1)
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return np.sum(p * np.log(p / q))

# Example: Fair coin vs biased coin
fair_coin = [0.5, 0.5]
biased_coin = [0.9, 0.1]

print(f"KL(fair || biased) = {kl_divergence(fair_coin, biased_coin):.4f}")
# Output: 0.5108

# Example: Similar distributions
dist_p = [0.4, 0.3, 0.3]
dist_q = [0.35, 0.35, 0.3]

print(f"KL(P || Q) = {kl_divergence(dist_p, dist_q):.4f}")
# Output: 0.0144 (very small - distributions are similar!)

# Example: Identical distributions
print(f"KL(P || P) = {kl_divergence(dist_p, dist_p):.4f}")
# Output: 0.0000 (zero!)
```

---

## Key Properties Summary

| Property | Value | Meaning |
|----------|-------|---------|
| D_KL(P \|\| P) | 0 | Same distribution = no divergence |
| D_KL(P \|\| Q) | ≥ 0 | Always non-negative |
| D_KL(P \|\| Q) | ≠ D_KL(Q \|\| P) | NOT symmetric! |
| Higher value | → | Distributions are more different |

---

## Summary: The One-Sentence Explanation

> **KL Divergence tells you how much "information" you would lose if you used distribution Q to approximate distribution P.**

In RL terms:
> **KL Divergence tells you how much your new policy has drifted from your old policy — and TRPO/PPO use it to keep that drift under control.**

---

## Connection to Other Topics

- **TRPO**: Uses KL divergence as a constraint (see [TRPO.md](./TRPO.md))
- **PPO**: Approximates KL constraint with clipping
- **RLHF**: Prevents fine-tuned LLMs from diverging too far from base model
