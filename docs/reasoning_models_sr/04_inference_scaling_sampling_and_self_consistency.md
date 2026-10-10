# Lecture 4 — Inference Scaling: Temperature, Top-p, and Self-Consistency

**Video:** [Build A Reasoning Model From Scratch 4: Inference Scaling 1](https://www.youtube.com/watch?v=t5y-kS9nNxU) (1:37:11)

## 1. Training-time versus inference-time scaling

- **Training-time scaling:** spend more compute to learn better parameters.
- **Inference-time scaling:** keep parameters fixed but spend more compute per problem.

Inference scaling may use a longer reasoning trace, multiple sampled traces, voting, scoring, search, or refinement. It exchanges latency and token cost for a higher chance of a correct answer.

This lecture builds three ingredients:

1. A chain-of-thought-style prompt.
2. Diverse sampling through temperature and top-p.
3. Self-consistency across several sampled solutions.

## 2. Chain-of-thought prompting

Append a direct instruction such as:

```text
Explain step by step.
```

The model is then encouraged to produce intermediate calculations before `\boxed{ANSWER}`. No weights change; the prompt causes the base model to use more output tokens as a scratchpad.

This increased the lecture's base-model MATH-500 accuracy from roughly 15% to 40.6%, but runtime rose from about 10 to 84.5 minutes because responses were longer.

## 3. From logits to probabilities

For the next position, the model returns a logit vector `z` of length `V`. Softmax converts it into a probability distribution:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.
$$

Greedy decoding uses `argmax(z)` and always selects the same token for the same numerical state. Diverse candidate reasoning paths require stochastic sampling.

## 4. Temperature scaling

Before softmax, divide logits by a positive temperature `T`:

$$
p_i(T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

```python
def scale_logits(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature
```

Interpretation:

- `T < 1`: differences grow; the distribution becomes sharper and more deterministic.
- `T = 1`: unchanged distribution.
- `T > 1`: differences shrink; more low-ranked tokens receive meaningful probability.
- `T = 0` is handled as a special greedy-decoding setting, not literal division.

Sample rather than taking the argmax:

```python
probs = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

A random seed makes an experiment more repeatable, but different seeds are intentionally used to obtain different candidates.

## 5. Why temperature alone is insufficient

At high temperature, probability spreads across the entire vocabulary. Extremely implausible tokens can then be sampled, harming coherence. We want diversity among plausible options, not arbitrary noise.

## 6. Top-p (nucleus) sampling

Top-p keeps the smallest high-probability token set whose cumulative mass reaches threshold `p`, removes all other tokens, and renormalizes the survivors.

Algorithm:

1. Sort probabilities descending.
2. Compute the cumulative sum.
3. Keep tokens while the probability mass *before* that token is below `top_p`; this includes the token that crosses the threshold.
4. Set all other probabilities to zero.
5. Scatter back to original vocabulary order.
6. Renormalize.

```python
def top_p_filter(probs, top_p):
    if top_p is None or top_p >= 1.0:
        return probs

    sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_p, dim=-1)
    keep = cumulative - sorted_p < top_p
    keep[..., 0] = True

    kept = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, kept)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
```

Unlike top-k, the number of retained tokens adapts to confidence. A peaked distribution may retain very few; a flatter one may retain many.

## 7. Flexible sampled generation

The cached generation loop changes only at token selection:

```python
if temperature is None or temperature == 0:
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
else:
    probs = torch.softmax(logits / temperature, dim=-1)
    probs = top_p_filter(probs, top_p)
    next_token = torch.multinomial(probs.cpu(), 1).to(logits.device)
```

The lecture samples on CPU and moves the ID back to the model device for more consistent behavior across some devices. The surrounding KV-cache and EOS logic remains unchanged.

Sampling is also the mechanism modified by statistical text-watermarking schemes: they subtly bias which plausible tokens are selected. Understanding the basic sampler is therefore useful beyond reasoning models.

## 8. Self-consistency

One stochastic chain may take a bad turn. Self-consistency generates `N` independent reasoning traces, extracts each final answer with the verifier, and selects the most frequent answer.

```text
same problem
 ├── sampled trace 1 → answer A
 ├── sampled trace 2 → answer B
 ├── sampled trace 3 → answer A
 └── sampled trace 4 → answer A
                         ↓
                   choose answer A
```

Strictly, selecting the most frequent answer is a **plurality** vote; it need not exceed 50%.

```python
from collections import Counter

def vote(extracted_answers):
    counts = Counter(extracted_answers)
    ranked = counts.most_common()
    if not ranked:
        return None

    best_count = ranked[0][1]
    winners = [answer for answer, count in ranked if count == best_count]
    return winners[0] if len(winners) == 1 else None
```

Keep both full traces and extracted answers. Full traces support debugging; normalized short answers are what should be counted.

### Tuning diversity

- Nearly identical long answers: gently increase temperature.
- Incoherent/off-topic answers: decrease temperature or top-p.
- Too few samples: the vote is noisy.
- More samples: potentially stronger consensus, but roughly proportional generation cost.
- A tie needs an explicit policy; Lecture 5 adds scorers that can break ties or rank every candidate.

## 9. Reference MATH-500 results

Selected reported results on a DGX Spark CUDA device:

| Method | Model | Accuracy | Time |
|---|---|---:|---:|
| Greedy baseline | Base | 15.2% | 10.1 min |
| Greedy baseline | Reasoning | 48.2% | 182.1 min |
| CoT prompt | Base | 40.6% | 84.5 min |
| Temperature + top-p | Base | 17.8% | 30.7 min |
| Top-p + CoT | Base | 33.4% | 129.2 min |
| Self-consistency `n=3` + top-p + CoT | Base | 42.2% | 211.6 min |
| Self-consistency `n=5` + top-p + CoT | Base | 48.0% | 452.9 min |
| Self-consistency `n=10` + top-p + CoT | Base | 52.0% | 862.6 min |
| Self-consistency `n=3` + top-p + CoT | Reasoning | 55.2% | 544.4 min |

The curve is not guaranteed to be monotonic for every finite run: `n=5` without CoT scored below `n=3` in the lecture's experiment. One seed and 500 examples do not establish a smooth scaling law.

## 10. Practical interpretation

- CoT prompting provided the largest low-complexity improvement in this setup.
- Random sampling alone offered little benefit; diversity needs a selection mechanism.
- Self-consistency can let a small base model approach or exceed a reasoning variant, but at substantial latency.
- Accuracy must be considered jointly with generated tokens, wall time, and hardware cost.
- Majority frequency does not prove correctness; correlated samples can repeat the same mistake.

## Takeaways

- Temperature controls distribution sharpness; top-p removes the implausible tail.
- Sampling creates candidate reasoning paths; it is not itself an accuracy guarantee.
- Self-consistency turns diversity into a decision by voting over normalized final answers.
- More inference compute can improve accuracy without changing model weights, but the cost can grow almost linearly with sample count.
- The next lecture replaces frequency-only selection with answer scoring and iterative critique.

