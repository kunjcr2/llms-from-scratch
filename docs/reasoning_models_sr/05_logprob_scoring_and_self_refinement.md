# Lecture 5 — Log-Probability Scoring and Self-Refinement

**Video:** [Build A Reasoning Model From Scratch 5: Inference Scaling 2](https://www.youtube.com/watch?v=TVMyOJ_3Gxo) (1:08:39)

## 1. From voting to scoring and revision

Lecture 4 generated several answers and selected the most frequent. This lecture introduces two other ways to spend inference compute:

- **Best-of-N:** sample multiple answers, score every answer, and keep the best.
- **Self-refinement:** generate one answer, critique it, revise it, and optionally keep only revisions whose score is no worse.

Scorers can also break self-consistency ties. The model's weights remain unchanged throughout.

## 2. Rule-based heuristic scoring

A simple scorer rewards desired structure and brevity:

```python
def heuristic_score(answer, brevity_scale=500.0):
    score = 0.0

    if extract_final_candidate(answer, fallback=None):
        score += 2.0                 # explicit \boxed{...}
    elif extract_final_candidate(answer, fallback="number_only"):
        score += 1.0                 # at least a numeric result

    score += 1.5 * math.exp(-len(answer) / brevity_scale)
    return score
```

The brevity bonus decays smoothly rather than imposing a hard cutoff. The lecture counts characters for simplicity; token count is a more model-relevant alternative.

This score measures format and concision, not correctness. A concise boxed wrong answer can score well, so heuristics should not be confused with a verifier.

## 3. Token probabilities

For a generated sequence `x₁, …, x_T` under model weights `W`, the joint conditional probability factors autoregressively:

$$
P(x_{1:T}\mid W)=\prod_{t=1}^{T}P(x_t\mid x_{<t}, W).
$$

To score an existing answer:

1. Concatenate prompt and answer token IDs.
2. Run one forward pass.
3. At each answer position, take the model probability assigned to the token that actually occurred next.
4. Combine those token scores.

The logits at position `t` predict token `t+1`, so targets are shifted by one. This alignment is a frequent source of off-by-one bugs.

## 4. Why log-probabilities?

Multiplying many probabilities underflows toward zero. Logs convert the product into a stable sum:

$$
\log P(x_{1:T}\mid W)
=\sum_{t=1}^{T}\log P(x_t\mid x_{<t},W).
$$

Use `torch.log_softmax(logits, dim=-1)` rather than `torch.log(torch.softmax(...))`; the fused log-softmax is numerically safer.

Log-probabilities are non-positive. Values closer to zero mean the model assigned higher probability.

## 5. Average answer log-probability

When comparing answers of different lengths, sum scores naturally penalize long text because every additional probability contributes another negative term. Average over answer tokens for length normalization:

$$
s(x_{1:T})=\frac{1}{T}\sum_{t=1}^{T}\log P(x_t\mid x_{<t},W).
$$

```python
@torch.inference_mode()
def average_answer_logprob(model, tokenizer, prompt, answer, device):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids) - 1
    end = full_ids.numel() - 1
    positions = torch.arange(start, end, device=device)
    targets = full_ids[start + 1 : end + 1]
    answer_logprobs = logprobs[positions, targets]
    return answer_logprobs.mean()
```

Only answer tokens are scored; otherwise prompt likelihood contaminates the comparison. Averaging improves length comparability but does not calibrate correctness. A model can be confidently wrong.

Floating-point results can differ across CPU, MPS, and CUDA because this calculation combines many small values.

## 6. Best-of-N

```text
sample N diverse answers
        ↓
score each complete answer
        ↓
return argmax(score)
```

Self-consistency chooses the most common final answer. Best-of-N chooses the highest-scoring individual response, even if its final answer appears only once.

Lecture reference results for the base model with three sampled CoT answers:

| Method | Accuracy | Time |
|---|---:|---:|
| CoT baseline | 33.4% | 129.2 min |
| Best-of-3 + heuristic | 40.6% | 327.7 min |
| Best-of-3 + average logprob | 43.2% | 330.2 min |

For self-consistency with `n=3`, an average-logprob tie-breaker improved the reported result from 43.2% to 44.8%.

## 7. Self-refinement

Self-refinement is a sequential loop rather than parallel candidate search:

```text
problem → draft → critique → revised answer
                           ↓
                 optional score/accept
                           ↓
                     repeat if desired
```

The critique prompt asks a reviewer persona to identify logical gaps, missing steps, or arithmetic errors and propose a short repair plan. The revision prompt includes the original problem, draft, and critique, then requires a concise boxed answer.

```python
def make_critique_prompt(question, draft):
    return f"""You are a meticulous reviewer. Identify logical errors,
missing steps, or arithmetic mistakes. If the answer seems correct, say so
briefly, then propose a concise fix plan.

Question:
{question}

Draft answer:
{draft}

Critique:"""

def make_refine_prompt(question, draft, critique):
    return f"""Revise the answer using the critique. Keep it concise and end
with a final boxed result: \\boxed{{ANSWER}}

Question:
{question}

Previous answer:
{draft}

Critique:
{critique}

Revised answer:"""
```

## 8. Acceptance logic

For each iteration:

1. Generate a critique of the current draft.
2. Generate a revision conditioned on the draft and critique.
3. Extract and score the revision.
4. Record draft, critique, revision, extractions, and both scores.
5. Accept the revision if `revised_score >= current_score`.

If no scorer is supplied, both scores are zero and every revision is accepted. The score therefore acts as a gate, but the result can only be as good as the scorer.

Important detail: score the revision against the original problem prompt, not against the critique/refinement meta-prompt.

## 9. Reference self-refinement results

Selected full MATH-500 results:

| Model/method | Iterations | Accuracy |
|---|---:|---:|
| Base baseline | — | 15.2% |
| Base, no scorer | 1 | 25.0% |
| Base, no scorer | 2 | 22.0% |
| Base, heuristic gate | 1 | 21.6% |
| Base, average-logprob gate | 1 | 21.4% |
| Reasoning baseline | — | 48.2% |
| Reasoning, no scorer | 1 | 56.6% |
| Reasoning, heuristic gate | 1 | 57.8% |
| Reasoning, average-logprob gate | 1 | 48.4% |

These results are deliberately instructive: a second refinement was not necessarily better, and logprob gating sometimes blocked beneficial changes or favored fluent wrong answers. Iteration count is a budget, not a guarantee.

## 10. Failure modes and design lessons

- **Self-bias:** the same model may not recognize its own mistake.
- **Critique hallucination:** a correct draft can be “fixed” into a wrong answer.
- **Confidence ≠ truth:** log-probability reflects model preference, not external correctness.
- **Heuristic gaming:** formatting and brevity bonuses can select polished errors.
- **Cost growth:** one refinement round requires draft, critique, and revision generation.
- **Context growth:** carrying long drafts and critiques increases input length.
- **Diminishing or negative returns:** later rounds can undo earlier improvements.

When a true verifier is available, use it directly. Heuristic and model-confidence scorers are most useful when ground truth is unavailable at inference time.

## Takeaways

- Log-probabilities provide a stable way to score how likely a model finds a sequence.
- Average logprob is length-normalized and useful for ranking, but is not a correctness measure.
- Best-of-N is parallel exploration plus scoring; self-refinement is sequential critique and revision.
- More refinement iterations can reduce accuracy.
- The next lecture turns the deterministic verifier into a training reward so good behavior changes the model's weights.

