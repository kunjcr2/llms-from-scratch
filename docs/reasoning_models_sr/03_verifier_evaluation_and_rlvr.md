# Lecture 3 — The Verifier for Evaluation and RLVR

**Video:** [Build A Reasoning Model From Scratch 3: The Verifier for Evaluation and RL with Verifiable Rewards](https://www.youtube.com/watch?v=JQJ_8_jSAoY) (1:26:46)

## 1. Why evaluation comes first

The base model must be measured before adding inference or training techniques. Otherwise, a polished example can be mistaken for an actual improvement.

The same verifier has two jobs:

1. **Evaluation:** compare the base model against every later version.
2. **Training:** supply correctness rewards for reinforcement learning with verifiable rewards (RLVR).

A production system should avoid using the exact same examples for both jobs. The verifier logic can be shared, but training and evaluation datasets should not overlap.

## 2. Four broad evaluation families

| Family | Mechanism | Strength | Limitation |
|---|---|---|---|
| Multiple choice | Select one provided option | Simple and objective | Often measures recognition/knowledge more than free-form reasoning |
| Verifier-based | Generate an answer and check it against ground truth | Objective and reusable as an RL reward | Only works when answers are mechanically checkable |
| Leaderboard/pairwise | Humans or models choose between two answers | Natural for open-ended preferences | Subjective and affected by judge bias |
| LLM-as-a-judge | Another LLM scores an answer using a rubric | Flexible for free-form tasks | Sensitive to the judge, rubric, and model biases |

No method dominates every task. This course uses math verification because it is objective and keeps attention on the reasoning algorithm. Code can also be verified with tests or execution, but adds language, sandboxing, compilation, security, and formatting concerns.

## 3. Verifier pipeline

```text
math problem
    ↓ prompt template
LLM response
    ↓ extract final candidate
candidate answer
    ↓ normalize notation
canonical expression
    ↓ symbolic/equality checks against ground truth
correct / incorrect
```

The difficult part is rarely comparing two clean integers. Models emit prose, LaTeX, nested braces, units, multiple values, special tokens, or no requested answer box at all.

## 4. Make the output contract explicit

The prompt asks for a final boxed result:

```python
def render_prompt(problem):
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{problem}\n\nAnswer:"
    )
```

This contract makes extraction more reliable, but a verifier must still handle violations.

## 5. Extracting `\boxed{...}` correctly

A regex such as `r"\\boxed\{(.*?)\}"` fails on nested expressions like `\boxed{\frac{1}{2}}`. Instead:

1. Find the final `\boxed` occurrence.
2. Skip whitespace and require `{`.
3. Scan one character at a time.
4. Increase brace depth at `{`, decrease it at `}`.
5. Finish when depth returns to zero.
6. Reject unbalanced input.

Using the **last** box is sensible because a response may discuss intermediate boxed values before giving its final result.

### Fallback policies

When no box is present, the lecture supports explicit policies:

- `number_then_full`: use the last simple number; if none exists, return all text.
- `number_only`: use the last simple number; otherwise return an empty string.
- `None`: require a box and otherwise return an empty string.

Loose fallbacks are useful for evaluation diagnostics. RL rewards later require the box, which also acts as a format reward.

## 6. Normalization

Equivalent answers often have different surface forms. Normalize before comparing:

- Remove model special tokens and outer math delimiters.
- Strip whole-answer `\text{...}` wrappers.
- Remove `\left`, `\right`, and LaTeX spacing commands.
- Canonicalize multiplication symbols to `*`.
- Convert `\dfrac`/`\tfrac` to `\frac`.
- Convert fractions into parser-friendly division.
- Convert roots into `sqrt(...)`.
- Convert `^` and Unicode superscripts into exponent syntax.
- Remove degree/percent formatting when appropriate for the benchmark.
- Remove thousands separators and normalize case/whitespace.
- Strip leading multiple-choice labels such as `C. 3`.

Normalization must be conservative. Over-aggressive rewriting can turn a wrong answer into a seemingly equivalent one.

## 7. Mathematical equivalence with SymPy

String equality is a fast first check. If strings differ, parse both expressions and simplify their difference:

```python
from sympy import simplify

def equivalent(ground_truth, prediction):
    if ground_truth == prediction:
        return True

    gt = safe_sympy_parse(ground_truth)
    pred = safe_sympy_parse(prediction)
    if gt is None or pred is None:
        return False

    try:
        return simplify(gt - pred) == 0
    except (TypeError, ValueError):
        return False
```

The parser enables implicit multiplication so `2x` can become `2*x`. It also:

- rejects extremely long garbage input,
- catches parser/tokenizer/polynomial errors,
- returns failure instead of crashing an evaluation run.

This recognizes equivalences such as `1/2 == 0.5` and algebraically identical expressions.

### Multi-part answers

For tuple/list-shaped outputs such as `(2, 3)`:

1. Split both prediction and ground truth into parts.
2. Require the same number of parts.
3. Compare each pair for mathematical equivalence.

This simple method assumes commas really separate top-level answers; a production parser may need to understand deeper nesting, sets, intervals, units, or order invariance.

## 8. A robust grader

The full grader composes small, testable stages:

```python
def grade_answer(pred_text, ground_truth_text):
    gt_parts = split_into_parts(normalize_text(ground_truth_text))
    pred_parts = split_into_parts(normalize_text(pred_text))

    if not gt_parts or len(gt_parts) != len(pred_parts):
        return False

    return all(
        equivalent(gt, pred)
        for gt, pred in zip(gt_parts, pred_parts)
    )
```

Test deliberately varied cases: boxed fractions, decimals, nested LaTeX, Unicode exponents, multiple-choice labels, missing boxes, wrong answers, malformed syntax, and multi-part outputs.

## 9. MATH-500 evaluation

The benchmark provides 500 problems with fields such as problem, worked solution, short answer, subject, and difficulty. Evaluation needs only the problem and short answer; the model creates its own solution.

For every example:

1. Render the prompt.
2. Generate an answer with a token cap.
3. Extract the candidate.
4. Grade it.
5. Accumulate correct/total.
6. Save prompt, raw output, extraction, ground truth, and grade to JSON.

Saving raw records is essential. When a score looks wrong, inspect whether the model failed, the extractor failed, or the equivalence checker failed.

## 10. Prompt sensitivity and contamination

Base models can change dramatically after tiny prompt edits—`Question` versus `Problem`, adding a persona, or removing the template. This can reflect fragile conditioning or memorization rather than improved reasoning.

Practical rules:

- Freeze the prompt template when comparing models.
- Report it with the score.
- Treat public benchmarks as potentially present in pretraining data.
- For important deployments, maintain a private, never-published evaluation set and run it locally with models that do not transmit the data.
- Evaluate enough examples; 10 questions can change by 10 percentage points after one different answer.

## 11. Reproducibility is approximate

Even greedy decoding can differ across CPU, CUDA, MPS, PyTorch versions, dtypes, or execution order. Small floating-point differences can change one selected token; autoregressive feedback then makes the rest of the answer diverge.

For serious comparisons:

- record software and hardware,
- use the same prompt and decoding settings,
- run multiple evaluations where feasible,
- report means and variability,
- do not over-interpret a one-example difference.

## 12. Reference results

The official full MATH-500 run reported:

| Model | Accuracy | Runtime on DGX Spark CUDA |
|---|---:|---:|
| Qwen3-0.6B Base | 15.6% in the video's run; 15.2% in later reference tables | about 10 min |
| Qwen3-0.6B reasoning | about 48–51%, depending on run/version | about 3 hours |

The reasoning variant is slower largely because it generates much longer answers. More tokens can enable useful intermediate computation, but they also increase latency and cost.

## Takeaways

- Evaluation must precede optimization.
- A verifier is objective only if parsing and normalization are correct.
- A structured output contract simplifies grading but does not replace defensive parsing.
- Preserve raw outputs so verifier bugs are auditable.
- Prompt sensitivity, benchmark contamination, and floating-point variation can dominate small reported differences.
- This verifier becomes the reward function used by RLVR in Lecture 6.

