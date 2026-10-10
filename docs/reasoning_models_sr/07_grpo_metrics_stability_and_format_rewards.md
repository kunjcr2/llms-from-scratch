# Lecture 7 - GRPO Metrics, Stability, and Format Rewards

**Video:** Build a Reasoning Model From Scratch 7 (follow-up to [Lecture 6](06_rlvr_and_grpo.md))

This lecture revisits the small GRPO implementation from Lecture 6. The focus is not a new training paradigm, but learning how to tell whether a run is healthy, making policy updates safer, and adding optional format signals such as `think` tags.

## 1. What changes from Lecture 6?

The basic loop is still:

```text
sample several answers for one prompt
        |
verify each answer
        |
normalize rewards into group-relative advantages
        |
update the policy
```

The lecture extends that loop in four directions:

- log and plot more useful training metrics;
- run long jobs as scripts and evaluate saved checkpoints separately;
- stabilize the policy-gradient update with clipped policy ratios;
- optionally add a reference-model KL term and a format reward.

The educational setup remains deliberately small: roughly four rollouts and at most 1,024 generated tokens. Production reasoning systems generally use many more samples and much longer trajectories, so instability at this scale does not automatically mean that the algorithm is wrong.

## 2. Why the GRPO loss is not enough

In supervised next-token training, a steadily decreasing cross-entropy loss is usually a useful sign. The GRPO loss is different: it is a policy-gradient objective whose value can fluctuate and can become very large when an update is bad. The main question is therefore not simply whether the loss decreases.

Track at least:

| Metric | Interpretation |
|---|---|
| `loss` | Policy-update magnitude; look for large spikes rather than a monotonic decline. |
| `reward_avg` | Mean verifier reward across the group. With binary rewards and four rollouts, `.25` means one correct answer. |
| `response_length` | Whether the model is learning useful reasoning or hitting the generation limit. |
| evaluation accuracy | Generalization on a held-out benchmark such as MATH-500. |
| advantage mean/std | A sanity check and an estimate of how much relative training signal exists. |
| entropy | How concentrated or diffuse the model's next-token distribution is. |
| policy ratio | How far the current policy moved from the policy used to generate the rollout. |

Use moving averages to expose trends while retaining the raw values. A high training reward is not sufficient: the training prompts may be easier than the evaluation set, and the model can exploit a weak verifier.

## 3. Long runs: scripts, logs, and checkpoints

For jobs longer than a few minutes, the lecture moves the notebook code into command-line Python scripts. This makes it practical to run on a separate machine and avoids tying up the main computer.

Useful script options include:

- number of training steps;
- number of rollouts per prompt;
- maximum new tokens and sampling temperature;
- whether to evaluate checkpoints;
- whether to skip groups whose advantages are all zero;
- estimated time remaining.

The scripts save both human-readable text logs and CSV metrics. CSV logs make it easy to plot loss, rewards, output length, and evaluation accuracy with Matplotlib. Before downloading and executing code from a repository, inspect the URL and skim the downloaded source. This is a small but important supply-chain safety habit.

Checkpoint evaluation on all 500 MATH-500 examples can take hours, so it is often better to train first, inspect the inexpensive training metrics, and evaluate selected checkpoints afterward. The best checkpoint may be early in the run; do not assume the final checkpoint is best.

## 4. Reading unstable-training plots

A representative unmodified run improves from a low reward to roughly 75% group reward and reaches about 45% MATH-500 accuracy, but later deteriorates. Typical warning signs are:

- extreme negative loss spikes;
- reward and evaluation accuracy dropping after an initial improvement;
- responses frequently reaching the maximum length;
- a growing mismatch between training reward and held-out accuracy.

Length saturation has two possible causes: the model needs more room to finish its reasoning, or it is producing incoherent text and failing to emit an end-of-sequence token. Increasing `max_new_tokens` can help, but costs memory and time. The small rollout count also makes the estimates noisy.

The practical lesson is to save and compare checkpoints, not just watch the current step. A run that improves briefly and then collapses is still useful if the earlier checkpoint is retained.

## 5. Advantage statistics

GRPO standardizes rewards inside each group:

$$
A_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}.
$$

The mean of the advantages should be approximately zero. This is a useful implementation check. The standard deviation is more informative for learning:

- high standard deviation means the group contains different-quality answers;
- zero standard deviation means every rollout received the same reward;
- with binary rewards, a useful group usually contains at least one correct and one incorrect answer.

If all answers are correct or all are wrong, the group has no relative ranking signal. Skip the update or prioritize prompts that produce a mixture of outcomes. This is more efficient than spending compute on zero-gradient examples.

## 6. Entropy: measuring uncertainty

For a next-token distribution `p`, entropy is:

$$
H(p) = -\sum_j p_j \log p_j.
$$

Using natural logarithms measures uncertainty in **nats**. In PyTorch, compute it efficiently from logits:

```python
logprobs = torch.log_softmax(logits, dim=-1)
probs = torch.exp(logprobs)
entropy = -(probs * logprobs).sum(dim=-1)
```

The maximum entropy is `log(vocab_size)`, reached when every token is equally likely. Entropy near zero means the model is nearly deterministic and may not explore. Entropy near the maximum means the model is almost random. The desirable value is task- and model-dependent; look for sudden changes and combine it with answer quality rather than treating a target number as universal.

An entropy increase during training can indicate that the policy is becoming less confident or being damaged by unstable updates. It is a diagnostic, not by itself a reward.

## 7. Clipped policy ratios

The basic loss uses the rollout log-probability directly:

$$
\mathcal{L}_{PG} = -\frac{1}{G}\sum_i A_i \log p_\theta(y_i \mid x).
$$

Large gradients can move the policy too far in one update. Store the old log-probability from the policy that generated the rollout and compare it with the current log-probability:

$$
r_i(\theta) = \exp\left(\log p_\theta(y_i\mid x) - \log p_{old}(y_i\mid x)\right).
$$

Then use a PPO-style clipped objective:

```python
ratio = torch.exp(new_logps - old_logps)
clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
objective = torch.minimum(ratio * advantages,
                         clipped_ratio * advantages)
loss = -objective.mean()
```

The clipping interval prevents extreme policy changes. Smaller `eps` is more conservative; the lecture's illustrative value of `10` is intentionally generous, while values around `0.1` are more typical. Track the mean ratio and the amount of clipping to see whether the constraint is active.

In the lecture's comparison, clipping changed loss spikes from thousands in magnitude to roughly single-digit values and kept reward, response length, and evaluation accuracy much more stable. It did not automatically keep improving benchmark accuracy, but it prevented the obvious collapse.

## 8. KL regularization and reward hacking

A KL term constrains the current policy relative to a reference policy, often a frozen copy of the initial model:

$$
\mathcal{L} = \mathcal{L}_{PG} + \beta\mathcal{L}_{KL}.
$$

The reference model discourages the policy from drifting into strange behaviors that happen to fool the verifier. This is important when the verifier is brittle: a model may learn a formatting or token-level exploit rather than the intended task. That failure mode is called reward hacking.

The lecture presents a simplified sequence-level KL surrogate based on current and reference log-probabilities. It is useful for illustrating the idea but is not a complete implementation of every production GRPO/PPO KL formulation. In particular, a frozen reference log-probability can behave like a constant in the wrong expression; an importance-weighted or otherwise more carefully derived term is preferable.

The demonstrated KL configuration destabilized the run: output length grew while reward and evaluation accuracy fell to zero. Therefore:

- KL is optional, not automatically beneficial;
- use it when verifier exploitation or excessive policy drift is a real concern;
- tune `beta` and validate the exact KL objective;
- start with clipped ratios, which were sufficient for the math example.

## 9. Format rewards and `think` tags

The original reward checks answer correctness and required answer formatting such as `\\boxed{...}`. A format reward can additionally encourage a complete reasoning wrapper:

```text
<think> ... reasoning ... </think>
```

The tag is a parsing convention, not proof that the text is faithful or useful reasoning. It can make internal content easier to separate for evaluation and safety research, but directly rewarding a visible reasoning format is cosmetic unless the downstream system needs it.

### Adding special tokens

If the base tokenizer has never seen the tags, it may split `<think>` into several ordinary subword tokens. Add them as special tokens so each tag has one stable token ID. The lecture adds think tags (and tool-response tags for future tool-use experiments) and aligns their IDs with the corresponding reasoning model tokenizer.

Adding token IDs alone is brittle: the base model was not pretrained on those tokens. Continued pretraining or supervised fine-tuning with the tags present is a better preparation before RL. For the demonstration, an already-trained reasoning variant is used instead.

### Format reward implementation

Inspect generated tokens after the prompt and return `1.0` only when the opening and closing tag IDs are both present and ordered correctly; otherwise return `0.0`. Test malformed, missing, and reordered tags before using the function in training.

The simplest combined reward is:

```python
total_reward = correctness_reward + format_weight * format_reward
```

However, this lets the model earn an easy format point for an incorrect answer. In the lecture's run, the format reward was almost always high while the correctness component was much lower, suggesting that the model optimized the easy proxy.

A better exercise is conditional format reward:

```python
total_reward = correctness_reward * (
    1.0 + format_weight * format_reward
)
```

Now a wrong answer receives no format bonus, while a correct answer receives extra credit for using the desired tags. Always inspect each reward component separately.

## 10. Research-inspired extensions

The lecture points to DAPO, Dr. GRPO, DeepSeek-V3, GDPO, and GSPO as sources of further ideas. The main practical takeaways are:

- filter groups with zero gradient signal;
- oversample prompts with a useful mix of correct and incorrect answers;
- avoid prompts that are always trivial or currently impossible;
- compare sequence-level and token-level objectives (sum versus average log-probabilities);
- consider omitting KL for domains such as math when it is unnecessary;
- tune KL strength by domain rather than using one global value;
- increase rollout count and trajectory length when hardware permits.

These changes are scale-sensitive. Results from a four-rollout, 1,024-token teaching run should not be extrapolated directly to large reasoning-model training.

## 11. Suggested experiment order

1. Run the baseline script and save raw logs, CSV metrics, and checkpoints.
2. Plot loss, reward, response length, evaluation accuracy, advantage statistics, and entropy.
3. Select the best checkpoint using held-out accuracy and inspect its actual answers.
4. Add clipped policy ratios and compare stability with the same data and seed where possible.
5. Add KL only if there is evidence of policy drift or verifier exploitation; validate the formulation and tune its coefficient.
6. Add a format reward, but track correctness and format rewards separately.
7. Try conditional format reward so formatting cannot compensate for an incorrect answer.

## Takeaways

- GRPO training quality cannot be judged from loss alone.
- Advantage spread tells you whether a rollout group contains learning signal.
- Entropy, response length, ratios, and held-out accuracy reveal different failure modes.
- Clipped policy ratios are an effective first stabilization mechanism.
- KL regularization can control drift, but a simplistic KL surrogate can make training worse.
- Format rewards should not overpower correctness; conditional bonuses are safer than unconditional addition.
- Long, expensive runs should be checkpointed and evaluated selectively.
