# Lecture 6 — Reinforcement Learning with Verifiable Rewards and GRPO

**Video:** [Build A Reasoning Model From Scratch 6: Reinforcement Learning 1](https://www.youtube.com/watch?v=237Hf7Q3lgg) (1:26:53)

## 1. Inference scaling versus training reasoning behavior

Previous lectures improved answers without changing weights. This lecture performs post-training:

```text
pretrained base LLM
        ↓ sample multiple solutions
verifiable rewards
        ↓ GRPO policy-gradient update
reasoning-trained LLM
```

Pretraining optimizes next-token prediction over large corpora. RL optimizes a sequence-level objective such as answer correctness. Inference scaling can still be applied after RL; the two approaches are complementary.

## 2. What distinguishes reasoning-model behavior?

Reasoning-trained models tend to produce longer intermediate traces, reconsider mistakes, and use extra tokens on difficult problems. The useful behavior is not simply verbosity:

- A capable model may solve an easy problem with a short trace.
- A weaker model may generate a long but incorrect trace.
- The training signal in this chapter rewards the final verifiable result, not prose length or how persuasive the explanation sounds.

DeepSeek-style RLVR showed that behaviors such as backtracking or “wait, that was wrong” can emerge even when no target reasoning trace is supplied. The model explores outputs and receives only outcome feedback.

## 3. RLHF versus RLVR

### RLHF

Classic RLHF uses human preference comparisons, commonly trains a learned reward model, and then optimizes the policy against that reward. It is flexible but data- and compute-intensive, and the learned reward can be imperfect.

### RLVR

RL with verifiable rewards replaces the learned reward model with a deterministic checker:

- math: compare the final answer symbolically,
- code: compile, execute, or run unit tests,
- other domains: use any reliable automatic success criterion.

The lecture uses a binary reward:

```python
def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(answer_text, fallback=None)
    if not extracted:
        return 0.0
    return float(grade_answer(extracted, ground_truth))
```

Requiring `\boxed{...}` gives reward only when the answer is both correctly formatted and mathematically correct—an implicit format reward.

The outcome reward does **not** score intermediate reasoning steps. Process reward models may sound attractive, but they add complexity and can reward convincing-looking rationales; the DeepSeek-R1 work reported that outcome-only verification was more effective for its setup.

## 4. Why GRPO instead of PPO?

Both are policy-gradient methods, where the LLM is the **policy**.

- PPO commonly uses a learned value/critic model to estimate how good a state is.
- Group Relative Policy Optimization (GRPO) removes the separate critic and derives a baseline from several responses to the same prompt.

That makes GRPO easier to explain and more memory-efficient for LLM training.

### Chef analogy

Give several chefs the same recipe challenge. Taste all dishes, score them, determine which are above or below the group's average, reinforce choices behind relatively better dishes, and discourage choices behind worse dishes. The “group relative” comparison replaces a separate critic.

## 5. Training data and leakage control

Training uses math problems derived from MATH with all MATH-500 evaluation examples explicitly excluded.

Only two fields are needed:

- `problem`: prompt given to the policy,
- `answer`: short ground-truth result used by the verifier.

The worked `solution` is intentionally not used. This is RL exploration, not supervised imitation of one preferred derivation.

## 6. The GRPO procedure

For each problem:

1. Sample a group of `G` responses (rollouts).
2. Compute a verifiable reward for every rollout.
3. Normalize rewards within the group into advantages.
4. Recompute each rollout's differentiable sequence log-probability.
5. Form an advantage-weighted policy-gradient loss.
6. Backpropagate and update model weights.

“Rollout,” “completion,” and “sampled response” refer to the same generated sequence here.

## 7. Sampling rollouts

Sampling uses the cached temperature/top-p generator from Lecture 4, but returns:

- full prompt + response token IDs,
- prompt length,
- decoded response text.

Use `@torch.no_grad()` for sampling, not `@torch.inference_mode()`. Inference-mode tensors have stronger restrictions and may later cause `RuntimeError: Inference tensors cannot be saved for backward` when reused in computations involved in training.

```python
@torch.no_grad()
def sample_response(model, tokenizer, prompt, device,
                    max_new_tokens=512, temperature=0.8, top_p=0.9):
    # Encode prompt, initialize KV cache, then repeatedly:
    # logits → temperature → softmax → top-p → multinomial sample.
    # Return full token IDs, prompt length, and decoded generated text.
    ...
```

## 8. Group-relative advantages

For rewards `r₁, …, r_G`, compute:

$$
A_i = \frac{r_i-\mu_r}{\sigma_r+\epsilon}.
$$

- Positive advantage: rollout is better than its group; increase its probability.
- Negative advantage: rollout is worse; decrease its probability.
- Near zero: little update.

If all group rewards are identical—all correct or all wrong—then every numerator is zero and the group supplies no learning signal. This is why several diverse rollouts matter. With binary rewards, useful learning requires at least one correct and one incorrect completion in the group.

## 9. Sequence log-probabilities

Lecture 5 averaged token logprobs to compare answers of different lengths. GRPO instead uses the **sum** for the complete rollout:

$$
\log p_W(y\mid x)
=\sum_{t=1}^{T}\log p_W(y_t\mid y_{<t},x).
$$

```python
def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    start = prompt_len - 1
    end = token_ids.numel() - 1
    positions = torch.arange(start, end, device=token_ids.device)
    targets = token_ids[start + 1 : end + 1]
    return logprobs[positions, targets].sum()
```

Do not call `.item()` here: that would convert the tensor to a Python number and sever the gradient graph.

Summed logprobs become more negative with length, so—other things equal—the objective favors shorter completions. This can encourage efficient answers, but length effects must be monitored.

## 10. Policy-gradient loss

The simplified lecture loss is:

$$
\mathcal{L}_{PG}
=-\frac{1}{G}\sum_{i=1}^{G}A_i
  \sum_{t=1}^{T_i}\log p_W(y_t^{(i)}\mid y_{<t}^{(i)},x).
$$

```python
logps = torch.stack(rollout_logprobs)
loss = -(advantages.detach() * logps).mean()
```

- Detach advantages so they are treated as fixed learning signals.
- The negative sign is required because PyTorch optimizers minimize, while policy gradient wants to maximize advantage-weighted log-probability.
- Positive-advantage sequences become more likely; negative-advantage sequences become less likely.

This chapter intentionally omits the reference-policy KL penalty and PPO-style ratio clipping. Later improvements add stabilizers; the lecture's goal is to expose the smallest intelligible GRPO core.

## 11. Minimal GRPO step

```python
def compute_grpo_loss(model, tokenizer, example, device, num_rollouts=8):
    prompt = render_prompt(example["problem"])
    rollout_logps, rewards, samples = [], [], []

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model, tokenizer, prompt, device
        )
        rewards.append(reward_rlvr(text, example["answer"]))
        rollout_logps.append(sequence_logprob(model, token_ids, prompt_len))
        samples.append(text)

    rewards_t = torch.tensor(rewards, device=device)
    advantages = (
        (rewards_t - rewards_t.mean()) /
        (rewards_t.std() + 1e-4)
    )
    logps = torch.stack(rollout_logps)
    loss = -(advantages.detach() * logps).mean()
    return loss, rewards, samples
```

The real implementation also records generated lengths, per-sample rewards, advantages, and loss components.

## 12. Training loop

For every step:

1. `optimizer.zero_grad()`
2. choose a training problem,
3. compute the GRPO loss from a rollout group,
4. `loss.backward()`,
5. clip gradient norm,
6. `optimizer.step()`,
7. log loss, mean reward, and response length,
8. periodically inspect samples and save checkpoints.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for example in training_data:
    optimizer.zero_grad()
    loss, rewards, samples = compute_grpo_loss(...)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

Save a checkpoint on `KeyboardInterrupt` as well as on a normal schedule. Logging raw samples matters because a scalar reward can hide degeneration, formatting hacks, or incoherent output.

## 13. Compute and stability

GRPO is expensive because each training example requires several long generations plus differentiable forward passes.

- Reducing rollout count or maximum tokens lowers memory and time but weakens the learning signal.
- The lecture notes that useful quality required around 8 rollouts and 512 maximum new tokens in its setup.
- A smaller demonstration may use 2 rollouts/256 tokens but is not expected to produce a strong model.
- The straightforward notebook avoids batching for readability; companion scripts provide batched/multi-GPU variants.
- More training is not automatically better. Unstabilized GRPO can peak and then degrade.

## 14. Reference result

| Model | MATH-500 accuracy | Average output tokens |
|---|---:|---:|
| Qwen3-0.6B Base | 15.2% | 78.85 |
| Official reasoning variant | 48.2% | 1369.79 |
| Base after simplified GRPO, step 50 | 47.4% | 586.11 |

After only 50 steps, the educational GRPO run nearly matched the official reasoning variant's accuracy while using shorter outputs. This is a single configuration, not evidence that short training universally reproduces a production reasoning model.

Training longer can make performance worse; evaluate checkpoints instead of assuming the last one is best.

## 15. Common mistakes

- Training on MATH-500 and then reporting MATH-500 evaluation.
- Allowing loose answer fallbacks in the reward and accidentally rewarding malformed output.
- Using `torch.inference_mode()` for rollout tensors later involved in training.
- Averaging rollout logprobs when the intended objective uses a sequence sum.
- Calling `.item()` before backpropagation.
- Forgetting to detach advantages.
- Using too few rollouts, so groups frequently have identical rewards.
- Watching only the loss and not generated text, reward, length, and benchmark checkpoints.

## Takeaways

- RLVR replaces an expensive learned reward model with a deterministic verifier where the task permits it.
- GRPO removes PPO's critic by comparing several completions for the same prompt.
- Relative advantages provide no signal when all rollout rewards match.
- Outcome-only rewards can produce useful reasoning behavior without target reasoning traces.
- The simplified GRPO implementation is educational but unstable; clipping, KL control, richer metrics, and careful checkpoint selection are natural next steps.

