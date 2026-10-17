# Lecture 8 - Reasoning Distillation

**Video:** Build a Reasoning Model From Scratch 8 (series finale)

This lecture trains a small `Qwen3-0.6B` base model to imitate a much larger reasoning teacher. Rather than performing expensive reinforcement learning with verifiable rewards (RLVR), it uses supervised learning on teacher-generated solutions. The practical result is a relatively inexpensive way to transfer much of a teacher's math-reasoning behavior to a smaller student.

## 1. Where distillation fits

The series has built a base model, evaluated it with a verifier, improved its inference behavior, and trained it with RLVR/GRPO. Distillation is an alternative training-time route:

```text
MATH training prompt
       |
       v
large teacher (DeepSeek R1)
       |
       v
teacher reasoning trace + final answer
       |
       v
supervised next-token training
       |
       v
small Qwen3-0.6B student
```

The teacher is DeepSeek R1 (described in the lecture as a 671B-parameter model); the student is the earlier Qwen3-0.6B base model. DeepSeek's R1 work also released smaller models trained to mimic a larger teacher, which is the motivating pattern here.

Distillation does not create new teacher-level reasoning from scratch. It makes the student reproduce examples of it. It can therefore be cheaper than RLVR and particularly attractive when the student is small, though it can inherit teacher mistakes and overfit the teacher-generated distribution.

## 2. Hard versus soft distillation

At every answer position, a teacher produces logits and therefore a distribution over its vocabulary.

| Approach | Target given to the student | Main advantage | Main limitation |
|---|---|---|---|
| **Hard distillation** | The teacher's selected output tokens | Simple to collect and works across different tokenizers | Discards probability mass on alternative tokens |
| **Soft distillation** | Teacher logits or softmax probabilities | Preserves more information at each token position | Requires full teacher distributions and compatible vocabularies |
| **Combined loss** | Token target plus a distribution-matching term | Can use both signals | Requires the conditions for soft targets |

Soft distillation is natural inside one model family: teacher and student can share a tokenizer and vocabulary. It is awkward across model families because a logit index refers to different tokens (and potentially different subword boundaries). Public APIs commonly omit full-vocabulary logits as well. These practical constraints make hard distillation the default for cross-family teacher/student pairs such as DeepSeek-to-Qwen or DeepSeek-to-Llama.

For hard distillation, ordinary autoregressive cross-entropy on the teacher's text is enough:

$$
\mathcal{L}_{hard} = -\sum_t \log p_\theta(y_t^{teacher} \mid x, y_{<t}^{teacher}).
$$

## 3. Off-policy and on-policy data collection

The other distinction is *when* examples are produced.

- **Off-policy:** query the teacher before training, save a fixed dataset, then train the student on it. The tutorial uses this approach.
- **On-policy:** repeatedly let the current student produce responses during training and consult the teacher for targets/distributions relevant to those responses.

Off-policy training is operationally convenient: teacher inference can run separately, data can be inspected and filtered, and student training has no teacher-call latency. On-policy approaches can better track the changing student, but require teacher work during training and are consequently more costly. The lecture notes that hard off-policy distillation is the most common practical choice; soft/on-policy variants may be beneficial but are more expensive and less broadly applicable.

## 4. Dataset: teacher solutions for MATH

Start from the MATH training split: approximately 12,000 problems are used to collect teacher solutions, while MATH-500 remains a disjoint evaluation set. Each teacher record contains:

| Field | Meaning |
|---|---|
| `problem` | Original math prompt |
| `ground_truth_answer` | Reference answer from MATH |
| `message_thinking` | Teacher reasoning trace |
| `message_content` | Teacher's final response |

The transcript's data is collected in advance from DeepSeek R1 and stored as JSON. The full dataset is roughly 100 MB. The reasoning trace and final content must be reconstructed in the same order the model emits them:

```text
<think>
teacher reasoning trace
</think>
teacher final answer
```

Keeping the trace gives the student demonstrations of both solution style and final answer. In contrast, the previous RLVR setup scored only an extracted final answer. This is the central difference: RLVR rewards a verified outcome, whereas distillation trains on every teacher output token.

The same verifier from the earlier lecture can still grade the saved teacher answers by extracting `\\boxed{...}` and comparing it with the reference. This verifies the dataset without running an LLM.

## 5. Tokenization and training examples

Use the reasoning variant's tokenizer and chat template, not an ad-hoc text concatenation. The prompt receives the user/assistant wrapper; the completion is appended without accidentally adding a second wrapper. Each example has the conceptual form:

```text
<user wrapper> problem <end-user wrapper>
<assistant start>
<think> reasoning </think>
final answer <assistant end>
```

The response should terminate with the tokenizer's appropriate assistant/message-end token. In the lecture this is an `im_end`-style token rather than assuming that the generic end-of-text token is correct.

Pre-tokenize once before the epoch loop. Re-tokenizing static strings on every epoch wastes time and makes the data path harder to inspect. Store at least:

- `input_ids`: prompt followed by teacher completion;
- `prompt_length`: boundary between prompt and completion;
- sequence length and any metadata needed for filtering.

### Length filtering

The implementation keeps examples no longer than 2,048 tokens. A long reasoning trace may make an otherwise useful example too expensive for a small single-example training loop. Filtering creates a compute/coverage trade-off: it bounds memory and runtime but biases training toward shorter solutions. Measure the retained count and inspect the discarded length distribution instead of treating the threshold as universally correct.

After filtering, shuffle and split the data into training and validation examples. The validation split is for loss monitoring; MATH-500 is still the held-out task benchmark and must not be used as a training split.

## 6. Loss: train only on the teacher completion

The student is conditioned on the problem, but it should not be trained to *predict the prompt text*. Mask prompt positions out of the loss. With shifted next-token prediction:

```python
logits = model(input_ids).logits[:, :-1]
targets = input_ids[:, 1:]

# Ignore target positions that correspond to prompt tokens.
targets[:, :prompt_length - 1] = IGNORE_INDEX
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=IGNORE_INDEX,
)
```

Exact indexing depends on whether `prompt_length` is measured before or after shifting, so test it on a decoded example. The desired invariant is simple: every loss-bearing target token belongs to the teacher response, including its reasoning tokens and final answer, while prompt targets are ignored.

Token-level log probabilities expose the same calculation:

$$
\log p_\theta(y_t \mid y_{<t}, x) = \operatorname{logsoftmax}(z_t)[y_t].
$$

Negative average completion log probability is the cross-entropy loss. Comparing log probabilities before and after training is a useful unit-level check that the student assigns higher probability to the teacher completion.

## 7. Training loop and diagnostics

The optimizer loop resembles the earlier RLVR loop, but the objective is supervised cross-entropy rather than GRPO:

```text
for epoch:
    shuffle training examples
    for example:
        zero gradients
        compute masked completion cross-entropy
        backpropagate
        clip gradients
        optimizer step
        log metrics and periodically save a checkpoint
```

Multiple epochs are feasible because no teacher generation occurs inside the loop. The lecture uses two epochs as a starting point, recommends exploring three or more, clips gradients at `1`, and logs/saves periodically. A long run is best executed through the provided standalone script rather than an interactive notebook.

Track both train and validation loss. Per-example training loss is naturally noisy; validation loss averaged over a validation subset is smoother. A falling validation loss indicates better likelihood on held-out teacher data, but it does not replace MATH-500 evaluation or detect all forms of imitation-induced overfitting.

The reported hardware example completes about two epochs / roughly 12,000--13,000 single-example steps in two to three hours on a DGX Spark; the script estimate is roughly 15 GB RAM and three to four hours depending on hardware. Treat these as teaching-run measurements, not capacity guarantees.

## 8. Evaluation and reported comparison

Evaluate saved checkpoints on MATH-500 with the reasoning tokenizer because the training completions use `think` tokens. Load the custom checkpoint weights while preserving that tokenizer configuration. Allow more generation tokens at evaluation than the 2,048-token training cap so a valid answer is not unnecessarily truncated.

The lecture reports that DeepSeek-R1 hard distillation more than doubles the small base model's MATH-500 accuracy after the first epoch, improves further around the second epoch (about 32% is mentioned), and continues improving over three epochs. A Qwen3-235B teacher produced stronger results in the reported comparison, plausibly because its model family and answer format are closer to the Qwen student.

These are experiment-specific results. Teacher quality, prompt format, data filtering, tokenizer choice, sequence limit, learning rate, epoch count, decoding settings, and evaluation verifier all affect the outcome. Compare checkpoints rather than assuming the final one is best.

## 9. Distillation versus RLVR

| Situation | Practical default suggested by the lecture |
|---|---|
| Training a frontier-scale model with a strong verifier | RLVR can be the right core method for developing reasoning behavior. |
| Improving a much smaller model with access to a capable teacher's outputs | Hard distillation is often cheaper and can outperform small-model RLVR. |
| Teacher/student share a tokenizer and full teacher logits are available | Soft or mixed distillation becomes practical to test. |
| Tokenizers differ or teacher probabilities are unavailable | Use hard distillation. |

This is a rule of thumb, not a theorem. Distillation leverages the expensive work already embodied in a teacher; RLVR can optimize behavior beyond a fixed demonstration set when rewards are reliable and training scale supports it. The methods can also be staged: distill a capable initial student, then apply carefully designed RLVR.

## 10. Future directions and exercises

- Study process reward models and graders that use intermediate reasoning, while considering the safety and transparency implications of exposing traces.
- Extend verifiable-reward methods beyond math and code, provided the domain has a robust correctness signal and defenses against reward hacking.
- Train or infer with adjustable reasoning effort so easy prompts do not always incur maximum reasoning cost.
- Repeat the pipeline with a larger student or batched execution if compute permits.
- Plot smoothed training loss beside validation loss, inspect the onset of overfitting, and evaluate multiple checkpoints on MATH-500.
- Compare teacher families, token limits, epoch counts, and data filters using the same verifier and fixed evaluation protocol.

## Takeaways

- Hard off-policy distillation is the practical path when models use different tokenizers or the teacher does not expose full logits.
- A teacher-generated reasoning trace plus final answer becomes supervised next-token data after correct chat formatting and masking.
- Pre-tokenization, completion-only loss masking, length filtering, and tokenizer compatibility are the implementation details most likely to determine whether the run is valid.
- Validation likelihood and benchmark accuracy answer different questions; use both, and retain checkpoints.
- For a small student, imitating a strong teacher can be a more compute-efficient route to reasoning performance than applying RLVR from scratch.
