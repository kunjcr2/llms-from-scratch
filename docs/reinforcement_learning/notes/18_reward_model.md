# Training a Reward Model (RM) for RLHF — Beginner-Friendly, End-to-End

> Goal: teach a small neural net to **score** model answers so that _good_ answers (as judged by people) get **higher scores** than _bad_ ones. You’ll use this RM later to guide PPO or to evaluate DPO runs.

---

## 0) What is a Reward Model?

- Input: **prompt** + **candidate answer** (text)
- Output: **one number** (the “reward” / quality score)
- Training data: **preference pairs** collected from humans
  For the **same prompt**, humans pick which of two answers is better:

  ```
  prompt:  "Explain gravity to a 5-year-old."
  chosen:  "Gravity is like an invisible friend that pulls things toward the ground."
  rejected:"Gravity is a force that results from curvature of spacetime described by general relativity."
  ```

- Objective: make the RM score the **chosen** higher than the **rejected**.

---

## 1) Data Schema (what you need)

At minimum, each example must have:

- `prompt` (string)
- `chosen` (string) → preferred response
- `rejected` (string) → less preferred response

> You can also include metadata (annotator IDs, reasons, task tags), but the above three fields are enough.

**Formatting tip:** concatenate prompt and answer so the encoder “sees” the context:

```
### Prompt:
{prompt}

### Answer:
{answer}
```

---

## 2) Model Architecture (simple & practical)

A **text encoder** (e.g., DistilBERT, DeBERTa-V3) → **pooling** → **linear head** that outputs **1 scalar**:

```
tokens -> encoder -> [CLS] or mean pooling -> Linear(hidden -> 1) -> reward score
```

- Encoder: `distilbert-base-uncased` (fast) or `microsoft/deberta-v3-small` (stronger)
- Pooling: use `pooler_output` if available; otherwise **mean-pool** last hidden states
- Head: single linear layer is enough

---

## 3) Loss Function (pairwise Bradley–Terry)

For each pair:

- Let `s⁺ = RM(prompt, chosen)`
- Let `s⁻ = RM(prompt, rejected)`
- We want `s⁺ > s⁻`. Use the **pairwise** loss:

$$
\mathcal{L} = -\log \sigma(s^{+} - s^{-}) \;\;=\;\; \text{softplus}\big(-(s^{+} - s^{-})\big)
$$

- `σ` is sigmoid; we implement the **numerically stable** form with `softplus`
- Minimizing this increases the gap `s⁺ − s⁻` in the right direction

---

## 4) Batching & Tokenization

- Use a **collate function** to tokenize **chosen** and **rejected** separately
- Truncate long texts (e.g., `max_length=512`) to fit memory
- Keep **prompt formatting consistent** so the model learns the right structure

---

## 5) Metrics (quick checks)

- **Pairwise Accuracy**: % of pairs where `s⁺ > s⁻` on a **held-out** set
- (Optional) AUC over `s⁺ − s⁻`
- Watch **overfitting** (train acc ≫ eval acc)

---

## 6) Hyperparameters (good starting points)

- Base encoder: `deberta-v3-small` or `distilbert-base-uncased`
- LR: `2e-5` (AdamW, `weight_decay=0.01`)
- Batch size: 8–32 (fit to your GPU)
- Max length: 512–1024 (longer often helps, costs VRAM)
- Epochs: 1–3 (small datasets overfit quickly)
- Tricks when low on VRAM: **freeze encoder** and train only the head; use **fp16**; reduce `max_length`

---

## 7) Practical Pitfalls (and fixes)

- **Truncation bias**: If answers are long, important info may be cut—raise `max_length` or place crucial info at the **front** (prompt template).
- **Reward hacking**: The RM may latch onto spurious patterns (e.g., “Yes!”).
  Mitigation: diverse data, length penalty in PPO, separate calibration checks.
- **Domain shift**: If PPO prompts differ from RM training prompts, RM may mis-score.
  Mitigation: include similar prompts in RM training.

---

## 8) Using the RM later

- **PPO**: policy generates → RM scores → PPO updates policy (with a KL penalty)
- **DPO**: doesn’t use an RM (optimizes directly from preference pairs), but your RM remains useful for **evaluation** or **guardrails**

---

## 9) Reproducibility

- Set seeds, log hyperparams, keep versions: `transformers`, `torch`, `datasets`
- Save: tokenizer, encoder config, head weights

