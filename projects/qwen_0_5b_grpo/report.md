# Qwen2.5-0.5B GSM8K Post-Training Experiments

## Overview

Comparison of **SFT, weighted SFT, rejection-sampling fine-tuning (RFT), GRPO, Dr. GRPO, and GSPO** on `Qwen/Qwen2.5-0.5B-Instruct`.

**Evaluation:** full GSM8K test split (`1,319` problems), greedy decoding, exact numeric-answer accuracy.

## Methods

### Standard SFT

LoRA fine-tuning on GSM8K gold solutions.

- LoRA rank: `16`
- LR: `2e-4`
- Epochs: `2`
- Completion-only cross-entropy
- `####` format rate: ~`99.85%`

### Weighted SFT

Modified token-level objective:

- Normal CE weight: `1×`
- Final-answer/correctness weight: `5×`

### RFT

Rejection-sampling fine-tuning:

1. Sample `8` solutions per training problem.
2. Keep correct, unique trajectories.
3. LoRA-SFT on retained generations.

### GRPO

Online RL using fresh sampled completions and binary answer-correctness reward.

- `num_generations = 8`
- `beta = 0`
- Group-relative advantage optimization

### Dr. GRPO

GRPO variant reducing normalization-induced bias.

- `loss_type = "dr_grpo"`
- `scale_rewards = False`
- No group reward-std normalization
- Constant completion-length normalization

### GSPO

Group Sequence Policy Optimization using **sequence-level importance ratios** instead of token-level importance ratios.

## Results

| Method                                                |         GSM8K Accuracy |
| ----------------------------------------------------- | ---------------------: |
| Base                                                  |  **42.08%** (555/1319) |
| Base → SFT (LoRA)                                     |  **32.30%** (426/1319) |
| Base → Weighted SFT (LoRA) (`1× CE + 5× correctness`) |             **<24.x%** |
| Base → SFT (Full)                                     |               **<30%** |
| Base → RFT                                            |             **~37.4%** |
| Base → GRPO                                           |  **45.03%** (594/1319) |
| Base → RFT → GRPO @ 300                               |  **36.54%** (482/1319) |
| Base → RFT → GRPO @ 500                               |  **34.95%** (461/1319) |
| Base → Dr. GRPO @ 200                                 |  **46.02%** (607/1319) |
| Base → Dr. GRPO @ 250                                 | **44.66%** (~589/1319) |
| Base → GSPO @ 200                                     |             **45.72%** |
| Base → GSPO @ 250                                     |             **44.58%** |

## Findings

- **Dr. GRPO @ 200 is the best observed checkpoint:** `42.08% → 46.02%` (`+3.94 pp`).
- GSPO also improved over base, peaking at **45.72%**, slightly below Dr. GRPO.
- Standard GRPO reached **45.03%**, so all direct RL methods outperformed the base model.
- SFT, weighted SFT, and RFT all reduced held-out accuracy.
- RFT → GRPO degraded progressively: `~37.4% → 36.54% → 34.95%`.
- Both Dr. GRPO and GSPO peaked around **step 200** and regressed by step 250.
- Training reward was not a reliable proxy for held-out greedy accuracy, making **intermediate checkpoint evaluation important**.

## Conclusion

For Qwen2.5-0.5B, **direct RL post-training consistently outperformed supervised alternatives**. Dr. GRPO produced the best observed result at **46.02%**, followed by GSPO at **45.72%** and standard GRPO at **45.03%**.
