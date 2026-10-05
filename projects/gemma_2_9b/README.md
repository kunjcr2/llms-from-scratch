---
library_name: peft
language:
- en
base_model:
- google/gemma-2-9b-it
pipeline_tag: text-generation
tags:
- lora
- math
- proof-writing
- sft
- gemma-2
---

# gemma-2-9b-proof-lora

A LoRA adapter fine-tuned on top of `google/gemma-2-9b-it` for **formal mathematical proof writing**. Given a theorem or mathematical statement, the model produces a rigorous, step-by-step proof with clearly labelled structure (Given → Definitions → Proof → QED).

The base model could write vague, hand-wavy math. After fine-tuning it writes structured, logically explicit proofs.

---

## Model Details

- **Developed by:** Kunj Shah ([@kunjcr2](https://huggingface.co/kunjcr2))
- **Model type:** Causal LM — LoRA adapter (PEFT)
- **Base model:** `google/gemma-2-9b-it`
- **Language:** English
- **Finetuned from:** `google/gemma-2-9b-it`
- **Training logs:** [W&B Run](https://wandb.ai/kunjcr2-dreamable/gemma-9b-lora/runs/4kxabh7l)
- **Model card:** [gemma-2-9b-proof-lora](https://huggingface.co/kunjcr2/gemma-2-9b-proof-lora)

---

## What Changed vs Base Model

Before fine-tuning, Gemma-2-9B-IT produces informal, conversational math responses when asked to prove something — it gestures at the right idea but skips logical steps and has no consistent structure.

After fine-tuning, the model writes proofs with:
- Explicit logical steps with clear connectives ("therefore", "it follows that", "by definition")
- Proper proof structure (Given, Proof, QED)
- No skipped justifications
- Correct proof termination

This is a **style and structure** shift, not a reasoning capability shift. The model already knew the math — SFT made the reasoning visible and explicit.

---

## Training Details

### Dataset
- **Source:** `open-web-math/open-web-math` — 6.3M documents of high-quality mathematical text (forums, lecture notes, arXiv, math blogs) with LaTeX preserved
- **Filtered to:** ~120,000 proof-shaped samples containing both theorem keywords (Theorem, Lemma, Proposition) and proof markers (Proof., QED, \square)
- **Format:** Gemma-2-IT chat format with completion-only loss — model supervised only on the proof, not the prompt

### Hyperparameters

| | |
|---|---|
| Epochs | 3 |
| Effective batch size | 32 (batch 2 × grad accum 16) |
| Learning rate | 1.5e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Max sequence length | 1536 |
| Optimizer | Paged AdamW 32-bit |
| Precision | bf16 + tf32 |
| Hardware | NVIDIA A100-SXM4-80GB |
| Training time | 9.59 hours |

### LoRA Config

| | |
|---|---|
| Rank (r) | 32 |
| Alpha | 64 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 108,036,096 (1.15% of total) |

---

## Results

| Metric | Value |
|---|---|
| Final train loss | 0.383 |
| Final train mean token accuracy | 0.886 |
| Best eval loss | 0.846 (step 1000) |
| Final eval loss | 0.856 |
| Final eval mean token accuracy | ~0.782 |

Train loss dropped from ~1.4 → 0.38 over 3 epochs. Eval loss stayed stable with no significant overfitting. The oscillations in training curves are expected due to `group_by_length` batching — short and long sequences alternate, causing variance in per-batch loss.

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "kunjcr2/gemma-2-9b-proof-lora")
tok = AutoTokenizer.from_pretrained("kunjcr2/gemma-2-9b-proof-lora")

prompt = (
    "<bos><start_of_turn>user\n"
    "You are a formal mathematics assistant. Prove the following:\n\n"
    "The square root of 2 is irrational."
    "<end_of_turn>\n<start_of_turn>model\n"
)

ids = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=512, temperature=0.3, do_sample=True)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Limitations

- This is an SFT-only run — the model learned proof *structure and style*, not verified logical correctness. A wrong proof written in a rigorous format will still look convincing.
- Dataset is general mathematical text, not curated theorem-proof pairs, so quality varies.
- For verified correctness, a GRPO stage with a structural + mathematical reward signal would be the next step.

---

## Citation

If you use this adapter, please cite the base model and dataset:

```bibtex
@misc{gemma2_proof_lora,
  author = {Kunj Shah},
  title = {gemma-2-9b-proof-lora},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/kunjcr2/gemma-2-9b-proof-lora}
}
```