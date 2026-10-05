---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- dpo
- transformers
- trl
datasets:
- HuggingFaceH4/ultrachat_200k
- HuggingFaceH4/ultrafeedback_binarized
language:
- en
license: apache-2.0
---

# ALL THE MODEL WEIGHTS ARE AT [Huggingface](https://huggingface.co/kunjcr2/qwen2.5-0.5b-sft-dpo)

# Qwen2.5-0.5B SFT + DPO (LoRA)

This repository contains:

* **Merged SFT backbone** trained on [HuggingFaceH4/ultrachat\_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
* **LoRA adapters fine-tuned with DPO** using [HuggingFaceH4/ultrafeedback\_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

Both are packaged in one repo:

* Root = SFT backbone (full weights + tokenizer)
* `dpo_adapters/` = LoRA adapters (DPO preference optimization)

---

## Model Details

* **Developed by:** Independent contributor
* **Base model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
* **Library:** [Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), [TRL](https://huggingface.co/docs/trl)
* **Training type:** Supervised fine-tuning (SFT) + Direct Preference Optimization (DPO)
* **Languages:** English (from UltraChat + UltraFeedback)
* **License:** Apache-2.0

---

## Uses

### Direct Use

* Chat-style assistant
* Text generation, reasoning, and dialogue

### Downstream Use

* Base for alignment research
* Further PEFT fine-tuning

### Out-of-Scope

* High-stakes or safety-critical applications (e.g., medical, legal, political advice)

---

## Bias, Risks, and Limitations

* Biases present in the UltraChat and UltraFeedback datasets will carry over
* Model can generate hallucinated or unsafe outputs
* Should not be deployed without safety filtering

---

## How to Use

### Pipeline only

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="kunjcr2/qwen2.5-0.5b-sft-dpo")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
```

### With DPO adapters

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("your-username/qwen2.5-0.5b-sft-dpo")
base = AutoModelForCausalLM.from_pretrained("your-username/qwen2.5-0.5b-sft-dpo", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "your-username/qwen2.5-0.5b-sft-dpo/dpo_adapters")
```

---

## Training Details

### Data

* **SFT:** UltraChat (100k subset)
* **DPO:** UltraFeedback (binarized)

### Hyperparameters

* **SFT:**

  * Epochs: 1
  * LR: 2e-5
  * Optimizer: AdamW
  * Batch size: 16 × 16 grad accumulation (effective 256)
* **DPO:**

  * Epochs: 1
  * LR: 1e-5
  * Beta: 0.1
  * LoRA rank: 8

### Hardware

* Colab A100 / L4 (mixed bf16/TF32)
* Gradient checkpointing enabled

---

## Evaluation

* **Qualitative:** improved helpfulness and preference alignment over raw SFT
* **Quantitative:** not benchmarked yet (POC)

---

## Environmental Impact

* **Hardware:** single A100 (Colab Pro+)
* **Runtime:** \~few hours total (SFT + DPO)
* **Emissions:** not calculated

---

## Citation

If you use this model, cite Qwen as the base and this repo for the SFT+DPO adapters.

```bibtex
@misc{qwen2.5sftdpo,
  title = {Qwen2.5-0.5B SFT + DPO (LoRA)},
  author = {Your Name},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/kunjcr2/qwen2.5-0.5b-sft-dpo}}
}
```

---

## Framework Versions

* Transformers 4.x
* TRL latest
* PEFT 0.17.1

---