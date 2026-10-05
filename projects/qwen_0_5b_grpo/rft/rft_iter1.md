# Current GSM8K Experiment Iteration

Model: `Qwen/Qwen2.5-0.5B-Instruct`

## What We Are Doing

We are testing different post-training methods on GSM8K while keeping evaluation consistent.

Current sequence:

1. Evaluate the base Qwen2.5-0.5B model.
2. Train with standard SFT.
3. Train with GRPO.
4. Generate rejection-sampling data:

   * Sample 8 solutions per GSM8K training question.
   * Keep only correct, unique model-generated solutions.
   * Train a LoRA adapter on those filtered solutions.
5. Upload the RFT LoRA adapter and generated RFT dataset to Hugging Face for reuse.
6. Load the RFT LoRA checkpoint from Hugging Face.
7. Continue training that checkpoint with GRPO.
8. Log the RFT → GRPO run with Weights & Biases.
9. Evaluate the final model on the same full 1,319-example GSM8K test set.

## Hugging Face Artifacts

Base model:

`Qwen/Qwen2.5-0.5B-Instruct`

RFT LoRA:

`kunjcr2/qwen2.5-0.5b-gsm8k-rft-lora`

RFT dataset:

`kunjcr2/gsm8k-rft-qwen2.5-0.5b`

## Current Experiment

**RFT → GRPO**

The goal is to initialize GRPO from the rejection-sampling LoRA checkpoint and compare it against GRPO initialized directly from the base model.
