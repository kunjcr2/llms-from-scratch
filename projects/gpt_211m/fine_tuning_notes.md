We learn about **Instructional Finetuning**.

## Lec1 - Creating datasets

- Pretrained models are good at text completions, but not at instructions.
- We use the dataset which looks like below.

```json
{
  "instruction": "here_comes-the_instruction",
  "input": "here_comes-the_input",
  "output": "here_comes_the_output"
}
```

And millions nd billions of the datasets !

And we convert those to -

- We use `AlphaCa prompt` stype by stanford as it MUCH more common. Search up to google to check what that is and how it looks. Another one is `Phi3 prompt`.

> # **CHECK `LLM_full_finetune.ipynb`** for ACTUAL Finetuning thing.

---

## Lec2 - LoRA - Shaw Talebi

> 1. **LoRA** - Fine tuning by adding new trainable params

- Lets say input is `x` and hidden layer is `h(x)`. So basically, `h(x) = W0*x` where we have the hidden layer by multiplying the input with the Weight matrix. This is what we have for "**Not LoRA**".
- In LoRA, we have `h(x) = W0*x + del(W)*x`. We have these additional weight matrix. Lets assume `del(W) = BA`. `h(x) = W0*x + BA*x`, so we have something like `B` an `A` being a vector and and we multipley them and add it to weight matrix and multiply with x to get hidden layer.
  - We have `W0`, frozen here and `B` and `A` are trainable with Far less params compared to the ones before

* Arguments:
  - Rank of the LoRA layer: `W = W' + B*A*x` where `B` and `A`'s dimensions are `(dim x r)` and `(r x dim)`, respectively.
  - lora_alpha: This is basically a number applied to `A @ B`, before it is added to `W0`. We usually do `lora_alpha / rank`, which scales up the updates of model which are `A` and `B`, in normal case we keep `lora_alpha = 2 * rank`.
  - lora_dropout: Its a classic one. We drop 10% of the weights.
  - bias: This basically controls the bias of the training. And we keep none for now, cuz we dont know what that is.
  - task_type: It tells PEFT, what you're doing. There are options like `Tasktype.CAUSAL_LM`, `Tasktype.SEQ_CLS`, `Tasktype.TOKEN_CLS` and `Tasktype.SEQ_2_SEQ_LM` and as we're using GPT-2, we go with `CAUSAL_LM`.

> # **CHECK `LLM_LoRA_finetune.ipynb`** for LoRA Finetuning thing.

---

## Quantization:

- Usually models parameters are saved with float32 datatypes which is basically 4 bytes per parameter and we call it `precision`. But as memomry started increasing, we started reducing float32 to 50% precision at float16. 
- Here we loose a little bot of precision or accuracy by keeping model weight as `7.556` instead of `7.55578`. We save it as 2 byres reducing memory requirements to half.
- This is EXACTLY what we call as `Quantization`. We can save it as float16, float8, int4 and int2; each showing number of bits.

For more info., `WAIT`.
