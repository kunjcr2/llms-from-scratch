!pip install -q trl peft datasets
!pip uninstall -y -q torchao

import numpy; print("numpy:", numpy.__version__)   # must be 2.0.x

import json, re, torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "qwen05b-gsm8k-sft"

# IDENTICAL to your baseline eval and your GRPO run. Do not touch it -- it is
# the only thing making baseline / SFT / GRPO comparable.
SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, "
    "then give the final numeric answer on its own last line in exactly "
    "this format:\n#### <answer>"
)

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
CALC_RE = re.compile(r"<<[^>]*>>")

def _clean(s):
    return s.replace(",", "").replace("$", "")


def strip_calc(sol):
    """GSM8K solutions embed calculator annotations: 'She has 48/2 = <<48/2=24>>24'.
    They are tool artifacts, not reasoning. Training on them teaches the model to
    emit junk tokens, and your last-number fallback would read digits out of them."""
    return CALC_RE.sub("", sol).strip()


# ---------------------------------------------------------------- data
def build_dataset(n=None):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if n:
        ds = ds.shuffle(seed=0).select(range(n))

    def fmt(ex):
        # GSM8K gold already ends in "#### <number>", which is exactly the format
        # the system prompt asks for. So SFT teaches format for free -- the thing
        # GRPO actively unlearned.
        sol = strip_calc(ex["answer"])
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "completion": [{"role": "assistant", "content": sol}],
        }

    return ds.map(fmt, remove_columns=ds.column_names)

import torch
import torch.nn.functional as F

class AnswerWeightedSFTTrainer(SFTTrainer):
    def __init__(self, *args, answer_weight=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_weight = answer_weight

        tok = self.processing_class
        self.marker_ids = tok.encode("####", add_special_tokens=False)

    def _answer_mask(self, labels):
        mask = torch.zeros_like(labels, dtype=torch.bool)

        marker = torch.tensor(
            self.marker_ids,
            device=labels.device,
            dtype=labels.dtype
        )

        for b in range(labels.size(0)):
            row = labels[b]

            for i in range(len(row) - len(marker) + 1):
                if torch.equal(row[i:i+len(marker)], marker):
                    start = i + len(marker)

                    mask[b, start:] = row[start:] != -100

        return mask

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]

        # IMPORTANT:
        # Do NOT send labels into TRL's patched model forward.
        # Otherwise TRL computes its own chunked CE and returns logits=None.
        model_inputs = {
            k: v for k, v in inputs.items()
            if k != "labels"
        }

        outputs = model(**model_inputs)
        logits = outputs.logits

        # causal shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        valid = shift_labels != -100

        answer_mask = self._answer_mask(labels)[:, 1:]

        weights = torch.ones_like(token_loss)
        weights[answer_mask] = self.answer_weight
        weights *= valid.float()

        loss = (token_loss * weights).sum() / weights.sum().clamp_min(1)

        return (loss, outputs) if return_outputs else loss

def train_sft(n=None, epochs=2):
    ds = build_dataset(n)
    print(f"examples: {len(ds)}")
    print(ds[0]["completion"][0]["content"][:300])

    args = SFTConfig(
        output_dir=OUT,
        max_length=640,             # GSM8K prompt ~180 tok + solution ~200 tok
        packing=False,              # keep prompt/completion masking intact
        # completion_only_loss defaults True for prompt-completion datasets:
        # loss is masked on the system prompt and question automatically.
        learning_rate=1e-4,         # LoRA wants ~100x the GRPO LR
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        seed=42,
    )

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    trainer = AnswerWeightedSFTTrainer(
        model=MODEL,
        args=args,
        train_dataset=ds,
        # peft_config=peft_config,
        answer_weight=5.0,
    )
    trainer.train()
    trainer.save_model(f"{OUT}/final")
    print(f"saved {OUT}/final")
    return trainer


# ---------------------------------------------------------------- eval
def extract_pred(text):
    if "####" in text:
        nums = NUM_RE.findall(_clean(text.split("####")[-1]))
        if nums:
            return nums[0], True
    nums = NUM_RE.findall(_clean(text))
    return (nums[-1], False) if nums else (None, False)


def same_number(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return False


def evaluate(adapter=None, limit=None, batch_size=256, max_new_tokens=400,
             tag="model"):
    """Same prompts, same greedy decoding, same 1319 test problems as your
    baseline. adapter=None evaluates the raw base model."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=dtype, device_map="cuda", attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=dtype, device_map="cuda", attn_implementation="sdpa")
    if not adapter: # just a small trick to go to else block directly
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()      # fold LoRA in -> faster generation
    else:
        model = AutoModelForCausalLM.from_pretrained(adapter, dtype=dtype, device_map="cuda", attn_implementation="sdpa")
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(limit))

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
        for q in ds["question"]
    ]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

    n_ok = n_fmt = 0
    for b in range(0, len(order), batch_size):
        idxs = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt",
                  padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        comps = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        for i, c in zip(idxs, comps):
            pred, fmt = extract_pred(c)
            n_ok += same_number(pred, _clean(ds[i]["answer"].split("####")[-1]).strip())
            n_fmt += fmt
        print(f"  {b + len(idxs)}/{len(order)}", flush=True)

    n = len(order)
    print(f"\n{tag:<10} accuracy {n_ok/n:.4f} ({n_ok}/{n})   "
          f"format {n_fmt/n:.4f}")
    del model; torch.cuda.empty_cache()
    return {"tag": tag, "accuracy": n_ok / n, "format_rate": n_fmt / n}

trainer = train_sft(n=None, epochs=2)        # all 7473, ~470 steps, changed to 1
sft_res = evaluate(adapter=f"{OUT}/checkpoint-468", tag="SFT")