## RFT -> GRPO

!pip install -q trl peft datasets wandb huggingface_hub
!pip uninstall -y -q torchao

import os, re, torch, wandb, trl
from google.colab import userdata
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

login(token=userdata.get("HF_TOKEN"))
wandb.login(key=userdata.get("WANDB_API"))

os.environ["WANDB_PROJECT"] = "qwen05b-gsm8k-rl"
wandb.init(
    entity="kunjcr2-dreamable",   # personal username
    project="qwen05b-gsm8k-rl",
    name="RFT-to-GRPO-500steps",
)

print("TRL:", trl.__version__)
print("GPU:", torch.cuda.get_device_name(0))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Change this if you used a different HF repo name
RFT_ADAPTER = "kunjcr2/qwen2.5-0.5b-gsm8k-rft-lora"

OUT = "qwen05b-gsm8k-rft-grpo"

SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, "
    "then give the final numeric answer on its own last line in exactly "
    "this format:\n#### <answer>"
)

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _clean(s):
    return s.replace(",", "").replace("$", "")

raw = load_dataset("openai/gsm8k", "main", split="train")

def build_row(ex):
    gold = _clean(ex["answer"].split("####")[-1]).strip()

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ],
        "gold": gold,
    }

train_ds = raw.map(
    build_row,
    remove_columns=raw.column_names
)

print(train_ds)
print(train_ds[0])

def extract_pred(text):
    text = _clean(text)

    if "####" in text:
        nums = NUM_RE.findall(text.split("####")[-1])
        if nums:
            return nums[0]

    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def same_number(a, b):
    if a is None or b is None:
        return False

    try:
        return abs(float(a) - float(b)) < 1e-4
    except:
        return False


def correctness_reward(
    completions,
    gold,
    log_metric=None,
    log_extra=None,
    **kwargs
):
    # Conversational GRPO completions are message lists
    texts = [
        c[-1]["content"] if isinstance(c, list) else c
        for c in completions
    ]

    preds = [extract_pred(x) for x in texts]

    rewards = [
        1.0 if same_number(pred, target) else 0.0
        for pred, target in zip(preds, gold)
    ]

    # Custom W&B metric
    if log_metric:
        log_metric(
            "batch_accuracy",
            sum(rewards) / len(rewards)
        )

    if log_extra:
        log_extra(
            "predicted_answer",
            [p if p is not None else "[none]" for p in preds]
        )

    return rewards

tok = AutoTokenizer.from_pretrained(MODEL)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(
    base,
    RFT_ADAPTER,
    is_trainable=True,   # IMPORTANT
)

model.print_trainable_parameters()

args = GRPOConfig(
    output_dir=OUT,

    # optimization
    learning_rate=2e-6,
    max_steps=500,
    warmup_steps=20,
    lr_scheduler_type="cosine",

    # 32 completion sequences / step
    # 8 generations per prompt
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    num_generations=8,

    # rollout
    max_completion_length=400,
    temperature=0.8,
    top_p=0.95,

    # Your successful previous GRPO setup
    beta=0.0,

    bf16=True,
    gradient_checkpointing=True,

    # logging
    logging_steps=1,
    report_to="wandb",
    run_name="RFT-to-GRPO-500steps",

    log_completions=True,
    num_completions_to_print=4,

    # checkpointing
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    remove_unused_columns=False,
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    reward_funcs=correctness_reward,
    train_dataset=train_ds,
    args=args,
)

trainer.train()

trainer.save_model(f"{OUT}/final")

wandb.finish()