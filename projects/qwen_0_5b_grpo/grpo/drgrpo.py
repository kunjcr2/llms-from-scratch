from google.colab import userdata
import os
import wandb

# Close any stale/previous run
if wandb.run is not None:
    wandb.finish()

# Force the correct destination
os.environ["WANDB_ENTITY"] = "kunjcr2-dreamable"
os.environ["WANDB_PROJECT"] = "qwen05b-gsm8k-rl"

# Login
wandb.login(
    key=userdata.get("WANDB_API"),
    relogin=True,
)

# Explicitly create the run in the correct workspace
run = wandb.init(
    entity="kunjcr2-dreamable",
    project="qwen05b-gsm8k-rl",
    name="qwen05b-base-drgrpo-b0",
    reinit="finish_previous",
)

print("Entity :", run.entity)
print("Project:", run.project)
print("Run    :", run.name)
print("URL    :", run.url)

assert run.entity == "kunjcr2-dreamable"
assert run.project == "qwen05b-gsm8k-rl"

!pip install -q trl peft datasets
!pip uninstall -y -q torchao

# Check numpy did not get bumped. If this is not 2.0.x, stop and restart runtime.
import numpy; print("numpy:", numpy.__version__)

import re, torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

import re

# ---------------------------------------------------------------- config
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN = 4000          # prompts to train on (7473 available)
USE_VLLM = False        # True = ~5x faster, but installing vllm may break Colab
MAX_COMPLETION = 400    # must match your eval, or before/after is not comparable

# IDENTICAL to the eval script. If you change this, your 42.08% baseline
# stops being a valid comparison and the whole experiment is meaningless.
SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step, "
    "then give the final numeric answer on its own last line in exactly "
    "this format:\n#### <answer>"
)

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _clean(s):
    return s.replace(",", "").replace("$", "")


def extract_pred(text):
    if "####" in text:
        nums = NUM_RE.findall(_clean(text.split("####")[-1]))
        if nums:
            return nums[0]
    nums = NUM_RE.findall(_clean(text))
    return nums[-1] if nums else None


def same_number(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return False


def _text(c):
    """Completions arrive as [{'role':'assistant','content':...}] for
    conversational prompts, or as a plain string otherwise."""
    return c[-1]["content"] if isinstance(c, list) else c


# ---------------------------------------------------------------- rewards
# TRL passes every extra dataset column through as a kwarg. Our "gold" column
# arrives as gold=[...] aligned with completions. **kwargs is REQUIRED for
# forward compatibility -- TRL passes extra args you do not ask for.
def correctness_reward(completions, gold, **kwargs):
    return [
        1.0 if same_number(extract_pred(_text(c)), g) else 0.0
        for c, g in zip(completions, gold)
    ]


# def format_reward(completions, **kwargs):
#     # Your baseline format_rate was 0.0008, so this starts near zero and has
#     # room to move -- it is your fast confirmation the loop is learning at all.
#     return [1.0 if "####" in _text(c) else 0.0 for c in completions]


# ---------------------------------------------------------------- data
ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.shuffle(seed=0).select(range(N_TRAIN))


def to_grpo(ex):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ],
        # named "gold" not "answer" -- must match the reward fn parameter name
        "gold": _clean(ex["answer"].split("####")[-1]).strip(),
    }

ds = ds.map(to_grpo, remove_columns=ds.column_names)
print(ds[0])

args = GRPOConfig(
    output_dir="qwen05b-gsm8k-drgrpo",

    num_generations=8,
    max_completion_length=MAX_COMPLETION,
    temperature=1.0,

    beta=0.0,
    epsilon=0.2,
    num_iterations=1,

    # Dr. GRPO
    loss_type="dr_grpo", # Was grpo before, but now we wanna try dr grpo wher we dont scale the advantages
    scale_rewards=False,

    mask_truncated_completions=True,

    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    num_train_epochs=1,

    max_grad_norm=0.2,
    bf16=True,
    gradient_checkpointing=True,

    use_vllm=USE_VLLM,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,

    logging_steps=5,
    save_steps=50,

    log_completions=True,
    num_completions_to_print=2,

    report_to="wandb",
    run_name="qwen05b-base-drgrpo-b0",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

trainer = GRPOTrainer(
    model=MODEL,
    args=args,
    reward_funcs=[correctness_reward], # removed format reward
    train_dataset=ds,
    peft_config=peft_config,   # with LoRA, the ref model is just the adapters
                               # disabled -- no second copy of weights in VRAM
)

trainer.train()

trainer.save_model(
    "qwen05b-gsm8k-drgrpo/final"
)

print("Saved to qwen05b-gsm8k-drgrpo/final")

wandb.finish()