!pip install -q trl peft datasets
!pip uninstall -y -q torchao

# Check numpy did not get bumped. If this is not 2.0.x, stop and restart runtime.
import numpy; print("numpy:", numpy.__version__)

import re, torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

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
    output_dir="qwen05b-gsm8k-grpo",
 
    # --- GRPO core
    num_generations=8,              # G: completions per prompt
    max_completion_length=MAX_COMPLETION,
    temperature=1.0,                # needs > 0 or all 8 samples are identical
    beta=0.0,                      # KL coef. 0.0 skips loading the ref model, making it 0 since this shi never worked
                                    # entirely (faster). 0.02 keeps the leash on.
    epsilon=0.2,                    # clip range
    num_iterations=1,               # 1 grad step per rollout -> ratio == 1
    loss_type="dapo",               # length-unbiased; TRL's current default
    scale_rewards="group",          # divide advantage by within-group std
    # reward_weights=[1.0, 0.2],      # correctness dominates, format is a nudge
    mask_truncated_completions=True,  # do not punish answers that got cut off
 
    # --- optimization
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    max_grad_norm=0.2,
    bf16=True,
    gradient_checkpointing=True,
 
    # --- vLLM (colocate = same GPU, no separate server)
    use_vllm=USE_VLLM,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
 
    # --- logging
    logging_steps=5,
    save_steps=50,
    log_completions=True,
    num_completions_to_print=2,
    report_to="none",
    # run_name="qwen05b-gsm8k-grpo-b0.02",
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

# import os
# os.environ["WANDB_PROJECT"] = "gsm8k-grpo"

# # before creating the trainer
# wandb.init(project="gsm8k-grpo", name="qwen05b-grpo-b0.02",
#            config={"baseline_acc": 0.4208, "baseline_fmt": 0.0008,
#                    "n_train": N_TRAIN, "beta": 0.02})

trainer.train()
# wandb.finish()
trainer.save_model("qwen05b-gsm8k-grpo/final")
print("saved to qwen05b-gsm8k-grpo/final")

import pandas as pd, matplotlib.pyplot as plt
 
df = pd.DataFrame(trainer.state.log_history)
df = df[df["reward"].notna()] if "reward" in df else df
 
panels = [
    ("reward",                          "total reward (weighted sum)"),
    ("reward_std",                      "reward_std  -- DEAD if near 0"),
    ("rewards/correctness_reward/mean", "correctness  (baseline 0.421)"),
    ("rewards/format_reward/mean",      "format  (baseline 0.001)"),
    ("kl",                              "KL to reference"),
    ("clip_ratio/region_mean",          "clip fraction"),
    ("completions/mean_length",         "mean completion length"),
    ("loss",                            "loss  (ignore -- not a progress metric)"),
]
panels = [(c, t) for c, t in panels if c in df.columns]
 
n = len(panels)
fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(13, 3 * ((n + 1) // 2)))
for ax, (col, title) in zip(axes.flat, panels):
    ax.plot(df["step"], df[col], marker="o", ms=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    if col == "rewards/correctness_reward/mean":
        ax.axhline(0.4208, ls="--", c="r", lw=1)   # your eval baseline
    if col == "reward_std":
        ax.axhline(0.1, ls="--", c="r", lw=1)      # danger line
for ax in axes.flat[n:]:
    ax.axis("off")
plt.tight_layout()
plt.show()
 
# ---- verdict ----
def band(col, frac):
    s = df[col].dropna()
    return s.iloc[: max(1, int(len(s) * frac))].mean(), s.iloc[-max(1, int(len(s) * frac)) :].mean()
 
print(f"{'metric':<34} {'first 20%':>10} {'last 20%':>10} {'delta':>10}")
print("-" * 68)
for col, _ in panels:
    if col == "loss":
        continue
    a, b = band(col, 0.2)
    print(f"{col:<34} {a:>10.4f} {b:>10.4f} {b - a:>+10.4f}")
 
std_last = df["reward_std"].dropna().iloc[-max(1, len(df) // 5):].mean()
print("\nreward_std (last 20%):", round(std_last, 4),
      "-> DEAD, no gradient signal" if std_last < 0.05 else "-> healthy, groups have variance")