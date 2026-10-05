## RFT (Rejection Sampled Fine Tuning)

!pip install -q trl peft datasets
!pip uninstall -y -q torchao

import os, re, json, torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "qwen05b-gsm8k-rft"

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
            return nums[0], True

    nums = NUM_RE.findall(_clean(text))
    return (nums[-1], False) if nums else (None, False)

def same_number(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except:
        return False

set_seed(42)

import os, json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

N_SAMPLES = 8
TEMPERATURE = 0.8
TOP_P = 0.95

# A100: start here.
# 64 prompts x 8 generations = 512 sequences.
# If memory is comfortable, try 96 or 128.
GEN_BATCH = 64

MAX_NEW_TOKENS = 400
SAVE_JSONL = "/content/rft_correct.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)

model.eval()

ds = load_dataset("openai/gsm8k", "main", split="train")

# -------------------------------------------------------------
# Resume support
# -------------------------------------------------------------

processed = set()

if os.path.exists(SAVE_JSONL):
    with open(SAVE_JSONL, "r") as f:
        for line in f:
            row = json.loads(line)
            processed.add(row["qid"])

print(f"Already processed: {len(processed)} / {len(ds)}")

# -------------------------------------------------------------
# Generation
# -------------------------------------------------------------

for start in range(0, len(ds), GEN_BATCH):

    ids = [
        i
        for i in range(start, min(start + GEN_BATCH, len(ds)))
        if i not in processed
    ]

    if not ids:
        continue

    prompts = [
        tok.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": ds[i]["question"],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in ids
    ]

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **enc,

                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,

                num_return_sequences=N_SAMPLES,

                max_new_tokens=MAX_NEW_TOKENS,

                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

    except torch.cuda.OutOfMemoryError:
        print("\nOOM!")
        print(f"GEN_BATCH={GEN_BATCH} is too large.")
        print("Restart and try GEN_BATCH=32.")
        raise

    prompt_len = enc["input_ids"].shape[1]

    completions = tok.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
    )

    # ---------------------------------------------------------
    # Filter correct trajectories
    # ---------------------------------------------------------

    with open(SAVE_JSONL, "a") as f:

        for local_idx, qid in enumerate(ids):

            gold = _clean(
                ds[qid]["answer"].split("####")[-1]
            ).strip()

            seen = set()
            kept = 0

            for k in range(N_SAMPLES):

                idx = local_idx * N_SAMPLES + k

                completion = completions[idx].strip()

                pred, fmt = extract_pred(completion)

                if not same_number(pred, gold):
                    continue

                # Deduplicate identical solutions
                normalized = " ".join(completion.split())

                if normalized in seen:
                    continue

                seen.add(normalized)
                kept += 1

                row = {
                    "qid": qid,
                    "question": ds[qid]["question"],
                    "completion": completion,
                    "gold": gold,
                    "format": fmt,
                }

                f.write(json.dumps(row) + "\n")

            # Write marker so resume knows this qid was processed
            if kept == 0:
                row = {
                    "qid": qid,
                    "question": ds[qid]["question"],
                    "completion": None,
                    "gold": gold,
                    "format": False,
                }

                f.write(json.dumps(row) + "\n")

    done = min(start + GEN_BATCH, len(ds))

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    print(
        f"{done}/{len(ds)} | "
        f"GPU allocated={allocated:.1f}GB | "
        f"reserved={reserved:.1f}GB",
        flush=True,
    )

print("\nGeneration finished.")

del model
torch.cuda.empty_cache()

rows = []

with open(SAVE_JSONL) as f:
    for line in f:
        x = json.loads(line)
        if x["completion"] is not None:
            rows.append(x)

questions_solved = len(set(x["qid"] for x in rows))

print("correct unique trajectories:", len(rows))
print("questions with >=1 correct:", questions_solved, "/", 7473)
print("coverage:", questions_solved / 7473)
print("avg kept / solved question:", len(rows) / questions_solved)

rft_rows = []

for x in rows:
    rft_rows.append({
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "completion": [
            {"role": "assistant", "content": x["completion"]}
        ],
    })

rft_ds = Dataset.from_list(rft_rows)

print(rft_ds)
print(rft_ds[0])

args = SFTConfig(
    output_dir=OUT,
    max_length=640,
    packing=False,

    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,

    num_train_epochs=1,

    bf16=True,
    gradient_checkpointing=True,

    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
    seed=42,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

trainer = SFTTrainer(
    model=MODEL,
    args=args,
    train_dataset=rft_ds,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(f"{OUT}/final")

def evaluate(adapter=None, limit=None, batch_size=256,
             max_new_tokens=400, tag="model"):

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test")

    if limit:
        ds = ds.select(range(limit))

    prompts = [
        tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in ds["question"]
    ]

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

    n_ok = 0
    n_fmt = 0

    for b in range(0, len(order), batch_size):

        idxs = order[b:b + batch_size]

        enc = tok(
            [prompts[i] for i in idxs],
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

        comps = tok.batch_decode(
            out[:, enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        for i, c in zip(idxs, comps):

            pred, fmt = extract_pred(c)

            gold = _clean(
                ds[i]["answer"].split("####")[-1]
            ).strip()

            n_ok += same_number(pred, gold)
            n_fmt += fmt

        print(f"{b + len(idxs)}/{len(order)}")

    n = len(order)

    print(
        f"\n{tag} accuracy {n_ok/n:.4f} "
        f"({n_ok}/{n})   format {n_fmt/n:.4f}"
    )

    return {
        "accuracy": n_ok/n,
        "format_rate": n_fmt/n,
    }


# rft_res = evaluate(
#     adapter=f"{OUT}/final",
#     tag="RFT"
# )

from google.colab import userdata
from huggingface_hub import login

HF_TOKEN = userdata.get("HF_TOKEN")

login(token=HF_TOKEN)

print("Logged into Hugging Face")

!pip install -q huggingface_hub

from huggingface_hub import notebook_login

notebook_login()  # paste your HF write token

MODEL_REPO = "kunjcr2/qwen2.5-0.5b-gsm8k-rft-lora"
DATASET_REPO = "kunjcr2/gsm8k-rft-qwen2.5-0.5b"

# -------------------------------
# Push LoRA adapter
# -------------------------------

trainer.model.push_to_hub(
    MODEL_REPO,
    commit_message="Upload GSM8K rejection-sampling LoRA"
)

trainer.processing_class.push_to_hub(MODEL_REPO)

print("Adapter uploaded:", MODEL_REPO)

# -------------------------------
# Push filtered RFT dataset
# -------------------------------

rft_ds.push_to_hub(
    DATASET_REPO,
    private=False
)

print("Dataset uploaded:", DATASET_REPO)

import matplotlib.pyplot as plt

logs = trainer.state.log_history

steps = [x["step"] for x in logs if "loss" in x]
loss = [x["loss"] for x in logs if "loss" in x]
grad_norm = [x["grad_norm"] for x in logs if "grad_norm" in x]
lr = [x["learning_rate"] for x in logs if "learning_rate" in x]

plt.figure(figsize=(7, 4))
plt.plot(steps, loss)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(steps[:len(grad_norm)], grad_norm)
plt.xlabel("Step")
plt.ylabel("Grad Norm")
plt.title("Gradient Norm")
plt.grid()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(steps[:len(lr)], lr)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate")
plt.grid()
plt.show()