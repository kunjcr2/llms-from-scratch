# pip install -q transformers datasets torch accelerate

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 0) Setup & Repro
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1) Load a preference dataset
#    Expect columns: prompt, chosen, rejected
#    (Replace with your own if needed)
# -----------------------------
train_split = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:4000]")
eval_split  = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs[:1000]")

def format_example(prompt, answer):
    return f"### Prompt:\n{prompt}\n\n### Answer:\n{answer}"

class PrefDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.prompts  = split["prompt"]
        self.chosen   = split["chosen"]
        self.rejected = split["rejected"]
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, i):
        return {
            "chosen_text":   format_example(self.prompts[i], self.chosen[i]),
            "rejected_text": format_example(self.prompts[i], self.rejected[i]),
        }

train_ds = PrefDataset(train_split)
eval_ds  = PrefDataset(eval_split)

# -----------------------------
# 2) Tokenizer & collate
# -----------------------------
ENCODER_ID = "microsoft/deberta-v3-small"  # or "distilbert-base-uncased"
MAX_LEN = 512

tok = AutoTokenizer.from_pretrained(ENCODER_ID, use_fast=True)

def collate(batch):
    chosen_texts   = [b["chosen_text"]   for b in batch]
    rejected_texts = [b["rejected_text"] for b in batch]
    enc_ch = tok(chosen_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc_rj = tok(rejected_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return {
        "chosen_input_ids": enc_ch["input_ids"],
        "chosen_attention_mask": enc_ch["attention_mask"],
        "rejected_input_ids": enc_rj["input_ids"],
        "rejected_attention_mask": enc_rj["attention_mask"],
    }

BATCH_SIZE = 8
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
eval_loader  = DataLoader(eval_ds,  batch_size=16,        shuffle=False, collate_fn=collate)

# -----------------------------
# 3) Reward Model: encoder + 1-d head
#    Uses pooler_output if available, else mean-pools last hidden state
# -----------------------------
class RewardModel(nn.Module):
    def __init__(self, base_id):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base_id)
        hidden = self.enc.config.hidden_size
        self.head = nn.Linear(hidden, 1)  # scalar reward
    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        else:
            last = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            h = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        r = self.head(h).squeeze(-1)  # [batch]
        return r

model = RewardModel(ENCODER_ID).to(device)

# (Optional) freeze encoder to save memory / speed up
# for p in model.enc.parameters(): p.requires_grad = False

# -----------------------------
# 4) Optimizer & loss
# -----------------------------
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

def pairwise_bt_loss(s_pos, s_neg):
    # -log(sigmoid(s_pos - s_neg)) == softplus(-(s_pos - s_neg))
    return torch.nn.functional.softplus(-(s_pos - s_neg)).mean()

# -----------------------------
# 5) Train loop (short & sweet)
# -----------------------------
EPOCHS = 1
GRAD_CLIP = 1.0
model.train()

global_step = 0
for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader, 1):
        ch_ids = batch["chosen_input_ids"].to(device)
        ch_msk = batch["chosen_attention_mask"].to(device)
        rj_ids = batch["rejected_input_ids"].to(device)
        rj_msk = batch["rejected_attention_mask"].to(device)

        s_pos = model(ch_ids, ch_msk)  # scores for chosen
        s_neg = model(rj_ids, rj_msk)  # scores for rejected
        loss  = pairwise_bt_loss(s_pos, s_neg)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        global_step += 1
        if global_step % 100 == 0:
            with torch.no_grad():
                acc = (s_pos > s_neg).float().mean().item()
            print(f"step {global_step:05d} | loss {loss.item():.4f} | train_pair_acc {acc:.3f}")

# -----------------------------
# 6) Eval: pairwise accuracy
# -----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in eval_loader:
        ch_ids = batch["chosen_input_ids"].to(device)
        ch_msk = batch["chosen_attention_mask"].to(device)
        rj_ids = batch["rejected_input_ids"].to(device)
        rj_msk = batch["rejected_attention_mask"].to(device)
        s_pos = model(ch_ids, ch_msk)
        s_neg = model(rj_ids, rj_msk)
        correct += (s_pos > s_neg).sum().item()
        total   += s_pos.size(0)
print(f"Eval pairwise accuracy: {correct/total:.3f}")

# -----------------------------
# 7) Save & simple scoring API
# -----------------------------
os.makedirs("rm-checkpoint", exist_ok=True)
torch.save(model.state_dict(), "rm-checkpoint/pytorch_model.bin")

def score_answer(prompt, answer):
    text = format_example(prompt, answer)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        s = model(enc["input_ids"], enc["attention_mask"]).item()
    return s

# Quick demo
demo_prompt = "List two ways to save electricity at home."
good = "Turn off unused lights and use energy-efficient bulbs."
bad  = "Buy more electricity and use it faster."
print("Good score:", score_answer(demo_prompt, good))
print("Bad  score:", score_answer(demo_prompt, bad))