#######################################################
# THIS CODE HAS MULTIPLE CRITICAL ISSUES. 
# CHECK BEFORE MOVING AHEAD
#######################################################

# pip install -U "transformers>=4.44" "trl>=0.12" datasets accelerate peft bitsandbytes

# Apparently this is much better than PPO since we dont have to train a reward model for this !
    # we can dictly optimize the policy using chosen and rejected.

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig 
from peft import LoraConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAXLEN = 1024

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# ---------- 1) SFT ----------
chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:8000]")
def sft_text(p, a):
    """
    Proper chat template is being applied
    """
    msgs = [{"role":"user","content":p},{"role":"assistant","content":a}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
chat = chat.map(lambda ex: {"text": sft_text(ex["prompt"], ex["response"])}, # text key is important
                remove_columns=chat.column_names)

# We get the policy model - lora config - sft config - sfttrainer - TRAIN SFT MODEL !
policy = AutoModelForCausalLM.from_pretrained(BASE_ID, device_map="auto", load_in_4bit=True)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
sft_cfg = SFTConfig(output_dir="ckpt_sft_qwen05b_dpo_refrozen",
                    max_seq_length=MAXLEN, per_device_train_batch_size=1,
                    gradient_accumulation_steps=16, num_train_epochs=1,
                    learning_rate=5e-5, logging_steps=50, report_to=[])
sft_tr = SFTTrainer(model=policy, tokenizer=tok, train_dataset=chat,
                    args=sft_cfg, peft_config=peft_cfg, formatting_func=lambda b: b["text"])
sft_tr.train()
sft_tr.model.save_pretrained("ckpt_sft_qwen05b_dpo_refrozen"); tok.save_pretrained("ckpt_sft_qwen05b_dpo_refrozen")

# ---------- 2) DPO data (prompt serialized with add_generation_prompt=True) ----------
prefs = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:12000]")
def prep_dpo(ex):
    """
    Applying prompt templateeeeeeeeeeee 
    We take the formatted prompt first and then we add chosen answer to it to make formatted chosen
    Same thing with the formatted Rejected part !

    And then at the last we add those to collumns of ex and we move ahead !
    """
    formatted_prompt = tok.apply_chat_template(
        [{"role":"user", "content": ex["prompt"]}],
        tokenize=False, add_generation_prompt=True
    )

    # For chosen: include prompt + assistant response
    # Extract assistant part from chosen message list
    # Let's assume chosen[-1] is the assistant's final response
    chosen_assistant = None
    for msg in reversed(ex["chosen"]):
        if msg["role"] == "assistant":
            chosen_assistant = msg["content"]
            break
    assert chosen_assistant is not None, "no assistant message in chosen"

    formatted_chosen = tok.apply_chat_template(
        [
          {"role":"user", "content": ex["prompt"]},
          {"role":"assistant", "content": chosen_assistant}
        ],
        tokenize=False
    )

    # Similarly for rejected
    rejected_assistant = None
    for msg in reversed(ex["rejected"]):
        if msg["role"] == "assistant":
            rejected_assistant = msg["content"]
            break
    assert rejected_assistant is not None, "no assistant message in rejected"

    formatted_rejected = tok.apply_chat_template(
        [
          {"role":"user", "content": ex["prompt"]},
          {"role":"assistant", "content": rejected_assistant}
        ],
        tokenize=False
    )

    ex["prompt"] = formatted_prompt
    ex["chosen"] = formatted_chosen
    ex["rejected"] = formatted_rejected
    return ex
prefs = prefs.map(prep_dpo)

# ---------- 3) Policy + FROZEN reference ----------
# Policy model which will generate the responses
policy_dpo = AutoModelForCausalLM.from_pretrained(
    "ckpt_sft_qwen05b_dpo_refrozen", device_map="auto", load_in_4bit=True
)
# Reference model which will be used to compare policy responses !
ref_dpo = AutoModelForCausalLM.from_pretrained(
    "ckpt_sft_qwen05b_dpo_refrozen", device_map="auto", load_in_4bit=True
)
# EXPLICIT FREEZE (DPOTrainer won't train it anyway, but we make it crystal clear)
ref_dpo.eval()
for p in ref_dpo.parameters():
    p.requires_grad = False

dpo_cfg = DPOConfig(output_dir="ckpt_dpo_qwen05b_refrozen",
                    beta=0.1, per_device_train_batch_size=1,
                    gradient_accumulation_steps=16, num_train_epochs=1,
                    max_seq_length=MAXLEN, learning_rate=5e-6,
                    logging_steps=50, report_to=[])

dpo_tr = DPOTrainer(model=policy_dpo, ref_model=ref_dpo,
                    args=dpo_cfg, train_dataset=prefs, tokenizer=tok, peft_config=peft_cfg)
dpo_tr.train() # Woohoooo !
dpo_tr.model.save_pretrained("ckpt_dpo_qwen05b_refrozen"); tok.save_pretrained("ckpt_dpo_qwen05b_refrozen")
print("Saved DPO policy → ckpt_dpo_qwen05b_refrozen")
