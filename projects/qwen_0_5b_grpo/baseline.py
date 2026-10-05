!pip install -q datasets

import json, re, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
 
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
 
 
def extract_gold(a):
    return _clean(a.split("####")[-1]).strip()
 
 
def same_number(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a.strip() == b.strip()
 
 
def load_model(model_name, dtype):
    """transformers 5.x renamed torch_dtype -> dtype. Handle both."""
    kw = dict(device_map="cuda", attn_implementation="sdpa")
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kw)

def main(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    limit=None,              # None = all 1319
    batch_size=None,         # None = auto from VRAM
    max_new_tokens=400,
    temperature=0.0,
    out="gsm8k_baseline_results.jsonl",
):
    assert torch.cuda.is_available(), "no GPU -- Runtime > Change runtime type"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if batch_size is None:
        batch_size = 256 if vram > 30 else 64
    print(f"gpu: {torch.cuda.get_device_name(0)} ({vram:.0f}GB)  "
          f"dtype: {dtype}  batch: {batch_size}")
 
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
 
    model = load_model(model_name, dtype).eval()
 
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"problems: {len(ds)}")
 
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)
        for q in ds["question"]
    ]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
 
    gk = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    gk.update(dict(do_sample=True, temperature=temperature, top_p=0.95)
              if temperature > 0 else dict(do_sample=False))
 
    n_correct = n_fmt = n_done = 0
    records, t0 = [], time.time()
 
    for b in range(0, len(order), batch_size):
        idxs = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt",
                  padding=True).to("cuda")
        with torch.inference_mode():
            out_ids = model.generate(**enc, **gk)
        comps = tok.batch_decode(out_ids[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
 
        for idx, comp in zip(idxs, comps):
            pred, fmt = extract_pred(comp)
            gold = extract_gold(ds[idx]["answer"])
            ok = same_number(pred, gold)
            n_correct += ok
            n_fmt += fmt
            records.append({"idx": idx, "question": ds[idx]["question"],
                            "completion": comp, "pred": pred, "gold": gold,
                            "correct": ok, "used_format": fmt})
 
        n_done += len(idxs)
        print(f"{n_done}/{len(order)}  acc={n_correct/n_done:.3f}  "
              f"fmt={n_fmt/n_done:.3f}  "
              f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB  "
              f"{time.time()-t0:.0f}s", flush=True)
 
    records.sort(key=lambda r: r["idx"])
    n = len(records)
    print("\n" + "=" * 44)
    print(f"accuracy     {n_correct/n:.4f}  ({n_correct}/{n})")
    print(f"format_rate  {n_fmt/n:.4f}  ({n_fmt}/{n})")
    print(f"wall time    {time.time()-t0:.0f}s")
    print("=" * 44)
 
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return {"accuracy": n_correct / n, "format_rate": n_fmt / n, "records": records}

res = main()

res