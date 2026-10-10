# Lecture 2 — Loading a Base Model, Text Generation, and KV Caching

**Video:** [Build A Reasoning Model From Scratch 2: Loading a Base Model, Text Generation, and KV Caching](https://www.youtube.com/watch?v=BJua0yjO5dk) (1:36:41)

## 1. Purpose of the base-model chapter

Before adding reasoning techniques, establish the generation machinery they will reuse:

```text
text → tokenizer → token IDs → pretrained LLM → next-token logits
     → token selection → append token → repeat → decoded text
```

This chapter deliberately uses readable PyTorch code. High-performance inference systems add batching, fused kernels, quantization, and sophisticated scheduling, but those optimizations can hide the central algorithm.

## 2. Why Qwen3-0.6B?

The course uses the **base**, not instruction/reasoning, variant as the starting policy.

- At 0.6B parameters it is small enough for consumer hardware.
- It has capable open weights and a corresponding official reasoning variant for comparison.
- It is smaller than common 1B alternatives.
- The companion package provides a pure-PyTorch Qwen3 reimplementation compatible with the official weights.

The architecture itself is not modified in these lectures. The reasoning methods sit on top of it, so understanding every Qwen3 layer is optional.

## 3. Tokens and tokenization

An LLM consumes integers, not strings. A tokenizer maps text fragments to IDs from a fixed vocabulary and reverses that mapping for display.

```python
token_ids = tokenizer.encode("Reasoning models use more compute.")
text = tokenizer.decode(token_ids)
```

The Qwen3 tokenizer has roughly 152k vocabulary entries. Token boundaries need not match words: punctuation, whitespace, common substrings, and special markers can each be tokens.

Two shapes matter:

- A prompt commonly begins as `[sequence_length]`.
- The model expects a batch dimension: `[batch_size, sequence_length]`, so a single prompt becomes `[1, sequence_length]`.

## 4. Loading weights and selecting a device

Conceptually, model loading is:

```python
model = Qwen3Model(config)
state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
```

Use `model.eval()` for inference so training-only behavior is disabled. Use `torch.inference_mode()` for ordinary generation to avoid building autograd graphs and reduce overhead.

A device-selection policy can prefer CUDA, then supported alternatives, then CPU. For a first pass, CPU can make results easier to compare across systems.

## 5. Autoregressive generation

For a vocabulary of size `V`, the model returns one logit per vocabulary item at every sequence position:

```text
logits shape = [batch, sequence_length, vocabulary_size]
```

Only the final position predicts the next token:

```python
next_token_logits = model(input_ids)[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
```

Greedy decoding selects the highest-logit token. The new token is appended, the longer sequence is fed back to the model, and the loop repeats.

```python
@torch.inference_mode()
def generate_greedy(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()

    for _ in range(max_new_tokens):
        logits = model(token_ids)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield next_token
        token_ids = torch.cat((token_ids, next_token), dim=1)
```

The generator yields tokens so the caller can decode and display output as it arrives. Streaming changes the user experience, not the underlying next-token process.

### Stop at EOS

The base model may continue into unrelated text after an `<|endoftext|>` marker because that marker separated documents during pretraining. Stop when `tokenizer.eos_token_id` is emitted; a maximum-token limit remains a safety boundary.

## 6. Why naive decoding is wasteful

At generation step `t`, naive decoding feeds the entire prompt plus all previously generated tokens through every Transformer layer again. Most attention key/value projections for old tokens are therefore recomputed repeatedly.

For a growing sequence, that duplicated work becomes increasingly expensive.

## 7. KV caching

Self-attention computes queries, keys, and values. During autoregressive generation:

- Old tokens' **keys and values** do not change.
- The newest token needs to attend to those old representations.
- Cache each layer's old K/V tensors and compute only the new token's projections.

```text
first pass:  process the entire prompt → fill cache → predict first new token
later pass:  process one new token + cached K/V → extend cache → predict next token
```

A simplified cached loop is:

```python
@torch.inference_mode()
def generate_with_cache(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    logits = model(token_ids, cache=cache)[:, -1]
    for step in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield next_token
        if step + 1 < max_new_tokens:
            logits = model(next_token, cache=cache)[:, -1]
```

The first pass still processes the prompt. The speedup appears on subsequent tokens. KV caching trades additional memory for lower repeated computation.

## 8. Measuring performance

Benchmark generation rather than only model construction:

```python
elapsed = time.perf_counter() - start
tokens_per_second = generated_token_count / elapsed
```

Important controls:

- Use the same prompt and number of generated tokens.
- Exclude or explicitly report compile warm-up time.
- Synchronize GPU work before timing when necessary.
- Record device, dtype, PyTorch version, cache mode, compilation mode, and batch size.
- Compare output validity as well as speed.

## 9. `torch.compile`

`torch.compile(model)` lets PyTorch capture and optimize the model's operations. Compilation has an up-front cost, so repeat generation before judging steady-state throughput.

Compilation and KV caching are independent optimizations and can be combined. The lecture's Mac Mini M4 CPU example increased from roughly 5 tokens/s without caching to about 29 tokens/s with caching, and about 68 tokens/s with caching plus compilation. Results were hardware-dependent; on some GPUs, compilation alone performed best for batch size 1.

### Windows caveats

PyTorch Inductor needs a functioning C/C++ toolchain; CUDA compilation may also need a compatible Triton installation. If setup is difficult, skip compilation—the educational code does not depend on it. Some Windows reports found `mode="max-autotune"` more beneficial than the default.

## 10. What this foundation enables

Later lectures modify only particular parts of this loop:

- Temperature and top-p replace greedy `argmax` with sampling.
- Self-consistency runs the loop several times.
- Log-probability scoring reuses the model's next-token distributions.
- GRPO samples rollouts, scores them, and updates weights using their sequence log-probabilities.

## Takeaways

- Generation is repeated next-token prediction; logits are unnormalized token scores.
- Greedy decoding is deterministic but cannot provide diverse candidate solutions.
- Stop on EOS and cap output length.
- KV caching avoids recomputing old attention keys/values and is central to efficient decoding.
- `torch.compile` can help, but its benefit is device-, mode-, and workload-dependent.
- With the base generation system established, the next requirement is a trustworthy verifier.

