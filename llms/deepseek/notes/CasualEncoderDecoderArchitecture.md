# Causal Encoder–Decoder (CED) in DeepSeek-V4.1-Flash

Primary sources: DeepSeek's [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf), [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), released [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json), and [minimal inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py). The report is authoritative for CED; the code/config are useful for the implementation-specific caveats below.

## The problem CED solves

For a normal 40-layer decoder-only Transformer, a cache miss on a prompt of `N` tokens requires every prompt token to traverse all 40 expensive Transformer blocks. In particular, layer `l` needs the previous layer's token representations in order to make its own K/V cache:

```text
H(l-1)  -- W_K,l / W_V,l -->  K_l, V_l  -- used by layer l attention
```

So the prompt cannot stop after layer 20: layer 21 needs `H20`, layer 22 needs `H21`, and so on through layer 40. This costs roughly `O(NL)` block work for `L` layers (attention details aside).

That is poorly matched to long-horizon agents. A coding or tool-using agent repeatedly appends tool output, files, web pages, or command results to an already long context. Each new, uncached prefix segment triggers **prefill**, often for many input tokens but only a few output tokens before the next tool call. Reducing decode cost alone does not fix that input-heavy workload: prefill can dominate total inference cost.

CED's central bet is that the upper half does not need to run every prompt token through its full attention-and-MoE blocks merely to create the upper-half **global** memory. It can create that memory directly from a single, already-computed causal representation, `H20`.

## Architecture at a glance

DeepSeek-V4.1-Flash's language backbone has 40 causal Transformer layers:

```text
layers  1..20 : causal encoder
layers 21..40 : decoder
```

It is still an autoregressive language model, not an encoder-decoder translation model with a bidirectional source encoder and cross-attention. Both halves are causal Transformer layers. Each layer has global attention plus sliding-window attention (SWA), except layers 1–2, which have SWA only. The feed-forward sublayer in every language layer is DeepSeekMoE. The model has 552B backbone parameters (plus a 196B Engram conditional-memory module) and uses one shared expert plus six routed experts of 384 per MoE layer. [Report, §2.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

The following is the CED dataflow, deliberately omitting ordinary residual/normalization/MoE details:

```text
prompt tokens x_1 ... x_N
           |
           v
  +------------------------+
  | causal encoder, L1–L20 |   runs over all N prompt tokens
  +------------------------+
           |
           v
        H20 [N, d]
           |
           +---- layer-specific global-memory projections ----+
           |                                                   |
           v                                                   v
  C_21 / (global K,V)_21                         C_35 / (global K,V)_35
  cache over prompt positions                     cache over prompt positions

new token representation
           |
           v
  L21 -> L22 -> ... -> L40  -- final hidden state --> norm + LM head --> next-token logits
   |      |            |
   |      |            +-- forms its own query; reads its global memory
   |      +--------------- forms its own query; reads its global memory
   +---------------------- forms its own query; reads its global memory
```

`C_l` above means the global KV representation for layer `l`; it is not a single context vector. It contains one cache row (or, with cache compression, one row per group of tokens) for positions in the causal prefix.

## The key CED substitution

Let `H_l` denote the hidden states *after* layer `l`, with shape `[N, d]`. In an ordinary decoder-only stack, for a layer such as 35:

```text
H34 [N, d] -- W_K,35 --> K35 [N, d_kv]
H34 [N, d] -- W_V,35 --> V35 [N, d_kv]
```

Layer 35 therefore cannot build its prompt cache until layers 21–34 have processed every prompt token.

In CED, global attention changes that dependency. For every decoder layer `l > L/2`, the report defines:

```text
C_l = H_(L/2) W_KV,l
Z_l = H_(L/2) W_Z,l
```

where `C_l` is the layer's global KV entry representation and `Z_l` its compression weights. Here `L = 40`, hence `H_(L/2) = H20`. The projection weights are layer-dependent: a CED design can give layer 21 and layer 35 different global-memory projections even though both start from `H20`.

The requested comparison, precisely stated:

```text
Normal decoder-only Transformer:
H34 -> K35 / V35

DeepSeek CED (global memory rule):
H20 -> global K35 / V35
```

The second line removes the need to calculate prompt-wide `H21 ... H34` before global memory for layer 35 exists. That is the source of the prefill saving. It does **not** say that layer 35's query is made from `H20`; queries remain decoder-layer-local.

### Queries and global K/V memory are different things

For a token currently being decoded at position `t`, decoder layer `l` has a current representation `h_(l-1,t)`. It computes its **query** from that representation:

```text
q_(l,t) = h_(l-1,t) W_Q,l
```

Then it attends with that query to the already-created global memory for its layer:

```text
Attention_l(t) = Attend(q_(l,t), C_l[1:t])
```

Thus CED does not make the decoder a one-shot readout of `H20`. It only changes the *source of the cacheable global K/V side*. The decoder still has distinct layer-21, layer-22, ..., layer-40 computations and distinct queries, which is why it retains upper-layer reasoning capacity during generation.

## Concrete tensor walk-through (simplified)

Use the requested teaching dimensions:

```text
N = 100 prompt tokens
d = 512 hidden dimension
d_kv = 256

H20 = (100, 512)
```

For a simple attention implementation with separate K and V projections, CED can be pictured as:

```text
layer 21 global cache
K21_global = H20 @ W_K,21       = (100, 512) @ (512, 256) = (100, 256)
V21_global = H20 @ W_V,21       = (100, 512) @ (512, 256) = (100, 256)

layer 22 global cache
K22_global = H20 @ W_K,22       = (100, 256)
V22_global = H20 @ W_V,22       = (100, 256)

layer 35 global cache
K35_global = H20 @ W_K,35       = (100, 256)
V35_global = H20 @ W_V,35       = (100, 256)
```

Each row remains position-specific. For example, row 73 of `K35_global` is a learned projection of `H20[73, :]`, and because `H20` came from a causal encoder, that row may depend on positions `1..73`, but not `74..100`.

When generating token 101, the layer-specific current states and queries might be:

```text
h20,101             = (1, 512)       # result of encoder layers for the new position
q21,101 = h20,101 @ W_Q,21 = (1, 256)
q22,101 = h21,101 @ W_Q,22 = (1, 256)
q35,101 = h34,101 @ W_Q,35 = (1, 256)

q21,101 attends to K21_global/V21_global over positions 1..101
q22,101 attends to K22_global/V22_global over positions 1..101
q35,101 attends to K35_global/V35_global over positions 1..101
```

The newly produced `H20[101, :]` is also projected and appended to the relevant global cache before/while the decoder processes that position, so causal self-attention can include the current position without a circular dependency: its global memory is derived from the *pre-decoder* state `H20`, not from the decoder output being calculated.

### Important: these are teaching shapes, not V4.1-Flash's literal cache tensors

The released model's hidden size is **5120**, not 512. Its published config has one KV head and `head_dim = 512`; it uses latent/compressed sparse attention rather than storing independent textbook `K: [N, 256]` and `V: [N, 256]` tensors. In the report's notation, `C_l` is a global **KV entry/latent** and `Z_l` supplies its compression weights. CSA2 can also group input tokens into compressed entries.

In the released V4.1-Flash configuration, the decoder's global CSA2 compression ratio is 1, and the cache-source schedule lists zero-based layer 20 (one-based layer 21) as the decoder global-KV source. Consequently, the actual CSA2 schedule is even more cache-sharing-heavy than the simplified equations: layer 21 materializes a decoder global main-KV latent from `H20`, and later decoder layers generally reuse that main KV while retaining their own queries. Decoder index sources occur at zero-based layers 20, 24, 28, 32, and 36 (one-based 21, 25, 29, 33, and 37), allowing fresh sparse selections without creating new main KV. Therefore, for the actual released schedule, the most concrete trace for layer 35 is:

```text
H20 -- W_KV,21 --> C21 (decoder global main-KV cache) -- CSA2 sharing --> layer 35 reads C21
```

This is fully consistent with the CED rule: layer 35's global memory is ultimately sourced from `H20`, **never `H34`**. The generic CED equation permits a layer-35 projection; CSA2 removes more duplicated global-cache work by sharing it. See [report §2.2–2.3](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) and the [released cache-source config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json).

One repository caveat: the released `inference/model.py` is explicitly a minimal/reference implementation. Its `Transformer.forward` loops through all 40 blocks even when `start_pos == 0`; it demonstrates the cache layout and CSA2 sharing, but not the production prefill scheduler that skips full decoder computation. The report, rather than that simple loop, specifies the CED prefill algorithm and its `O(NL/2 + n_win L/2)` cost.

## Prefill: what runs, what is cached, and what is avoided

Suppose the 100 tokens are a newly seen prompt segment.

1. Embed the 100 tokens and run **all 100 positions through causal encoder layers 1–20**. This yields:

   ```text
   H20 = (100, 512)             # simplified example
   ```

2. Project `H20` to create the decoder's global K/V entries. In the simple example, that creates arrays such as `K21_global`, `V21_global`, `K22_global`, `V22_global`, ..., each with `(100, 256)` shape. The real model produces/global-caches CSA2 main-KV latents and indexer state, with cross-layer sharing and FP4 main-KV storage.

3. **Do not run all 100 prompt positions through decoder layers 21–40** merely to obtain their global cache. That is the skipped work.

4. There is one essential qualification: CED does *not* shortcut SWA. The report says local SWA K/V at every layer `l` is still derived from that layer's own hidden states `H_l`. To seed decoder SWA state, V4.1-Flash uses **Decoder SWA Bounded Replay**: it processes only the final `n_win` prompt tokens through the decoder for local-window state, rather than processing every prompt token. The official config uses `n_win = 128`. For long `N`, this is much smaller than `N`.

Ignoring the smaller projection/replay terms, a normal 40-layer prefill costs about:

```text
O(N * 40)
```

CED costs:

```text
O(N * 20 + n_win * 20) ≈ O(N * 20), when N >> n_win
```

That is why the report describes the saving as *nearly* half, rather than exactly half: CED still needs decoder global-cache projections and decoder SWA replay, and attention/indexing work has its own costs. It is not claiming that upper decoder layers disappear from the model.

## Decode: one new token goes through the full stack

After prefill, consider generating the representation at new position 101. Decode is fundamentally different from prefill because the model must produce the actual upper-layer computation for the one token whose output will determine the next-token distribution.

```text
embedding(x101)
  -> encoder L1 -> ... -> encoder L20
  -> h20,101
  -> decoder L21 -> h21,101
  -> decoder L22 -> h22,101
  -> ...
  -> decoder L35 -> h35,101
  -> ...
  -> decoder L40 -> h40,101
  -> final norm + LM head -> logits over the vocabulary
  -> sample/choose x102
```

For the decoder examples:

* **Layer 21** forms `q21,101` from its current representation (`h20,101`) and reads the global memory derived from `H20`. In the released CSA2 schedule it is the decoder's Full-mode global-KV source, so its `H20`-derived latent is also written to the shared main global cache. It additionally has layer-21 local SWA KV.

* **Layer 22** forms `q22,101` from `h21,101`, not from `H20`. It reads the already available decoder global main-KV cache; under CSA2 it reuses that main cache, while its own layer-local SWA KV is computed from its current state.

* **Layer 35** forms `q35,101` from `h34,101`. It again reads global memory whose provenance is `H20`, not `H34`. In the actual schedule it shares the decoder main cache and may make a new sparse-index selection according to its CSA2 mode; it still computes its own query and local SWA KV.

Finally, `h40,101` is normalized and sent through the LM head to make the logits that predict token 102. CED therefore saves prompt-wide upper-decoder work during prefill, but it does not skip upper-decoder work for a generated token. The full 40-layer path is why the decoder can transform the current token differently at every upper layer.

## Why call it a *causal encoder*?

“Encoder” here names its architectural role—generate reusable memory before the decoder—not its attention mask. It is **causal** because token position `i` in `H20` is computed with a causal mask:

```text
H20[i] = f(x_1, x_2, ..., x_i), not f(x_1, ..., x_N) for i < N.
```

A normal bidirectional encoder, such as BERT's, would let `H20[i]` inspect future input tokens. Projecting that state into a cache for an autoregressive decoder would leak future information during training/inference at intermediate positions. Causal encoder states preserve the same left-to-right factorization as a decoder-only LM and can safely serve as global memory for causal attention.

## Why 8B active parameters in prefill but 16B in decode?

DeepSeek reports **8B active parameters per token during prefill** and **16B during decode**. These are activation/compute figures for this MoE model, not its total parameter count: its backbone is 552B parameters, but only the selected experts and active paths execute for a token.

The asymmetry follows directly from the work above:

* **Prefill:** nearly all input tokens execute the 20-layer causal encoder only. The decoder's global cache is cheaply projected from `H20`; only the bounded SWA replay asks decoder layers to process a short trailing region. On a long prompt, that amortizes to approximately the active computation of half the backbone: 8B per input token.

* **Decode:** each new generated token must execute layers 1–20 *and* 21–40 to form its layer-specific queries, local states, and final `H40` for the LM head. Its global K/V is reused efficiently, but the decoder's per-token computation cannot be bypassed. That activates 16B per output token.

So “8B vs 16B” is not caused by fewer global-cache bytes alone. It is the result of executing roughly half as many expensive MoE Transformer blocks for most prefill tokens.

## Normal decoder-only Transformer vs CED

| Aspect | Normal decoder-only Transformer | DeepSeek CED in V4.1-Flash |
|---|---|---|
| Prompt processing | Every prompt token runs all 40 layers. | Prompt tokens run encoder layers 1–20; decoder full-block work is avoided except bounded SWA replay. |
| Layer-35 global memory | `H34 -> K35/V35`. | CED rule: `H20 -> global K35/V35`; actual CSA2 schedule shares a layer-21 `H20`-derived main cache into layer 35. |
| Layer-35 query during decode | From current `h34,t`. | Also from current `h34,t`. CED does not replace it with an `H20` query. |
| Decoder local SWA KV | From the layer's own hidden states. | Still from the layer's own hidden states; CED does not share/replace these. |
| Long-prompt prefill cost | Approximately `O(N * 40)`. | `O(N * 20 + n_win * 20) ≈ O(N * 20)` for `N >> n_win`. |
| Decode path for one token | All 40 layers. | All 40 layers; global memory is prebuilt/reused. |
| Active parameters per token (reported) | No corresponding V4.1 CED figure. | 8B prefill; 16B decode. |

## Five key takeaways

1. CED targets the cache-miss **prefill** cost of long-input, short-output agent workflows.
2. DeepSeek-V4.1-Flash divides its 40 causal layers into a 20-layer causal encoder and a 20-layer decoder.
3. The decisive dependency change is `H34 -> K35/V35` becoming `H20 -> global K35/V35` for decoder global memory.
4. Decoder queries are still generated from each decoder layer's running hidden state; CED changes global K/V provenance, not the whole decoder computation.
5. CED and CSA2 compose: CED makes decoder global memory originate at `H20`; CSA2 further shares that memory and sparse-index work across layers. SWA remains layer-local and needs bounded replay.

## Common misconceptions

**“CED is a conventional encoder–decoder with bidirectional encoding and cross-attention.”** No. Its encoder is causal, and the decoder's global attention reads cache entries constructed from causal encoder states. It preserves autoregressive semantics.

**“`H20` is one compressed context vector.”** No. `H20` has one contextual vector per causal position. The projection maps its rows into per-position (or compressed-group) global cache entries; no vague single-vector context compression is required.

**“The decoder does not run during prefill.”** Not literally. It avoids full decoder computation for the entire prompt, but still projects global cache entries and replays the prompt tail to initialize layer-local SWA KV.

**“Every decoder layer's entire attention cache comes from `H20`.”** Only the **global** cache follows the CED rule. SWA/local KV is conventional and layer-local. CSA2 adds separate reuse rules for global main-KV, indexer K, and Top-K indices.

**“Layer 35 no longer depends on layer 34.”** It no longer needs `H34` to create its *global K/V cache* during prefill. During decode, its query and current state absolutely depend on the output of layer 34.

**“8B prefill means the model has only 8B parameters.”** No. It is the reported active-parameter count per prefill token, enabled by avoiding upper-decoder execution for most prompt tokens in a much larger MoE model.
