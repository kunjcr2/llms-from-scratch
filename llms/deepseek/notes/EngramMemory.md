# Engram conditional memory in DeepSeek-V4.1-Flash

## Sources and scope

The primary sources for these notes are:

- [DeepSeek-V4.1-Flash model configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/config.json)
- [Official `inference/engram.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py)
- [Official inference `model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
- [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348), especially its descriptions of DeepSeekMoE and the residual stream
- [Official Engram paper/repository](https://github.com/deepseek-ai/Engram)

The V4 report's published architecture section focuses on hybrid attention, mHC, and MoE. The concrete V4.1-Flash Engram numbers and runtime behavior come from the released model configuration and inference implementation. That distinction is useful: do not infer V4.1's Engram behavior from the older Engram demo, because the demo has a different configuration and mocks surrounding layers.

## 1. What problem does Engram solve?

An ordinary Transformer is very good at transforming representations, but it has no direct primitive saying: "given this exact local token pattern, fetch the corresponding learned record." It can reconstruct such patterns through attention and feed-forward computation, but reconstruction costs compute and competes with other uses of the network's capacity.

Engram adds a separate path for relatively static, local, identity-like information:

- common word and subword sequences;
- spelling and capitalization/normalization variants;
- names, entities, phrases, and local syntactic patterns;
- associations that are useful whenever the same short token sequence appears.

The intended information is not a live document database and not a conversational memory. It is learned parametric memory indexed by token n-grams. The entries are trained vectors, not text snippets. An entry can therefore encode a useful feature or local pattern rather than literally storing a sentence.

Why the existing components are not enough:

- **Attention** retrieves information from the current sequence by computing content-dependent interactions. It is dynamic and context-sensitive, but repeatedly reconstructing a familiar local pattern through attention is expensive, especially when the useful dependency is short and nearly deterministic.
- **Transformer weights** are dense transformations. They contain knowledge, but a matrix multiplication applies the same weight matrix broadly to every token; there is no cheap direct address for one particular n-gram.
- **MoE experts** provide conditional computation: a router chooses a few neural functions, and those functions perform substantial matrix multiplications. That is excellent for allocating computation, but it is not a static lookup of a particular local pattern.

Engram supplies the missing operation: use token identity and local history to address a small number of learned rows, then let the current hidden state decide how much of the retrieved value to use.

## 2. The simplest correct mental model

For one token position:

```text
token IDs and recent history
	|
	v
normalize/map token IDs to compressed IDs
	|
	v
construct 2-, 3-, and 4-grams ending at this position
	|
	v
for each n-gram and each of 8 heads: multiply IDs, XOR, modulo a prime
	|
	v
add the bucket offsets and fetch 24 learned rows
	|
	v
concatenate rows -> one projection -> 4 keys + 1 shared value
	|
	v
compare each key with its hidden-state copy to make a gate
	|
	v
add gated value to the four-copy residual stream
	|
	v
the Transformer block consumes the updated stream
```

The important correction to an overly simple picture is that Engram does not merely average retrieved embeddings and inject them. It retrieves 24 rows, projects them into a value and four keys, computes one gate per hyper-connection copy, and performs a residual addition.

## 3. Actual V4.1-Flash configuration

The released configuration contains:

| Setting | V4.1-Flash value | Meaning |
| --- | ---: | --- |
| Engram layers | `1`, `14` | zero-based Transformer layer IDs |
| Maximum n-gram size | `4` | lookups are 2-, 3-, and 4-grams |
| Hash heads | `8` | eight independent bucket addresses per n-gram order |
| Bucket-start vocabulary size | `16,000,000` | each n-gram/head range starts searching above this size for a unique prime modulus |
| Table rows, layer 1 | `384,006,168` | unpadded rows in the layer-1 table |
| Table rows, layer 14 | `384,016,682` | unpadded rows in the layer-14 table |
| Row dimension | `256` | `engram_head_dim`; rows are stored in FP8 in the released inference code and dequantized on lookup |
| Compressed tokenizer vocabulary | `99,092` | the token-ID space used by the hash multipliers |
| Hyper-connection copies | `4` | four residual-stream copies receive separately gated values |

There are $3 \times 8 = 24$ retrieved rows per token at an Engram layer. The two tables contain

$$384{,}006{,}168 + 384{,}016{,}682 = 768{,}022{,}850$$

rows. At 256 values per row, that is

$$768{,}022{,}850 \times 256 = 196{,}613{,}849{,}600$$

embedding values, or approximately **196.6B Engram parameters** before counting storage metadata/scales. This is the source of the often-rounded "196B Engram parameters" figure. It is not 196B dense Transformer parameters activated by every token.

The two layer tables have slightly different row counts because all n-gram/head bucket ranges across the layout receive distinct prime-sized regions. The small difference is a consequence of the prime allocation, not a semantic difference between layer 1 and layer 14.

## 4. One concrete example: `The capital of France is Paris`

This example uses **toy token IDs**, not DeepSeek tokenizer IDs. Pretend tokenization produced:

```text
position:   0     1       2    3       4   5
token:    The capital  of France  is Paris
toy ID:    101    102  103    104  105   106
```

At position 5, the n-grams ending at `Paris` are:

```text
2-gram: [is, Paris]             -> [105, 106]
3-gram: [France, is, Paris]     -> [104, 105, 106]
4-gram: [of, France, is, Paris] -> [103, 104, 105, 106]
```

In the real implementation, each input ID first passes through `token_map`. The map is made by decoding tokens and applying NFKC, accent stripping, lowercasing, whitespace normalization, and special handling for byte fragments. Thus several surface forms can intentionally share a compressed ID. For this toy walkthrough, assume the compressed IDs remain 103, 104, 105, and 106.

For hash head $j$ at layer $\ell$, the implementation has an odd learned/fixed multiplier $m_{\ell,k}$ for lookback slot $k$. A simplified description for the 3-gram is:

$$
h_{\ell,3,j} = (m_{\ell,0} 104) \oplus (m_{\ell,1} 105) \oplus (m_{\ell,2} 106),
$$

where $\oplus$ is bitwise XOR. The bucket is:

$$
b_{\ell,3,j} = h_{\ell,3,j} \bmod p_{\ell,3,j},
$$

where $p_{\ell,3,j}$ is that n-gram/head's unique prime modulus. The actual table index is the bucket plus an offset reserved for that particular n-gram order and head:

$$i_{\ell,3,j} = b_{\ell,3,j} + o_{\ell,3,j}.$$

The same operation produces indices for the 2-gram and 4-gram, giving 24 indices total. Each index selects a 256-dimensional row from the Engram table for the current layer:

```text
24 indices
    |
    +--> row for 2-gram, head 0 ... row for 2-gram, head 7
    +--> row for 3-gram, head 0 ... row for 3-gram, head 7
    +--> row for 4-gram, head 0 ... row for 4-gram, head 7
    |
    v
[e_0, e_1, ..., e_23], each e_i has shape [256]
```

The actual row values are learned vectors. The hash values do not themselves carry a semantic embedding; they only choose addresses.

## 5. Hashing, translated from `engram.py`

### 5.1 Why token IDs?

Token IDs are stable discrete symbols available during both training and inference. They make the address deterministic and cheap: the same local token-ID pattern produces the same candidate addresses. Hashing compressed IDs rather than raw IDs also makes normalized variants share addresses when the tokenizer normalization says they are equivalent.

### 5.2 The rolling XOR construction

The runtime caches compressed token IDs so that prefill and one-token-at-a-time decoding use the same history. For a current position it gathers the current token and up to three previous tokens. At the beginning of a sequence, or after an invalid/dead image token, it substitutes the compressed pad ID; an n-gram never crosses a dead span.

Let $c_t$ be the compressed ID at position $t$. For each layer $\ell$, the implementation creates bounded odd multipliers $m_{\ell,0}, \ldots, m_{\ell,3}$. The multipliers are generated deterministically from a per-layer random generator seeded by `10007 * layer_id`; they are bounded to avoid signed 64-bit overflow.

The running hash is:

$$
r_1(t) = m_{\ell,0}c_t \oplus m_{\ell,1}c_{t-1},
$$

$$
r_2(t) = r_1(t) \oplus m_{\ell,2}c_{t-2},
$$

$$
r_3(t) = r_2(t) \oplus m_{\ell,3}c_{t-3}.
$$

These correspond to 2-, 3-, and 4-gram hashes. For every order $n$ and head $j$:

```text
rolling = multiplied current token
for each additional lookback token:
    rolling = rolling XOR (multiplier[lookback] * token_id)
bucket = rolling modulo prime[n-gram order, head]
index = bucket + disjoint_region_offset[n-gram order, head]
```

The code computes all heads in parallel. The resulting tensor has shape `[batch, sequence, 2 Engram layers, 24 hash columns]` before the layer-specific slice is passed into that layer's Engram module.

### 5.3 Why different bucket regions?

If 2-, 3-, and 4-grams all used the same integer range, an index from one order could overwrite or be confused with an index from another. `EngramLayout` allocates a distinct prime-sized region for every `(n-gram order, head)` pair. The offsets concatenate these regions into one flat embedding table. This lets the implementation use one embedding object per layer while preserving the identity of each order/head.

### 5.4 Why eight hash heads?

One hash is vulnerable to collisions. Eight independently addressed rows give the module multiple views of the same n-gram. The following projection can combine those views, and a collision in one head need not be repeated in the others. This is not eight attention heads: there are no sequence-wide attention scores here, only eight independent deterministic addresses per n-gram order.

### 5.5 What does a collision mean?

A collision means two different n-grams map to the same bucket in one order/head region and therefore share that row. The row receives gradients from both patterns, so it becomes a compromise feature. Collisions are intentional and managed by large regions, multiple heads, and the learned downstream projection; they are not an assertion that the hash uniquely identifies every n-gram.

## 6. What happens after lookup?

This is the part that turns a collection of rows into a conditional residual update.

### 6.1 Shapes in V4.1-Flash

For the real model, let `D = 5120`, `H = 4` hyper-connection copies, `R = 256` row dimension, and `C = 24` hash columns.

```text
hidden stream x       [B, L, H, D]       = [B, L, 4, 5120]
looked-up rows        [B, L, C, R]       = [B, L, 24, 256]
flattened rows        [B, L, C*R]        = [B, L, 6144]
```

The `ParallelEngramEmbedding` lookup returns the 24 rows. In the released inference code, its FP8 rows are dequantized to BF16 on lookup. A single learned `wkv` projection maps the concatenated rows to:

```text
kv = wkv(rows)                     [B, L, H*D + D] = [B, L, 25600]
key, value = split(kv)
key                               [B, L, H, D]     = [B, L, 4, 5120]
value                             [B, L, D]        = [B, L, 5120]
```

There is one shared value for all four residual copies, but one key per copy.

### 6.2 Query-key gate

The current hidden state is the query. For copy $g$ at token $t$, the implementation computes a normalized, dimension-scaled dot product:

$$
d_{t,g} = \frac{\langle x_{t,g} \odot q_g \odot k_{t,g},\;\mathbf{1}\rangle}{\sqrt{D}}
\;\operatorname{RMSNormFactor}(x_{t,g})
\;\operatorname{RMSNormFactor}(k_{t,g}),
$$

where `q_weight` and `k_weight` are learned per-copy/per-dimension factors, multiplied together in the code. Equivalently, the implementation computes a dot product after normalizing the hidden state and key along their last dimension, with the learned elementwise weighting included.

It then applies a signed square root followed by a sigmoid:

$$
g_{t,g} = \sigma\left(\operatorname{sign}(d_{t,g})\sqrt{\max(|d_{t,g}|,10^{-6})}\right).
$$

The signed square root compresses the magnitude while preserving whether the match is positive or negative. The sigmoid converts it into a gate in $(0,1)$.

For a toy hidden dimension `D = 4`, one position and one copy look like:

```text
x[g]       = [x0, x1, x2, x3]          current residual copy
key[g]     = [k0, k1, k2, k3]          projected memory key
value      = [v0, v1, v2, v3]          projected shared memory value

d = normalized_sum(x[g] * learned_weight * key[g]) / sqrt(4)
gate = sigmoid(signed_sqrt(max(abs(d), 1e-6)))
out[g] = x[g] + gate * value
```

There is no softmax across the 24 rows after retrieval and no competition among n-gram orders. Their information is mixed by `wkv`; the hidden state controls admission of the resulting value.

### 6.3 Where it enters the Transformer

At layer 1 or 14, `Block.forward` first applies Engram to the four-copy residual stream:

$$
X' = X + G(X,\operatorname{lookup}(\text{n-grams})).
$$

Then the normal block processing continues. The block's hyper-connection machinery collapses/mixes the four copies into the sublayer input, normalizes it, and sends it through attention; later it sends the stream through the MoE FFN and residual mixing. Therefore the next Transformer operations see a hidden representation that already contains a token-conditioned memory contribution. Engram is not a side output read only by the logits head, and it does not replace attention or the MoE.

In the V4.1 implementation, Engram is called before the block's attention/FFN processing, while the block's mHC logic controls how the updated four-copy stream is consumed and propagated. This is why the correct residual picture is `updated stream -> ordinary Transformer block`, not `lookup -> expert router`.

## 7. Why it is called conditional memory

Consider one parameter matrix in a dense layer. For a token, the matrix participates in a matrix multiplication regardless of which local phrase the token belongs to. Its parameter capacity is global, but its access pattern is dense.

An Engram row is different:

```text
particular local token history
	|
	v
deterministic hash addresses
	|
	v
only those selected rows are read
```

Most of the approximately 768M rows across the two tables are not touched for a particular token. The token still pays for hashing, 24 row reads, and the relatively small projection/gating path, but it does not multiply through the entire 196B-row table. The parameter count measures available addressable memory; it does not equal per-token dense compute.

The qualification matters: conditional access does not make the table free. It creates a memory-bandwidth and placement problem rather than a dense-GEMM problem. That tradeoff is exactly why the inference code shards the table by rows and has a specialized embedding lookup path.

## 8. Engram versus MoE

**Engram is not MoE expert routing.** They are two different conditional mechanisms in the same model.

| Property | MoE | Engram |
| --- | --- | --- |
| What is selected? | A few neural experts | A few rows in a hash table |
| Addressing signal | Router scores from the hidden state | Token IDs and local n-gram identity, hashed deterministically |
| Main operation | Expert matrix multiplications and nonlinearities | Embedding reads, one projection, key-query gates |
| Conditional resource | Computation | Static learned memory access |
| Output | Expert-transformed hidden state | Gated memory value added to the residual stream |
| Can two identical n-grams address the same rows? | Not necessarily; routing depends on hidden state | Yes, deterministically, subject to token map and layer/head hashes |

MoE asks: "Which functions should compute on this representation?" Engram asks: "Which learned local-pattern records should be fetched for this token history?" A token can use Engram and still route through the normal six selected V4.1 routed experts plus the shared expert.

## 9. How training makes the table useful

During training, the forward path is the same in principle:

1. Convert token IDs to compressed IDs and construct the causal n-grams.
2. Hash them to integer addresses.
3. Gather the corresponding rows.
4. Project rows into keys and a value.
5. Gate the value against the current hidden state and add it to the residual stream.
6. Compute the language-model loss downstream.

The address calculation is discrete and has no useful gradient. The selected row values do have gradients. If a loss gradient reaches a retrieved row, that row is updated by the optimizer. The projection, key parameters, and gate parameters also receive gradients.

Suppose a recurring n-gram frequently appears in contexts where a particular feature helps predict the next token. Its selected row is repeatedly used, so updates push that row toward a vector that supplies the feature. The key/gate path learns when the current hidden state should accept it. A collision causes multiple n-grams to share updates; the model learns a shared compromise feature, while the multiple heads and large bucket regions reduce the damage.

This is ordinary gradient-based learning around a discrete lookup: the address is selected without differentiation, but the values fetched at that address are differentiable parameters.

## 10. Practical inference implications

Engram behaves more like a deterministic random-access memory read than like a huge matrix multiplication:

- **Arithmetic:** compute a small fixed number of integer hashes, then do 24 indexed row reads and a modest projection/gating calculation.
- **Memory traffic:** the large table is mostly storage. Only selected rows need to be fetched, though random access can make locality and bandwidth important.
- **Sharding:** `ParallelEngramEmbedding` partitions rows across distributed ranks. Each rank looks up its local portion, masks nonlocal indices, and combines results when needed.
- **Storage format:** the released inference implementation stores rows in FP8 with per-block scales and dequantizes the selected rows to BF16. This reduces storage and transfer cost relative to BF16 rows.
- **Host-memory potential:** deterministic addresses make prefetching/offloading plausible. The table need not be treated like a dense GEMM weight matrix that must be resident in the same high-throughput compute path for every token.
- **Decode state:** `NgramHashState` caches token history across prefill and decode. A single newly generated token can therefore form its 2-, 3-, and 4-grams without recomputing the whole sequence's hash history.

The practical advantage is not "196B parameters with no cost." It is "196B addressable capacity with access cost proportional to a constant number of rows per token, rather than proportional to the full table."

## 11. Complete ASCII flow diagram

```text
input_ids [B,L]
    |
    +--> compressed token map
    |       NFKC / accent / case / whitespace normalization
    |
    +--> causal token-history cache
	    |
	    +--> 2-gram ending at t
	    +--> 3-gram ending at t
	    +--> 4-gram ending at t
		    |
		    v
	  per Engram layer l and hash head j
	  odd_multiplier[k] * compressed_id[t-k]
		    |
		    v
	      rolling bitwise XOR
		    |
		    v
	      modulo unique prime
		    |
		    v
	   add disjoint region offset
		    |
		    v
    24 hash indices [B,L,24] for this layer
		    |
		    v
    ParallelEngramEmbedding: 24 x 256-d rows
		    |
		    v
       flatten [B,L,6144] -> wkv projection
		    |
		    +--> 4 keys [B,L,4,5120]
		    +--> 1 shared value [B,L,5120]
				  |
 hidden residual X [B,L,4,5120]  |
	  |                       |
	  +--> normalized query-key dot per copy
				  |
				  v
		 signed sqrt -> sigmoid gate [B,L,4]
				  |
				  v
       X' = X + gate[...,None] * value[:, :, None, :]
		    |
		    v
       mHC / attention / MoE processing in the Transformer block
		    |
		    v
		 next layer
```

## 12. Engram vs attention vs MoE

| Mechanism | Reads from | Address/selection | What it contributes | Typical cost shape |
| --- | --- | --- | --- | --- |
| Attention | Other hidden states/tokens in the context | Learned content-dependent scores | Dynamic context-dependent information | Projections plus interactions with selected context entries |
| MoE | Learned expert networks | Hidden-state router chooses top experts | Conditional computation and nonlinear transformation | A few expert FFN matrix multiplications per token |
| Engram | Learned embedding rows | Token-ID n-gram hashes choose fixed addresses | Conditional local-pattern memory value | Constant number of indexed rows plus one projection/gate |

## 13. Five key takeaways

1. Engram is a learned n-gram-addressed memory path, not another attention mechanism.
2. V4.1-Flash inserts it at layers 1 and 14, using 2-, 3-, and 4-grams with 8 heads each.
3. Each token retrieves 24 rows, concatenates them, and projects them into four keys plus one shared value.
4. A normalized query-key gate decides independently for each hyper-connection copy how much value enters the residual stream.
5. The approximately 196.6B parameters are mostly inactive for any one token; the model pays for selected random-access rows, not a dense 196B-parameter computation.

## Common misconceptions

- **"Engram is a database of text."** No. Its rows are learned vectors. They may encode useful local facts or patterns, but lookup returns vectors, not strings.
- **"Engram replaces attention."** No. It supplies a fast local-memory path; attention still handles dynamic context interactions.
- **"Engram is an MoE router."** No. Hashes select memory rows; a hidden-state router selects computational experts.
- **"The hash uniquely identifies every phrase."** No. Finite bucket regions create collisions. Multiple heads and large regions make collisions manageable, not impossible.
- **"All 196B Engram values are computed for every token."** No. A token reads 24 rows at an Engram layer.
- **"The n-gram rows are simply averaged."** No. They are concatenated, projected into keys/value, and gated against the current hidden state.
- **"The model hashes raw text strings at runtime."** No. It hashes compressed token IDs derived from tokenizer output and a deterministic normalization map.
- **"The V4 paper's MoE description alone specifies Engram."** No. The released V4.1-Flash config and `inference/engram.py` are the authoritative sources for these Engram-specific details.

## If the Transformer already has attention and MoE experts, why the hell does it need Engram?

Because attention and MoE are primarily computational mechanisms: attention dynamically mixes context representations, and MoE dynamically chooses neural functions to transform a representation. Neither gives the model a cheap, direct, reusable address for a familiar local token pattern. Engram adds that third capability: deterministic conditional memory. It can hand early layers a learned feature for a recurring phrase immediately, allowing attention to spend more capacity on broader context and allowing MoE experts to spend more computation on transformations and reasoning instead of repeatedly reconstructing static local patterns.
