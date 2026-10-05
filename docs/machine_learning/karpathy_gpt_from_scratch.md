# 🧠 Let's Build GPT from Scratch — Technical Summary

> Based on Andrej Karpathy’s livestream: "Let's build GPT: from scratch, in code, spelled out"

---

## 1. Tokenization

- **Character-level tokenizer**
  - Vocabulary built from all unique characters in the dataset.
  - `stoi` and `itos` mappings used to convert between text ↔ integer IDs.
- **Encoding**: Text → Integer tensor
- **Decoding**: Integer tensor → Text

---

## 2. Dataset Preparation

- Input sequence length: `block_size` (context window, e.g., 8–64 tokens)
- Input-target pairs:
  - `x = input[0:n-1]`
  - `y = input[1:n]`
  - Predict next character given previous ones

---

## 3. Model Architecture (MiniGPT)

### 🔹 Embedding Layer

- Learnable `token_embedding_table`: `[vocab_size, n_embed]`
- Maps token IDs to dense vectors

### 🔹 Positional Encoding

- Learnable `position_embedding_table`: `[block_size, n_embed]`
- Added to token embeddings to provide order info

---

## 4. Self-Attention Mechanism

### 🔹 Head (Single Self-Attention)

- Linear projections for `key`, `query`, `value`:  
  `W_k`, `W_q`, `W_v` → shape: `[n_embed, head_size]`
- Attention scores:

```

att = (Q @ K.T) / sqrt(head\_size)

```

- **Causal mask**: Lower-triangular mask to block future positions
- Softmax over attention scores
- Output: `weights @ V` → contextual embedding per token

---

## 5. Multi-Head Attention

- Multiple attention heads in parallel
- Outputs from all heads are concatenated:

```

concat = \[head1, head2, ..., headN]

```

- Final linear projection to combine head outputs

---

## 6. Feedforward Network (MLP)

- 2-layer MLP with non-linearity (ReLU or GELU)

```

x → Linear(n\_embed → 4*n\_embed) → ReLU → Linear(4*n\_embed → n\_embed)

```

---

## 7. Transformer Block

- Consists of:

1. Multi-Head Self-Attention
2. Residual connection
3. LayerNorm
4. Feedforward MLP
5. Residual connection
6. LayerNorm

- Sequential execution:

```python
x = x + mha(LayerNorm(x))
x = x + mlp(LayerNorm(x))
```

---

## 8. Full Model Assembly

- Stack of `n_layers` transformer blocks
- Final LayerNorm
- Final linear projection: `[n_embed] → [vocab_size]` (logits for softmax)

---

## 9. Loss and Training

- Cross-entropy loss between logits and target IDs
- Optimizer: AdamW
- Training loop:

  1. Get batch of input/target pairs
  2. Forward pass → logits
  3. Compute loss
  4. Backpropagate
  5. Update weights

---

## 10. Text Generation

- Autoregressive sampling:

  - Start with a context (e.g., `[A, B, C]`)
  - Predict next token
  - Append prediction to context
  - Repeat up to `max_new_tokens`

- Sampling strategies: greedy (argmax) or multinomial sampling with temperature

---

## ✅ Key Hyperparameters

- `vocab_size`: number of unique tokens
- `block_size`: context window (sequence length)
- `n_embed`: embedding dimension
- `n_head`: number of attention heads
- `n_layer`: number of transformer blocks

---

## 🧠 Core Equations

### Attention:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
```

### Transformer Block:

```text
x = x + MultiHead(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

---

## 🔍 Notable Simplifications

- No weight tying
- No dropout
- Learnable positional embeddings instead of sinusoidal
- Character-level input, not byte-pair encoding

---
