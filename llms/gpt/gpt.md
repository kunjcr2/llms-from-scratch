# How LLMs Are Made (From Scratch)

> A simple pipeline outlining how GPT-like models are built from scratch.

---

## Data Preprocessing Pipeline

We start by loading raw text data, convert it into token encodings using **tiktoken**, then create datasets and dataloaders. Finally, token embeddings are extracted through an embedding layer.

1. **Load the data** into a variable.
2. Use `tiktoken` to get the GPT-2 vocabulary (size \~50,257):

   ```python
   vocab = tiktoken.get_encoding("gpt2")
   ```

3. Create a **Dataset class** with `input_ids` and `target_ids` tensors:

   - Use `max_length` (context length) and `stride` to segment the data.

4. Build a **DataLoader** with a specified batch size.
5. Create a **token embedding layer** with parameters `(vocab_size, embedding_dim)`:

   - It acts as a lookup table initialized randomly.
   - Embeddings are learned during training.
   - Total parameters = vocab size × embedding dimension.

6. Pass `input_ids` through the token embedding layer → **token embeddings**.
7. Create a **positional embedding layer** `(context_length, embedding_dim)`.
8. Add **token embeddings + positional embeddings** → input embeddings to the model.

---

## Attention Mechanism

Attention helps the model understand relationships between tokens, especially for long sequences where RNNs fail.

1. **Simplified attention** (base level):

   - Calculate attention scores by dot product of token embeddings.
   - Apply softmax to normalize scores.
   - Multiply attention weights by embeddings → **context embeddings**.

2. **Self-attention**:

   - Learn weights for query (Q), key (K), and value (V) matrices.
   - Compute scores: `Q @ K.T / sqrt(d_k)` followed by softmax.
   - Compute output: `attention_weights @ V`.

3. **Causal attention** (to prevent peeking at future tokens):

   - Use an upper triangular mask to set future token scores to -∞.
   - Softmax makes masked scores zero, allowing only past and present tokens.
   - Apply dropout for regularization.

4. **Multi-headed attention**:

   - Multiple causal attention heads run in parallel.
   - Concatenate all heads’ outputs column-wise.

5. **Efficient multi-head attention**:

   - Combine all Q, K, V matrices into large matrices for one big matrix multiplication.
   - Improves efficiency compared to running separate attention heads.
   - Refer to `./attention/LLM_attention.ipynb` for detailed implementation.

---

## LLM Architecture

1. **Data flow**:

   - Input embeddings → Transformer blocks → output logits.
   - Transformer block components:

     - Layer normalization
     - Multi-head attention
     - Feed-forward neural networks (FFN)

2. **Layer Normalization**:

   - Prevents exploding/vanishing gradients.
   - Normalizes input:

     $$
     \text{output} = \text{scale} \times \frac{x_i - \text{mean}}{\sqrt{\text{variance} + \epsilon}} + \text{shift}
     $$

   - `scale` and `shift` are trainable parameters.

3. **Feed Forward Network (FFN)**:

   - Two linear layers with GeLU activation in between.
   - GeLU is smoother and better than ReLU:

     ```python
     0.5 * x * (1 + torch.tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
     ```

   - The hidden layer size is typically 4× the embedding dimension.

4. **Residual connections**:

   - Add the input of the block to its output.
   - Helps gradients flow through the network (prevents vanishing).

5. Combined, these form a **Transformer block** (covered in lecture 23).

---

## Complete GPT Architecture

- Full GPT-2 architecture code is available at:
  `./architecture/LLM-GPT-arch.ipynb`
- The entire model is encapsulated in a class, e.g.:

  ```python
  model = GPTModel(GPT_124M_CONFIG)
  ```

---

## Training GPT Model

1. **Loss function: Cross-entropy**

   - Calculated using:

     ```python
     torch.nn.functional.cross_entropy(logits_flat, targets_flat)
     ```

   - Measures how well the predicted logits match the target tokens.
   - Perplexity is derived from cross-entropy.

2. Split dataset (`the_verdict.txt`) into training and validation sets.
3. Use DataLoaders for batching and evaluate loss on both sets.
4. **Training loop** (see `LLM-training.ipynb`):

   - Optimizer: AdamW
   - Steps per batch:

     ```python
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     ```

5. **Temperature scaling** controls model creativity at generation:

   - Higher temperature → softer probability distribution → more diverse tokens.
   - Lower temperature → sharper probabilities → more deterministic.

6. **Sampling strategies**:

   - `generate()` function uses top-k sampling + temperature scaling.
   - Selects the top-k logits, scales with temperature, samples next token probabilistically.

7. **Model saving/loading**:

   - Handled post-training (see `5_post_training` image and notes).

---

## Final Notes

- Detailed finetuning notes: `6_fine_tuning` folder

---

_Visited on 6/21/2025_
