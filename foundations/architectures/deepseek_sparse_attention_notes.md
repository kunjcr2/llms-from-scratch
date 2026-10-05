# DeepSeek Sparse Attention (DSA) and Attention Mechanisms

This document outlines the first principles of attention mechanisms, their efficiency improvements, and how they culminate in DeepSeek Sparse Attention (DSA), which drastically reduces compute costs while matching performance.

## 1. Tokenization and Embeddings
- Large Language Models (LLMs) break input texts into pieces called **tokens**.
- **Tokenization:** Each token is assigned a unique ID in a dictionary.
- **Embeddings:** Because IDs lack semantic meaning, tokens are mapped to $d$-dimensional continuous vectors called **token embeddings**. Words with similar meanings have embeddings that are close in this space.
- Token embeddings initially contain no contextual sequence information.

## 2. Attention Mechanism
Attention helps the model determine how each token contributes to the others, adding contextual meaning.

### Vector Form
- **Query ($Q$) and Key ($K$) Vectors:** Token embeddings are multiplied by learned weight matrices $W_Q$ and $W_K$ to create $1 \times d_k$ Query and Key vectors.
- **Attention Scores:** We quantify relevance between tokens via the vector dot product of their $Q$ and $K$ vectors. 
  - To predict token by token (autoregressive), a mask ensures tokens only attend to past tokens.
  - Dot products are passed through a Softmax function to become a normalized probability distribution (Attention Weights).
- **Value ($V$) Vectors:** We extract contextual payload via weight matrix $W_V$.
- **Aggregation:** We sum the $V$ vectors, weighted by the attention scores, for each token.
- **Output:** An output projection matrix $W_O$ converts these back to $d$-dimensional space, creating a residual embedding that is added back to the original token embeddings.

### Matrix Form
- Stacking vectors into matrices $Q$, $K$, and $V$ allows efficient batch computation:
  $$ \text{Attention}(Q, K, V) = \text{softmax}(Q \cdot K^T + M) \cdot V $$
- $M$ is the masking matrix. The result is multiplied by $W_O$ to produce the residual embedding $\Delta x$.

## 3. Multi-Head Attention (MHA)
A single set of $Q, K, V$ matrices struggles to capture all complex relationships. 
- We introduce $h$ separate "heads" with different sets of projection matrices.
- The embeddings are split and projected. For each head $i$, we get output $O_i$.
- The outputs $O_1, ..., O_h$ are concatenated and passed through $W_O$.
- **Cost:** High computation and memory usage.

## 4. Key-Value (KV) Caching
- **Problem:** During inference (decoding token by token), the $K$ and $V$ vectors for all previously generated tokens never change. Recomputing them for every new token is wasteful.
- **Solution:** Cache the $K$ and $V$ vectors in memory. 
- **Cost:** This consumes massive memory (e.g., 131GB for a 32K sequence length on 128 heads).
- (Query vectors are not cached as they are only used for the current step).

## 5. Multi-Query Attention (MQA)
- **Idea:** Reduce the number of $K$ and $V$ heads from $h$ down to 1.
- All $h$ Query heads share the same single Key and Value vector per token.
- **Result:** Massive reduction in memory usage (e.g., 128-fold reduction).
- **Trade-off:** Decreased performance and less ability to capture complex attention relationships.

## 6. Grouped Query Attention (GQA)
- Strike a balance between MHA and MQA.
- Divide the $h$ heads into $n_g$ groups. Share a single $K$ and $V$ head per group.
- E.g., dividing 128 heads into 16 groups gives an 8-fold memory reduction.
- Used in Llama, Qwen, and Gemma models.
- **Mathematical Perspective:** $K$ and $V$ vectors are obtained through low-rank factorization, duplicated via a fixed up-projection matrix (like the identity matrix).

## 7. Multi-Head Latent Attention (MLA)
- DeepSeek uses learned **down-projection** and **up-projection** matrices instead of fixed grouping.
- **$KV$ Compression:** Embeddings are compressed via a down-projection matrix into a low-dimensional latent vector $c_{KV}$ (e.g., dimension $d_c = 576$).
- **Decompression:** Learned up-projection matrices reconstruct the required multi-head $K$ and $V$ vectors from $c_{KV}$.
- **Benefit:** Drastic memory reduction (cache the small $c_{KV}$ instead of full KV) while strictly improving performance over MHA.
- Queries can also undergo low-rank compression to reduce activation memory during training.
- **Inference Trick:** Up-projection matrices can be mathematically absorbed into $W_Q$ and $W_O$ matrices due to associativity, avoiding extra matrix multiplications at generation time.

## 8. Rotary Positional Embedding (RoPE) and Decoupled RoPE
- **RoPE:** Encodes position by applying 2D dimension rotations to $Q$ and $K$ vectors based on their position index. The dot product between rotated vectors then solely depends on their relative distance.
- **Problem with MLA:** The RoPE rotation occurs *between* the decompression up-projection and the dot product. Hence, the up-projection matrices cannot be absorbed into queries, and $K$ vectors must be recomputed every step.
- **Solution (Decoupled RoPE):** DeepSeek concatenates a separate, smaller shared Key and multi-head Queries that hold the explicitly rotated RoPE information. This allows MLA to work with absolute/relative positions efficiently.

## 9. DeepSeek Sparse Attention (DSA)
- As context length grows, calculating attention against all cached tokens slows down throughput.
- **Lightning Indexer:** A mechanism to quickly score relevance between the current query and all previous tokens, so we only compute full attention on the top relevant tokens.
- Computes partial queries/keys (shared keys, per-head queries, and partial RoPE) evaluated through a ReLU to obtain non-negative index scores.

### Quantization & Hadamard Transform
- Calculating the indexer score densely defeats the purpose.
- **Quantization:** Query and key vectors are quantized to low-precision 8-bit representations, allowing fast, coarse relevance approximation.
- **Outlier Issue:** Normal embeddings have outlier features (huge spikes), causing severe inaccuracy when naively quantized.
- **Hadamard Transform:** To fix this, DeepSeek applies a Fast Walsh-Hadamard Transform before quantization. This uniformly mixes/spreads out the values across all coordinates deterministically. It requires no matrix construction (fast addition/subtraction bounds) and retains information much better than random orthogonal matrices.

## 10. DSA Training Strategy
Trained in progressive stages so as not to ruin the large model:
1. **Dense Warm-Up:** Freeze the main model MLA layer. Train only the lightning indexer to predict a dense target distribution (derived from the mean attention scores of the heads).
2. **Fine-Grained Selection:** Unfreeze. Train the lightning indexer with fine-grained token selection using its own detached loss. The main model updates parameters purely based on typical language modeling loss, adapting to the sparse patterns the indexer provides.
