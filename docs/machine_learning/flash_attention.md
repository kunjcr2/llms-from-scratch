[FUCKING GOOD VIDEO](https://youtu.be/gBMO1JZav44?si=cKZDu5v2lZOZW0tp)

# ⚡ What Happens Inside FlashAttention? (Step by Step)

## 1. **Start with Token Embeddings**

- **What:** You feed in your sequence of tokens (words/parts of words), each turned into a vector (embedding).
- **Why:** This is standard for all transformers—tokens must be converted into numbers the model can work with.

## 2. **Divide the Sequence into Blocks**

- **What:** FlashAttention chops your sequence into small chunks, called **blocks** (e.g., 128 tokens at a time).
- **Why:** This keeps everything small enough to fit into the GPU’s fast memory (SRAM/cache), avoiding slow main memory.

## 3. **Compute Attention for Each Block**

- **What:** For each block, FlashAttention computes the Query, Key, and Value vectors (Q, K, V) as in regular attention.
- **How:** But instead of creating the full attention matrix for the entire sequence, it does this **one block at a time**—for example, only block 1 with block 2, etc.
- **Why:** This avoids creating a giant (sequence_length x sequence_length) attention matrix, saving lots of memory.

## 4. **Stream Blocks Through the GPU**

- **What:** The algorithm moves through the sequence, one block at a time, **calculating just what it needs and discarding the rest**.
- **How:** It reads in blocks, computes their partial attention results, and writes them out before moving to the next.
- **Why:** This streaming approach keeps only small chunks in fast memory, maximizing efficiency.

## 5. **Numerically Stable Softmax (Online Softmax)**

- **What:** To combine attention scores, it does the softmax calculation in a special way (in “online” fashion) that prevents rounding errors.
- **How:** As each block is processed, FlashAttention keeps track of running max values and sums—so softmax is correct even though the attention matrix was never fully built.
- **Why:** Regular softmax can have overflow/underflow problems on big inputs; online softmax fixes this, making FlashAttention both fast **and** precise.

## 6. **Assemble the Final Output**

- **What:** The result is a sequence of new vectors—just like normal attention, but built efficiently, block by block.
- **Why:** Now, the model can handle much longer texts without running out of memory, and it all happens much faster.

## ✨ **TL;DR (Quick Recap Table)**

| Step | What happens?                            | Why?                                |
| ---- | ---------------------------------------- | ----------------------------------- |
| 1    | Embed tokens as vectors                  | Numbers for computation             |
| 2    | Split into small blocks                  | Fits in fast memory (SRAM/cache)    |
| 3    | Compute QKV for each block               | No full attention matrix needed     |
| 4    | Stream/process blocks, discard after use | Keeps memory use very low           |
| 5    | Online softmax for stability             | Accurate results, no errors         |
| 6    | Output re-assembled from all blocks      | Final sequence for next layer/model |

## 🧠 **Why Do All This?**

- **Classic attention** needs too much memory for long texts.
- **FlashAttention** lets LLMs run on much longer sequences (think 32k tokens or more) and makes both training and inference _much_ faster and cheaper.
