# 🪟 Sliding Window Attention & Longformer

## What is Sliding Window Attention?

**Sliding Window Attention** is a technique designed to let transformer models handle very long input sequences (like thousands of tokens) **efficiently**—much more than standard self-attention allows.

- In standard self-attention, **every token attends to every other token**, so memory and compute cost grow **quadratically** with input length (`O(n²)`).
- **Sliding window attention** restricts each token to only attend to a fixed number (`w`) of neighboring tokens—its “window”.

---

## How Does It Work?

Let’s say your **window size = 4** (2 tokens before, 2 tokens after):

```text
Input tokens:   A B C D E F G
                | | | | | | |
Token C attends to:  A B C D E
Token F attends to:  D E F G
```

- Each token only interacts with tokens in its window.
- The “window” slides across the sequence.

**Result:**

- Complexity reduces to **linear** with sequence length (`O(nw)` where `w` is the window size).

---

## Longformer: Expanding on the Idea

**Longformer** is a transformer architecture that popularized sliding window attention for real-world tasks.

### Key Features:

- **Sliding Window Attention:** Each token attends only to its window.
- **Dilated/Global Attention:** Select tokens (e.g., \[CLS], headlines) can still attend to all tokens—useful for summarization or classification tasks.

### Visual Example:

```
Sequence:         T1 T2 T3 T4 T5 T6 T7 T8
Window size=3:     |   |   |   |   |   |
T4 attends to:     T2 T3 T4 T5 T6
```

---

## Why Use Sliding Window Attention?

- **Scalability:** Enables handling inputs with **tens of thousands** of tokens.
- **Efficiency:** Reduces computation and memory footprint dramatically.
- **Local Patterns:** Works great for texts where local context matters (e.g., long documents, DNA sequences).

---

## Summary Table

| Feature                   | Standard Self-Attention | Sliding Window (Longformer)  |
| ------------------------- | ----------------------- | ---------------------------- |
| Complexity                | O(n²)                   | O(nw)                        |
| Sequence length supported | Short (2k–4k)           | Very Long (16k, 32k, 100k+)  |
| Attends to                | All tokens              | Local window (+ global, opt) |

---

## References

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- [Official Longformer GitHub](https://github.com/allenai/longformer)
