# BLEU Evaluation Metric (Mini Guide)

## What is BLEU?
**BLEU** (Bilingual Evaluation Understudy) measures how similar a candidate text is to reference(s) by counting **n-gram overlaps**.

### Key Concepts
| Concept | Explanation |
|---------|-------------|
| **N-grams** | Contiguous word sequences (1-gram="the", 2-gram="the cat", etc.). BLEU typically uses 1–4 grams. |
| **Clipped Precision** | Limits each n-gram count to its max occurrence in any reference—prevents gaming by repeating words. |
| **Brevity Penalty (BP)** | Penalizes candidates shorter than references. Without it, outputting one correct word scores 100% precision. |

### Why BLEU Can Mislead
- **Paraphrases**: "The cat sat" vs "A feline rested" share meaning but few n-grams.
- **Meaning mismatch**: "I love you" vs "I hate you" differ by one word but opposite meaning.
- **Word order**: BLEU ignores global structure; nonsense orderings can score well.

---

## Worked Example
| | Sentence |
|-|----------|
| **Reference** | the cat sat on the mat |
| **Candidate** | the cat on the mat |

**1-gram matches**: the(2), cat(1), on(1), mat(1) → 5/5 = 1.0  
**2-gram matches**: the cat(1), cat on(0), on the(1), the mat(1) → 3/4 = 0.75

---