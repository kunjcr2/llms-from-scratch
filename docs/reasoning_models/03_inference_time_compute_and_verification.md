# Lecture 3: Inference Time Compute Scaling & Verification in Reasoning LLMs

## 1. Introduction

This lecture continues the discussion on **inference time compute scaling** for reasoning-based large language models (LLMs).  
Previously, prompting techniques like **Chain-of-Thought (CoT)** and **Zero-shot CoT** were explored.  
This lecture introduces **search against verifiers** as another inference-time computation strategy.

---

## 2. Recap: Prompting Techniques

### Chain-of-Thought (CoT) Reasoning

- Prompt model to explicitly generate intermediate reasoning steps before giving the answer.
- More effective in **larger models** than smaller ones.

### Zero-Shot CoT

- No input-output examples provided.
- Simply instructs the model to “think step-by-step”.
- Significantly outperforms plain zero-shot reasoning.

---

## 3. Search Against Verifiers

Instead of producing a single answer:

1. Generate **multiple answers** (A1, A2, A3, A4, …).
2. Use a **verification layer** to select the best one.
3. This increases **inference time compute** since extra processing is required.

Analogy: Picking the best crop after sampling multiple ones from the field.

---

## 4. Verification Layer

- Can be **human** or **model-based**.
- Model-based verifiers are called **Reward Models**.

**Example Reward Model:**  
[`reward-model-deberta-v3-large-v2`](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)

- Predicts which generated answer is better according to human judgment.
- Used in RLHF (Reinforcement Learning from Human Feedback) and toxicity detection.

---

## 5. Types of Reward Models

### Outcome Reward Models (OM)

- Evaluate **only the final answer**.
- Ignore reasoning steps.

### Process Reward Models (PRM)

- Score **each reasoning step** individually.
- Focus on correctness of reasoning path rather than just the final answer.
- More common in reasoning LLM verification.

**Example:**  
Question: Roger has 5 tennis balls, buys 2 cans of 3 balls each.  
Reasoning steps scored individually — wrong step gets low score.

---

## 6. Advantages of Verifiers

- No need to fine-tune or retrain the reasoning LLM.
- Improve answer reliability by filtering through a verification layer.

---

## 7. Types of Verifiers

1. **Majority Voting (Self-Consistency)**

   - Generate N answers, choose the most frequent.
   - No scoring; risk of majority being wrong.

2. **Best-of-N Sampling**

   - Generate N answers.
   - Use a verifier (OM or PRM) to select the highest-scoring answer.

3. **Beam Search Verification**
   - More advanced; explores multiple reasoning paths.

---

## 8. Beam Search in Detail

Originally proposed in **speech recognition** in 1976:  
**Reference Paper:** _The HARPY Speech Recognition System_ ([Link](https://www.isca-speech.org/archive/pdfs/asr_1976/allen76_asr.pdf))

### Basic Idea

- Maintain a fixed number of best candidates (**beam width m**) at each step.
- Extend each candidate by possible next steps.
- Keep top candidates based on scores from a verifier.

### Parameters

- **Beam Width (m):** Number of candidates kept at each step.
- **Number of Beams (n):** Initial number of generated paths.

---

## 9. Beam Search Example (Dummy Numbers)

**Task:** Solve: Roger has 5 balls, buys 2 cans of 3 balls each. How many balls?  
**Parameters:**

- Beam width = 2 (keep top 2 at each step)
- Number of beams = 4 (start with 4 initial thoughts)
- Scoring model = PRM (scores each reasoning step)

**Step 1 – Generate Initial Thoughts (n = 4):**
| Thought | Step Content | PRM Score |
|---------|--------------------------------------|-----------|
| T1 | "Roger starts with 5 balls" | 0.8 |
| T2 | "Roger starts with 6 balls" (wrong) | 0.2 |
| T3 | "Roger has 5 balls initially" | 0.75 |
| T4 | "Initial balls = 4" (wrong) | 0.3 |

✅ Keep top 2: **T1 (0.8)**, **T3 (0.75)**

---

**Step 2 – Expand Top 2 (Beam width = 2 each → total 4 new candidates):**
| Parent | New Thought | PRM Score |
|--------|--------------------------------------|-----------|
| T1 | "He buys 2 cans, each has 3 balls" | 0.85 |
| T1 | "He buys 3 cans, each has 3 balls" | 0.4 |
| T3 | "He buys 2 cans of 3 balls" | 0.8 |
| T3 | "He buys 2 cans, each has 2 balls" | 0.35 |

✅ Keep top 2: **(T1→2 cans, 3 balls)** (0.85), **(T3→2 cans, 3 balls)** (0.8)

---

**Step 3 – Expand Top 2 (Again total 4 candidates):**
| Parent | New Thought | PRM Score |
|-------------------|---------------------------------------|-----------|
| T1 path | "Total = 5 + (2×3) = 11 balls" | 0.9 |
| T1 path | "Total = 5 + (3×3) = 14 balls" (wrong)| 0.3 |
| T3 path | "Total = 5 + (2×3) = 11 balls" | 0.88 |
| T3 path | "Total = 5 + (2×2) = 9 balls" (wrong) | 0.25 |

✅ Keep top **1 final** (highest score): **"Total = 11 balls" (0.9)**

---

### Intuition

Beam search **balances exploration and pruning**:

- Wider beam width → more exploration.
- Smaller beam width → faster but more pruning.

---

## 10. Implementation Notes

- Models used:
  - Reasoning LLM: [`Zefiro-7B`](https://huggingface.co/giux78/zefiro-7b-dpo-qlora-ITA-v0.7)
  - PRM: [`reward-model-deberta-v3-large-v2`](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)
- Google Colab with A100 GPU recommended.
- Parameters to experiment with:
  - Beam width (m)
  - Number of beams (n)
  - Temperature for diversity

---

## 11. Paper Reference (Main)

**Title:** _Scaling Test-Time Compute for Reasoning_  
**Authors:** Ethan Perez et al.  
**arXiv Link:** [https://arxiv.org/abs/2307.06881](https://arxiv.org/abs/2307.06881)

---

## 12. Key Takeaways

- Inference-time compute scaling improves reasoning quality without retraining.
- Verification can be **majority voting**, **best-of-n**, or **beam search**.
- PRMs are powerful for step-wise reasoning evaluation.
- Beam search is a structured way to explore multiple reasoning paths efficiently.

---
