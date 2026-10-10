# Build a Reasoning Model From Scratch — Lecture Notes

Concise, implementation-oriented notes for Sebastian Raschka's six-video series. The notes follow the lectures' order and preserve the instructor's progression: establish a base model, build a verifier, improve inference, and finally train with reinforcement learning.

## Course map

| Lecture | Topic | Main outcome |
|---|---|---|
| [1](01_motivation_and_code_setup.md) | Motivation and code setup | Understand the base LLM → reasoning model → agent-harness stack and create a reproducible environment. |
| [2](02_base_model_generation_and_kv_cache.md) | Base model, generation, and KV caching | Load Qwen3-0.6B, implement autoregressive decoding, and accelerate it. |
| [3](03_verifier_evaluation_and_rlvr.md) | Verifier-based evaluation | Extract, normalize, and grade MATH-500 answers to establish a reliable baseline. |
| [4](04_inference_scaling_sampling_and_self_consistency.md) | Inference scaling I | Implement temperature, top-p sampling, chain-of-thought prompting, and self-consistency. |
| [5](05_logprob_scoring_and_self_refinement.md) | Inference scaling II | Score answers with heuristics/log-probabilities and implement iterative self-refinement. |
| [6](06_rlvr_and_grpo.md) | Reinforcement learning I | Implement RL with verifiable rewards using a readable GRPO training loop. |

## The complete pipeline

```text
pretrained Qwen3-0.6B base model
        │
        ├── verifier + MATH-500 baseline
        │
        ├── inference-time improvements
        │     ├── chain-of-thought prompt
        │     ├── temperature + top-p sampling
        │     ├── self-consistency / Best-of-N
        │     └── self-refinement
        │
        └── training-time improvement
              └── RLVR with GRPO
```

The important distinction is whether a method changes model weights:

- **Inference-time scaling** spends more compute while answering; model weights stay fixed.
- **Training-time scaling** spends compute updating the model; the improved behavior is stored in its weights.
- They are complementary: inference-time methods can be applied after RL training.

## Reference results from the lectures

These are teaching-run results on all 500 MATH-500 problems, not universal model rankings. Runtime and even accuracy can vary with prompts, seeds, PyTorch versions, and hardware.

| Method | Model | Accuracy |
|---|---|---:|
| Greedy baseline | Qwen3-0.6B Base | 15.2% |
| Official reasoning variant | Qwen3-0.6B reasoning | 48.2% |
| Chain-of-thought prompting | Base | 40.6% |
| CoT + top-p + self-consistency, `n=10` | Base | 52.0% |
| One self-refinement pass, no scorer | Base | 25.0% |
| GRPO after 50 steps | Base → trained model | 47.4% |

The series' recurring lesson is that more compute is not automatically better. More samples, critique rounds, or training steps help only when the selection signal, verifier, and optimization remain reliable.

## Primary sources

- [Official companion repository](https://github.com/rasbt/reasoning-from-scratch)
- [Companion hub](https://sebastianraschka.com/reasoning-from-scratch/)
- [Lecture 1](https://www.youtube.com/watch?v=Kh9mqTzjuEQ)
- [Lecture 2](https://www.youtube.com/watch?v=BJua0yjO5dk)
- [Lecture 3](https://www.youtube.com/watch?v=JQJ_8_jSAoY)
- [Lecture 4](https://www.youtube.com/watch?v=t5y-kS9nNxU)
- [Lecture 5](https://www.youtube.com/watch?v=TVMyOJ_3Gxo)
- [Lecture 6](https://www.youtube.com/watch?v=237Hf7Q3lgg)

