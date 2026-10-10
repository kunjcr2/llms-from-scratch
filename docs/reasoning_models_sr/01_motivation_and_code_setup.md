# Lecture 1 — Motivation and Code Setup

**Video:** [Build A Reasoning Model From Scratch 1: Motivation & Code Setup](https://www.youtube.com/watch?v=Kh9mqTzjuEQ) (43:25)

## 1. What this series builds

A reasoning model is not a separate neural-network species. It is still an LLM—usually the same Transformer architecture—but post-training and inference methods make it produce useful intermediate reasoning and solve harder tasks more reliably.

The conceptual stack is:

```text
conventional pretrained LLM
        ↓ reasoning-oriented post-training/inference
reasoning LLM
        ↓ wrapped by tools, memory, control flow, UI, etc.
agent harness
```

- The **LLM** is the language-model engine.
- The **reasoning model** is an LLM with reasoning-oriented behavior.
- An **agent harness** is software around the model. Coding agents and tool-using assistants are harnesses whose engine is commonly a reasoning model.

Modern products may combine ordinary and reasoning behavior in one model. A reasoning-effort setting can steer how much computation or how many reasoning tokens it uses, so “LLM” and “reasoning model” are no longer mutually exclusive labels.

## 2. Why implement methods from scratch?

The goal is educational clarity, not reproducing a frontier model at frontier scale.

- A diagram gives intuition; executable code removes ambiguity.
- Reimplementing a method exposes assumptions, data flow, tensor shapes, and failure cases.
- Running the code provides evidence that the conceptual description really works.
- A strong foundation makes later ideas—sampling, watermarking, scoring, RL, and agent systems—easier to understand.
- Even when an AI system writes production code, knowing how to read and reason about code helps identify mistakes and direct changes.

“From scratch” is scoped carefully here. The course starts with a pretrained Qwen3 model. It implements reasoning and evaluation methods from scratch on top of that model; it does not repeat the full Transformer pretraining journey.

## 3. Learning path

The series follows the dependency order required to measure genuine improvement:

1. Load a conventional pretrained LLM.
2. Build a verifier and establish a baseline.
3. Apply inference-time scaling without changing model weights.
4. Train reasoning behavior with reinforcement learning.
5. Later in the wider book/course, study distillation as an efficient alternative for small models.

Evaluation comes before improvement because an unmeasured technique cannot be distinguished from a convincing demo.

### Relationship to the earlier LLM-from-scratch material

- The earlier material covers architecture, pretraining, and conventional fine-tuning.
- This series starts from an existing base model and adds evaluation and reasoning methods.
- Either series can be studied first. Following current curiosity can be more productive than enforcing a rigid order.

## 4. Hardware philosophy

The project uses a small model so the techniques remain accessible.

- Chapters/lectures 2–4 are practical on a CPU and also support CUDA, Apple MPS, and Intel XPU.
- The RL and distillation stages are computationally expensive; a CUDA GPU is strongly preferred.
- A small educational run demonstrates the same algorithmic structure as a large run, but does not reproduce frontier-model quality.
- Training a DeepSeek-scale model would require a large team and budget; recreating an already available model is not the purpose of the course.

## 5. Reproducible setup with `uv`

The official repository contains `pyproject.toml` plus a lockfile. The lockfile records a compatible, resolved dependency set; `uv sync` creates a project-local virtual environment and installs it.

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
uv sync
uv run jupyter lab
```

Why a local environment matters:

- It isolates this project's packages from system Python.
- The notebook and terminal can use the same interpreter.
- Deleting the project environment removes it cleanly.
- The repository can update dependency guidance when libraries or operating systems change.

Check the active interpreter inside a notebook:

```python
import sys
import torch

print(sys.prefix)
print(sys.executable)
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
```

If the terminal works but a notebook cannot import the same packages, the notebook probably selected a different kernel. Point VS Code or Jupyter at the Python executable printed by:

```bash
uv run python -c "import sys; print(sys.executable)"
```

## 6. Local and remote workflows

The workflow is the same on a local machine or rented GPU:

1. Connect to the machine (for example, VS Code Remote SSH).
2. Clone the repository on that machine.
3. Run `uv sync` there.
4. Select that environment's interpreter as the notebook kernel.
5. Confirm accelerator availability before a long run.

If a CUDA machine reports `False` for `torch.cuda.is_available()`, investigate the PyTorch build, NVIDIA driver, and bundled CUDA compatibility. A possible targeted update is:

```bash
uv sync --upgrade-package torch
```

Do not blindly run remote installation scripts. Inspect them first or choose a package-manager installation method you trust.

## 7. Platform caveats

- Apple MPS support has improved substantially, but numerical behavior and training stability may still differ from CPU/CUDA.
- A GPU being physically present does not mean the installed PyTorch build can use it.
- Package versions matter when comparing results; record Python, PyTorch, tokenizer, and project versions.
- The official repository is the preferred source because fixes and compatibility notes can evolve after a video is recorded.

## Takeaways

- A reasoning model is an LLM with added reasoning behavior, not necessarily a different architecture.
- An agent is the larger software system around that model.
- “From scratch” means implementing the reasoning machinery transparently at an affordable scale.
- Reproducibility starts with an isolated environment and an explicitly selected interpreter.
- The next step is to load the base model and understand exactly how it generates one token at a time.

