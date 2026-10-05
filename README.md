# LLMs from Scratch

A personal knowledge base and learning portfolio covering large language models, machine learning, deep learning, reinforcement learning, computer vision, and applied AI projects.

This repository brings together the theory, implementations, experiments, notebooks, and references I use to understand modern AI systems from first principles. It is organized both as a record of what I have learned and as a resource for anyone following a similar path.

**Portfolio:** [kunjcr2.github.io](https://kunjcr2.github.io)

## Repository Overview

```text
llms-from-scratch/
├── foundations/       # Mathematics, core architectures, and PyTorch
├── docs/              # Topic-focused notes, guides, and implementations
├── llms/              # LLM architectures and from-scratch learning material
├── computer_vision/   # Vision notes, implementations, notebooks, and diagrams
├── projects/          # End-to-end models, fine-tuning, and research experiments
└── resources/         # Curated papers and further reading
```

All regular files and folders use descriptive `snake_case` names. Conventional ecosystem filenames such as `README.md`, `Dockerfile`, and `.gitignore` retain their standard names.

## Featured Work

### MedAssistGPT

[MedAssistGPT](projects/med_assist_gpt/) is a 401M-parameter medical-domain language model pretrained from scratch on two million PubMed abstracts.

Key architectural and training features include:

- Rotary positional embeddings (RoPE)
- Grouped-query attention (GQA) with four key-value groups
- SwiGLU feed-forward layers
- RMSNorm and 24 Transformer blocks
- Flash Attention optimization for A100 GPUs
- Memory-mapped datasets and multiprocessing
- Gradient accumulation, checkpointing, and Weights & Biases integration

[View the model on Hugging Face](https://huggingface.co/kunjcr2/MedAssistGPT)

### Other selected projects

| Project | Focus |
| --- | --- |
| [AdaptRoute](projects/adapt_route/) | Task-aware small-language-model routing with soft LoRA merging |
| [GatorGPT](projects/gator_gpt/) | Transformer language model with GQA, RoPE, and SwiGLU |
| [LLM Firewall](projects/llm_firewall/) | Prompt-injection detection and adversarial safety experiments |
| [Qwen 0.5B GRPO](projects/qwen_0_5b_grpo/) | SFT, RFT, GRPO, GSPO, and Dr. GRPO experiments on GSM8K |
| [Qwen 2.5 SFT + DPO](projects/qwen_2_5_0_5b_sft_dpo/) | Supervised fine-tuning followed by preference optimization |
| [Neural Sort](projects/neural_sort/) | Sequence sorting with pointer networks |
| [SVD Recommender](projects/svd/) | Recommendation experiments using singular value decomposition |

See the [projects index](projects/README.md) for additional work.

## Knowledge Areas

### Foundations

The [foundations](foundations/) section contains the material needed before moving into full model architectures:

- [Mathematics for machine learning](foundations/mathematics/): statistics, probability, linear algebra, calculus, and second-order methods
- [PyTorch reference](foundations/pytorch/): tensors, autograd, training loops, computer vision, NLP, Transformers, and classical ML
- [Architecture fundamentals](foundations/architectures/): mixture of experts, sparse attention, and per-layer embeddings

### Large Language Models

The [llms](llms/) section focuses on model internals and from-scratch implementations.

| Area | Coverage |
| --- | --- |
| [GPT](llms/gpt/) | Tokenization, attention, architecture, pretraining, post-training, and fine-tuning |
| [DeepSeek](llms/deepseek/) | MLA, MoE, RoPE, KV cache, MTP, sparse attention, and hyper-connections |
| [Mamba](llms/mamba/) | State-space models, selective SSMs, and Mamba implementations |
| [Mixture of Depths](llms/mixture_of_depths/) | Dynamic compute allocation, recurrent depth, and looped Transformers |
| [World Models](llms/world_models/) | Simulators, RSSMs, Iris, I-JEPA, energy-based models, and V-JEPA |

Additional focused references cover [Gated DeltaNet](llms/gated_delta_net.md), [Kimi Delta Attention](llms/kimi_gated_delta_net.md), [Jev architecture](llms/jev_architecture.md), sparse attention, and Qwen architectures.

### Machine Learning and AI Topics

The [docs](docs/) section groups broader topics by subject:

| Topic | Material |
| --- | --- |
| [Machine learning](docs/machine_learning/) | Backpropagation, attention, autoencoders, model merging, quantization, prompting, security, and production ML |
| [Reinforcement learning](docs/reinforcement_learning/) | A complete sequence from fundamentals through PPO, reward modeling, DPO, and GRPO |
| [Reasoning models](docs/reasoning_models/) | Reasoning LLMs, inference-time compute, and verification |
| [RAG](docs/rag/) | Data ingestion, chunking, embeddings, retrieval, and vector storage |
| [Optimization](docs/optimization/) | SGD, momentum, RMSProp, Adam, vLLM inference, multiprocessing, and multithreading |
| [Evaluation](docs/evaluation/) | BLEU, classification metrics, and regression metrics |
| [System design](docs/system_design_annotated_notes.md) | Annotated system-design notes |

The reinforcement-learning section also includes an [interactive mind map](docs/reinforcement_learning/rl_mindmap.html) and runnable algorithm implementations in its [`code`](docs/reinforcement_learning/code/) directory.

### Computer Vision

The [computer_vision](computer_vision/) section contains:

- Architecture notes for CNNs, DeiT, Swin Transformer, DETR, CLIP, LLaVA, Flamingo, SAM, TimeSformer, DDPM, and vision-language models
- Python implementations in [`computer_vision/code`](computer_vision/code/)
- Interactive architecture flowcharts in [`computer_vision/flowcharts`](computer_vision/flowcharts/)
- A video-classification notebook using R3D-18 and UCF101

## Suggested Learning Paths

### Build an LLM from first principles

1. Review the [mathematics foundations](foundations/mathematics/).
2. Work through the [PyTorch reference](foundations/pytorch/README.md).
3. Follow the six-part [GPT series](llms/gpt/).
4. Study modern components in the [DeepSeek notes](llms/deepseek/notes/).
5. Explore complete training work in [GatorGPT](projects/gator_gpt/) or [MedAssistGPT](projects/med_assist_gpt/).

### Learn post-training and alignment

1. Begin with [reinforcement-learning fundamentals](docs/reinforcement_learning/notes/01_rl_fundamentals.md).
2. Continue through policy gradients, PPO, reward modeling, DPO, and GRPO.
3. Review the runnable implementations in [reinforcement-learning code](docs/reinforcement_learning/code/).
4. Compare the techniques with the [Qwen GRPO experiments](projects/qwen_0_5b_grpo/).

### Study multimodal models and world models

1. Review [vision-language model notes](computer_vision/notes/vision_language_models.md).
2. Study CLIP, LLaVA, Flamingo, SAM, and video Transformers in [computer vision](computer_vision/notes/).
3. Follow the ordered [world-model lecture series](llms/world_models/), from basic simulators through V-JEPA.

## Notebook Index

| Topic | Notebook |
| --- | --- |
| GPT tokenizer | [`llms/gpt/1_tokenizer/tokenizer.ipynb`](llms/gpt/1_tokenizer/tokenizer.ipynb) |
| Attention | [`llms/gpt/2_attention/attention.ipynb`](llms/gpt/2_attention/attention.ipynb) |
| GPT architecture | [`llms/gpt/3_architecture/gpt_architecture.ipynb`](llms/gpt/3_architecture/gpt_architecture.ipynb) |
| GPT pretraining | [`llms/gpt/4_training/pretraining.ipynb`](llms/gpt/4_training/pretraining.ipynb) |
| GPT post-training | [`llms/gpt/5_post_training/post_training.ipynb`](llms/gpt/5_post_training/post_training.ipynb) |
| LoRA fine-tuning | [`llms/gpt/6_fine_tuning/lora_fine_tuning.ipynb`](llms/gpt/6_fine_tuning/lora_fine_tuning.ipynb) |
| DeepSeek implementation | [`llms/deepseek/code/deepseek_complete.ipynb`](llms/deepseek/code/deepseek_complete.ipynb) |
| Mamba | [`llms/mamba/mamba.ipynb`](llms/mamba/mamba.ipynb) |
| Reasoning and verification | [`docs/reasoning_models/03_inference_time_compute_and_verification.ipynb`](docs/reasoning_models/03_inference_time_compute_and_verification.ipynb) |
| Backpropagation | [`docs/machine_learning/backpropagation.ipynb`](docs/machine_learning/backpropagation.ipynb) |
| Vision Transformer demo | [`computer_vision/code/vit_demo.ipynb`](computer_vision/code/vit_demo.ipynb) |
| Video classification | [`computer_vision/code/r3d18_ucf101_video_classification.ipynb`](computer_vision/code/r3d18_ucf101_video_classification.ipynb) |
| MedAssistGPT | [`projects/med_assist_gpt/med_assist_gpt.ipynb`](projects/med_assist_gpt/med_assist_gpt.ipynb) |
| Neural Sort | [`projects/neural_sort/neural_sort.ipynb`](projects/neural_sort/neural_sort.ipynb) |

## Papers and References

[`resources/papers.md`](resources/papers.md) contains a curated reading list covering model architectures, mixture-of-experts systems, efficient inference, alignment, safety, fine-tuning, multimodal learning, and evaluation.

## Using the Repository

Clone the repository and open the topic or project you want to study:

```bash
git clone https://github.com/kunjcr2/llms-from-scratch.git
cd llms-from-scratch
```

This is a collection of independent learning modules rather than one installable Python package. Dependencies vary by project and notebook; consult the closest `README.md`, notebook imports, or script imports before running a component.

---

Created and maintained by [Kunj Shah](https://kunjcr2.github.io).
