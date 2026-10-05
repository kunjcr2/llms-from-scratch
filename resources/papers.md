# Papers

---

## Architecture Foundations

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://openai.com/index/language-unsupervised/) — Radford et al., 2019
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al., 2016
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2016
- [An Image is Worth 16x16 Words: Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [A Path Towards Autonomous Machine Intelligence (JEPA)](https://openreview.net/forum?id=BZ5a1r-kVsf) — LeCun, 2022
- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)](https://arxiv.org/abs/2301.08243) — Assran et al., 2023
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — Liu et al., NeurIPS 2023
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) — El-Nouby et al., 2022
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Qwen Team, 2025
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) — Qwen Team, 2025
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model`](https://arxiv.org/abs/2405.04434)

---

## Mixture of Experts

- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — Fedus et al., 2021
- [DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models](https://arxiv.org/abs/2401.06066) — Dai et al., 2024
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024

---

## DeepSeek V4

### Official DeepSeek Papers

- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) — Xie et al., 2025. Pre-V4 architectural paper introducing manifold-constrained hyper-connections (mHC) to restore stable identity-like residual behavior in widened hyper-connections.
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) — DeepSeek-AI et al., submitted April 26, 2026. Main V4 technical report covering V4-Pro and V4-Flash, 1M-token context, hybrid CSA/HCA attention, mHC, and Muon-based training.
- [DSpark Paper (in the official DeepSpec repo)](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) — DeepSeek, June 2026. Official speculative-decoding work released alongside [DeepSpec](https://github.com/deepseek-ai/DeepSpec), DeepSeek's training and evaluation codebase for speculative decoding algorithms.

### Related Third-Party / Collaborative Work

- [FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention](https://arxiv.org/abs/2606.09079) — Wang et al., 2026. Third-party work built on top of the DeepSeek-V4 architecture for long-context KV-cache reduction via lookahead sparse attention.
- [DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference](https://arxiv.org/abs/2602.21548) — Wu et al., 2026. Collaborative infrastructure work from the broader DeepSeek research line on KV-cache loading and storage-to-GPU bottlenecks in agentic inference.

### Commonly Confused With V4, But Not Actually V4 Papers

- [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372) — Cheng et al., 2026. This is the Engram paper often speculated to be V4-related before release, but it is not part of the final V4 technical report.
- `Muon` optimizer — originally from Keller Jordan et al., 2024; DeepSeek-V4 adopts and scales it, but it is not itself a DeepSeek V4 paper.

---

## Efficiency and Compute

- [Mixture-of-Depths: Dynamically Allocating Compute in Transformer Models](https://arxiv.org/abs/2404.02258) — Raposo et al., 2024
- [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524) — 2025
- [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871) — Jolicoeur-Martineau, 2024
- [Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734) — Wang et al., 2025

- [StreamIndex: Memory-Bounded Compressed Sparse Attention via Streaming Top-k](https://arxiv.org/pdf/2605.02568)

## Model Compression and Quantization

- [ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](https://arxiv.org/abs/2502.02631) — Liu et al., 2025. Unified framework for 1-bit, 1.58-bit, 2-bit, 3-bit, and 4-bit quantization that compares PTQ and QAT across the low-bit regime.

---

## Alignment and RLHF

- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022
- [Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al., 2022
- [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773) — Dai et al., 2023
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Gao, Schulman, Hilton, 2022
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — Sutton et al., 1999

---

## Safety and Jailbreaking

- [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483) — Wei et al., 2023
- [Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)](https://arxiv.org/abs/2307.15043) — Zou et al., 2023
- [Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR)](https://arxiv.org/abs/2310.08419) — Chao et al., 2023
- [Tree of Attacks with Pruning (TAP)](https://arxiv.org/abs/2312.02119) — Mehrotra et al., NeurIPS 2024
- [AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs](https://arxiv.org/abs/2410.05295) — Liu et al., ICLR 2025
- [The Crescendo Multi-Turn LLM Jailbreak Attack](https://arxiv.org/abs/2404.01833) — Russinovich et al., Microsoft, 2024
- [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352) — 2024
- [Adaptive Attacks Break Defenses Against Indirect PI](https://arxiv.org/abs/2503.00061) — Zhan et al., NAACL 2025
- [Indirect Prompt Injections: Are Firewalls All You Need?](https://arxiv.org/abs/2510.05244) — Shi et al., 2025
- [Defeating Prompt Injections by Design (CaMeL)](https://arxiv.org/abs/2503.18813) — Google DeepMind, 2025
- [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://arxiv.org/abs/2009.11462) — Gehman et al., 2020
- [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://arxiv.org/abs/2110.08193) — Parrish et al., 2021
- (110 pages long) [DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models](https://arxiv.org/abs/2306.11698) — Wang et al., 2023
- [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674) (Llama Guard 2 / Llama Guard 3 Technical Reports) — Inan et al., 2023
- [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249) — Mazeika et al., 2024
- [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260) — Souly et al., NeurIPS 2024
- [WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs](https://arxiv.org/abs/2406.18495) — Han et al., NeurIPS 2024
- [SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](https://arxiv.org/html/2502.12562v3)

---

## Fine-Tuning and PEFT

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023

---

## Multimodal Safety Benchmarks

- **[UniSAFE (2026)](https://arxiv.org/pdf/2603.17476)**
- **[Uni-Safebench: Does Unification Come at a Cost?](https://arxiv.org/pdf/2604.00547)**
- **[SaLAD: When Helpers Become Hazards](https://arxiv.org/pdf/2601.04043)**
- **[The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning](https://arxiv.org/pdf/2601.14127)**
- **[SafeVid](https://openreview.net/pdf?id=SeNFo7JGly)**
  - [Dataset (SafeVid-350K)](https://huggingface.co/datasets/yxwang/SafeVid-350K)
- **[Red Teaming Visual Language Models](https://arxiv.org/abs/2401.12915)**
- **[OmniSafeBench-MM: A Unified Benchmark and Toolbox for Multimodal Jailbreak Attack-Defense Evaluation](https://arxiv.org/abs/2512.06589)**
- **[AILuminate Multimodal Safety Test Suite v0.5](https://mlcommons.org/wp-content/uploads/2025/12/MLCommons-Security-Jailbreak-0.5.1.pdf)** → **[MSTS: A Multimodal Safety Test Suite for Vision-Language Models](https://arxiv.org/abs/2501.10057)**
- **[ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks](https://arxiv.org/abs/2510.02677)**
- **[Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](https://arxiv.org/abs/2505.11842)**
- **[Trust-VideoLLMs: Benchmarking the Trustworthiness in Multimodal LLMs for Video Understanding](https://arxiv.org/abs/2506.12336)**

---

## Safety Datasets

- [BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://arxiv.org/abs/2307.04657) — Ji et al., NeurIPS 2023
- [HADES: Images are Achilles' Heel of Alignment — Exploiting Visual Vulnerabilities for Jailbreaking Multimodal LLMs](https://arxiv.org/abs/2403.09792) — Li et al., 2024
- [MMDT: Decoding the Trustworthiness and Safety of Multimodal Foundation Models](https://arxiv.org/abs/2503.14827) — Xu et al., 2025
- [MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models](https://arxiv.org/abs/2311.17600) — Liu et al., 2023
- [SIUO: Cross-Modality Safety Alignment](https://arxiv.org/abs/2406.15279) — Wang et al., NAACL 2025
- [SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Model](https://arxiv.org/abs/2406.12030) — Zhang et al., 2024
- [VLGuard: Safety Fine-Tuning at (Almost) No Cost — A Baseline for Vision Large Language Models](https://arxiv.org/abs/2402.02207) — Zong et al., 2024
- [VLSBench: Unveiling Visual Leakage in Multimodal Safety](https://arxiv.org/abs/2411.19939) — Hu et al., 2024

---

## Courses and Books

- [HuggingFace Smol Course](https://github.com/huggingface/smol-course) — Hugging Face, 2024
- [ATLAS](https://arxiv.org/pdf/2510.22037) - Google Research
