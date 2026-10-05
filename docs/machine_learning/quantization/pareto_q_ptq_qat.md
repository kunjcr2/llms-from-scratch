# ParetoQ, PTQ, and QAT

This note connects three ideas that often get mixed together:

- **PTQ**: post-training quantization, where you quantize after training.
- **QAT**: quantization-aware training, where quantization is part of training.
- **ParetoQ**: a unified low-bit quantization framework that compares and improves both training and quantization design across 1-bit to 4-bit settings.

## 1. Quick intuition

Think of quantization as shrinking the model's weights from floating point into a smaller set of discrete values.

- **PTQ** is the cheapest route: train in full precision, then compress afterward.
- **QAT** is more expensive: the model learns while seeing quantization noise, so it can adapt.
- **ParetoQ** asks a sharper question: if we care about the accuracy-size tradeoff across very low bit widths, what combination of training schedule and quantizer actually gives the best Pareto frontier?

The key ParetoQ takeaway is that there is no single universally best bit-width. The best choice depends on whether you want maximum compression, better accuracy, or a hardware-friendly latency target.

## 2. PTQ vs QAT

| Method | When quantization happens | Training cost | Typical strength | Typical weakness |
| --- | --- | --- | --- | --- |
| PTQ | After full-precision training | Low | Fast to deploy, simple pipeline | Accuracy drops sharply below 4-bit |
| QAT | During training or fine-tuning | Higher | Much better low-bit adaptation | More compute, more tuning, harder to train |

### PTQ

PTQ usually works by calibrating ranges and then mapping weights/activations to a discrete grid. The model never learns under quantization noise, so PTQ is easy to use but fragile in the sub-4-bit regime.

In practice, PTQ is often good when:

- you want a quick compression pass,
- you can tolerate some accuracy loss,
- you are staying near 4-bit or above,
- or you mostly care about deployment simplicity.

### QAT

QAT inserts fake-quantization into training, often using a straight-through estimator so gradients can flow through discrete rounding decisions.

Intuitively, QAT teaches the network to live inside the quantized grid instead of fighting it. That makes it much better for ternary, 2-bit, and other extreme low-bit settings.

In practice, QAT is useful when:

- low-bit accuracy matters more than training simplicity,
- you want the model to recover from quantization error,
- or you are optimizing for an actual low-bit deployment target rather than a single compressed checkpoint.

## 3. What ParetoQ adds

ParetoQ is described in [ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](https://arxiv.org/abs/2502.02631) and the PyTorch write-up [ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization](https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/).

Its main contribution is a **unified framework** for comparing and improving quantization across:

- 1-bit,
- 1.58-bit,
- 2-bit,
- 3-bit,
- 4-bit.

ParetoQ studies two things together:

1. **Training schedule**: how much of the budget should remain full precision before QAT begins.
2. **Quantization function design**: which discretization rule works best at each bit width.

## 4. Technical findings

### 4.1 Training budget matters

ParetoQ shows that the best low-bit result usually does **not** come from QAT from scratch.

The reported pattern is:

- keep most of the budget in full-precision pretraining,
- reserve a smaller slice, roughly around 10%, for QAT fine-tuning,
- and let the quantized model adapt after the base representation is already strong.

Why this makes sense:

- full-precision training learns the broad semantic structure,
- QAT then learns how to preserve that structure inside a discrete grid,
- and too much early quantization can lock the model into a poor representation too soon.

### 4.2 There is a transition around 2-bit to 3-bit

ParetoQ reports a notable behavioral shift:

- **3-bit and above**: the fine-tuned model tends to stay closer to the original full-precision distribution.
- **2-bit and below**: the representation changes much more aggressively, so quantization is no longer a small perturbation.

This is why sub-3-bit quantization is qualitatively different, not just a smaller version of 4-bit.

### 4.3 Quantization function choice is critical

ParetoQ argues that low-bit results are extremely sensitive to the quantization grid and range design.

The paper combines different choices into one framework:

- **1-bit**: Elastic Binarization.
- **1.58-bit and 2-bit**: the proposed SEQ-style quantization grid.
- **3-bit and 4-bit**: LSQ-style quantization.

The high-level intuition is that the grid should match the bit regime:

- at very low bits, symmetry and balanced level spacing matter a lot,
- at higher bits, learnable scaling can absorb more of the distortion,
- and using the wrong grid can hide the true scaling-law behavior.

## 5. What the paper claims in practice

From the PyTorch blog and the repository README, ParetoQ reports that:

- the framework outperforms prior PTQ and QAT baselines in 2-bit, 3-bit, and 4-bit settings,
- ternary, 2-bit, and 3-bit models often land on a better accuracy-size frontier than 4-bit,
- and 2-bit is especially attractive when you care about memory reduction plus hardware speed.

The practical message is not "always choose the lowest bit". It is:

- for some models, 4-bit is not the best cost-performance point,
- ternary or 2-bit can sit on a better Pareto frontier,
- and a unified benchmark is needed before making strong claims about the best bit-width.

## 6. Mental model

If you want a compact way to remember the difference:

- **PTQ** answers: can I compress this trained model cheaply?
- **QAT** answers: can I retrain the model so it survives quantization?
- **ParetoQ** answers: across very low bits, which training and quantization choices actually dominate the accuracy-size tradeoff curve?

## 7. Practical takeaway

Use PTQ when you need a fast baseline and can tolerate some loss.

Use QAT when the target bit width is low enough that the model must adapt to the grid.

Use ParetoQ-style thinking when comparing 1-bit to 4-bit settings, because the training schedule and the quantization function are both part of the answer, not just the bit count.
