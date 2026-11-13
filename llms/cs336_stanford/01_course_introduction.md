# 1. Course Introduction

## Course premise

CS336 is organized around building language models from scratch to understand how their parts work together. “From scratch” is a teaching choice: a quarter is too short to recreate every dependency, so the course focuses on the components that provide the most learning value.

The guiding question is:

> What is the best model we can build with a fixed data and compute budget?

The goal is not only to make a model work. It is to understand the trade-offs among model quality, data, compute, memory, communication, and engineering effort.

## Why understand the stack?

Using a pretrained model or prompting it can be effective. The limitation is that each abstraction hides choices and failure modes. When a model does not do what is needed, understanding tokenization, architecture, training, and inference expands the set of possible explanations and fixes.

The course connects hands-on implementation with research. Building a model is used to expose the mechanics behind decisions that might otherwise look like unexplained settings in a paper or configuration file.

## What small-scale experiments teach

Small models are practical for learning and testing, but they do not always predict frontier behavior.

- The fraction of work spent in different model components can change with scale. The lecture gives the example of MLP compute growing from about 44% at smaller scale to about 80% for a 175-billion-parameter model.
- Some capabilities become visible only after a model reaches a larger scale. A small experiment may show little improvement even when the larger-scale behavior differs.
- Architecture and data choices that work on a small run may not be the best choices at scale.

Small runs are still valuable for understanding mechanics, checking correctness, comparing controlled variants, and estimating larger runs. Interpret empirical conclusions within the scale at which they were measured.

## Three kinds of knowledge

1. **Mechanics:** how Transformers, kernels, optimizers, and parallelism work.
2. **Engineering mindset:** profile and benchmark, track where resources go, and optimize the complete workload.
3. **Modeling intuition:** which data or architecture choices improve results.

Mechanics and resource-aware engineering transfer relatively well across scale. Modeling intuition is harder to transfer: many design choices must be tested empirically, and a result at one scale may not persist at another.

The “bitter lesson” is not that algorithms do not matter. The useful algorithms are those that continue to improve as available compute grows. A small efficiency gain can matter greatly when a training run is expensive.

## Course roadmap

The course is divided into five units, roughly corresponding to five assignments.

### 1. Basics

Build and train a language model. The basic pipeline includes:

- Converting raw text into token IDs.
- Defining a Transformer and its components.
- Choosing a training objective, optimizer, initialization, learning-rate schedule, and batch size.
- Running training and measuring model quality, including perplexity.
- Counting resources to understand where compute is spent.

The assignment is intended to connect these pieces into a working model rather than treat them as unrelated topics. Correctness checks allow components to be tested before a full training run.

### 2. Systems

Study how to get more useful work from accelerators:

- Estimate compute and memory requirements.
- Profile workloads and identify bottlenecks.
- Write or optimize GPU kernels, including with Triton.
- Distribute model state and computation across multiple GPUs.
- Understand inference, where a model is used to generate outputs.

The recurring systems principle is to reduce unnecessary data movement. Data transfer within a GPU or between GPUs can be as important as the arithmetic itself.

### 3. Scaling laws

At large budgets, it is usually impossible to try every model configuration. Instead, use smaller experiments to develop a training recipe and predict how it will behave at a larger compute budget.

A scaling recipe maps a compute budget to choices such as model size, token count, batch size, and learning rate. Small-run results are used to fit a scaling relationship and forecast larger runs. Good predictions require choices that behave consistently across scales; predictability matters along with finding a locally strong configuration.

The lecture gives a rough compute-optimal rule of thumb of about 20 training tokens per parameter. This is not universal: architecture, data, and inference constraints can change the preferred balance.

### 4. Data engineering

Data helps determine what a model learns and what it can do. The unit covers evaluation and the work needed to turn raw sources into training data:

- Convert sources such as web pages, PDFs, books, or code into usable text.
- Filter low-quality or irrelevant material.
- Remove duplicates.
- Combine sources into a data mixture.
- Consider synthetic, pretraining, mid-training, and post-training data.

Evaluation helps specify the capabilities being targeted. Internal measures such as perplexity can support model development, while external evaluations should reflect the uses for which a model is being presented. A single average score can hide differences across capabilities.

### 5. Alignment

Pretraining teaches next-token prediction. Alignment can use feedback about which outputs are better:

1. Generate candidate responses.
2. Score them with people, verifiers, or a language-model judge.
3. Update the model to favor better responses.

This can use preference optimization such as DPO or reinforcement-learning methods such as PPO and GRPO. Reinforcement learning can be hard to tune and adds a systems problem: inference workers generate examples while training workers update the model. Delayed or stale generations can reduce how closely training data matches the current policy.

## Efficiency as the common thread

Efficiency is not limited to GPU kernels. It appears throughout the course:

- **Tokenization:** shorter sequences reduce work for the model.
- **Architecture:** changes may reduce arithmetic, memory, or inference cost.
- **Data filtering:** compute spent on redundant or poor data cannot be spent on useful data.
- **Scaling laws:** smaller experiments can guide decisions about expensive runs.
- **Systems:** kernels and parallelism aim to use hardware more effectively.

The objective is best performance under the resources actually available, not maximum model size in isolation.

## Language-model history in brief

Language models predate modern neural networks. N-gram models were used in applications such as speech recognition and machine translation. Neural language modeling developed through several stages:

- Recurrent models such as LSTMs.
- Feedforward neural language models with limited context.
- Sequence-to-sequence models and attention for translation.
- Transformers, followed by larger models and systems techniques such as mixture of experts.
- ELMo and BERT, which were pretrained and then fine-tuned for tasks.
- GPT-style models, which helped popularize scaling and prompting as a general interface.

Open-weight models and more detailed reports have made it easier to study modern systems. However, many training details, especially data mixtures, remain unavailable, so complete reproduction can still be difficult.

The interface to language models has shifted from fine-tuning, to prompting, to conversation, and increasingly to agent-like workflows. The course emphasizes that many underlying foundations remain recognizable: GPUs, kernels, gradient-based optimization, and Transformer attention. Longer contexts increase the importance of inference efficiency.

## Tokenization: the basic problem

Text is represented as Unicode strings, while a language model processes sequences of token IDs. A tokenizer provides both directions:

- **Encode:** convert text into token IDs.
- **Decode:** convert token IDs back into text.

A basic correctness property is round-tripping: encoding and then decoding text should recover the original string.

Tokenization affects sequence length, vocabulary size, and how much computation the model performs. Common tokenizers may treat a word with a preceding space as a different token from the same word without that space. Numbers may be split into digits or groups of digits, depending on the tokenizer.

## Why not tokenize only by characters, bytes, or words?

- **Characters:** Unicode has many possible characters, and many are rare. A large sparse vocabulary can be inefficient, while text may still require many tokens.
- **Bytes:** the vocabulary is only 256 values, but each byte becomes a token. This gives long sequences and a compression ratio of one byte per token.
- **Words:** common words are meaningful units, but the vocabulary can grow without bound. Unseen words require an unknown token or another fallback, which loses information.

These approaches expose the trade-off: a small vocabulary can make sequences long; a large vocabulary can be sparse and expensive.

## Byte Pair Encoding

Byte Pair Encoding (BPE) builds a vocabulary from the training text. It starts with bytes and repeatedly merges the most frequent adjacent pair.

1. Encode the training corpus as bytes; each byte initially represents a token.
2. Count adjacent token pairs.
3. Select a frequent pair and create a new token for it.
4. Replace its occurrences and repeat until reaching the desired vocabulary size.

As merges accumulate, the token sequence becomes shorter and the vocabulary grows. Common byte sequences can become single tokens. Rare sequences can still be represented by smaller pieces, avoiding a single unknown token for every unseen word.

BPE is a data-driven heuristic, not a guarantee that every token is linguistically meaningful. Its practical value is that it gives a finite vocabulary while still representing arbitrary input.

## Tokenizer implementation details

- A simple encoder that checks every merge in order can be very slow when the vocabulary is large. Efficient implementations index the merges that can apply.
- Text is often split into chunks before applying BPE, which makes processing faster and respects tokenizer rules.
- Special tokens need explicit handling in a modern tokenizer.
- The decoder must reconstruct the original byte sequence and then the original text.
- Compression ratio can be measured as bytes of text divided by number of tokens. A higher ratio means fewer tokens for the same text, but increasing vocabulary size also has costs.

The course presents tokenization because current frontier models still use it. A future end-to-end byte-level approach would still need to build useful abstractions and allow variable-sized chunks so that different parts of the input can receive different amounts of computation.
