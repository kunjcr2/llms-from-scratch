# Cross-Architecture Neural Cryptography

This is an adversarial-neural-cryptography experiment based on the Alice–Bob–Eve setup. Alice and Bob are trained against a CNN Eve, then frozen. Fresh attacker architectures are trained from scratch on ciphertext produced by the frozen Alice.

The purpose is to test **attacker-architecture overfitting**: does a learned cipher only fool the CNN it trained against?

## Results and figure naming

The complete experimental record is in [RESULTS.md](RESULTS.md). Attacker-size figures use the convention `week<week>_attacker_size_stage1_<eve>[_partial]_seed<seed>_<updates>_updates.svg`, where `stage1_<eve>` identifies the Eve used while Alice and Bob were synchronized, and `partial` marks an interrupted study.

| Figure | Study |
| --- | --- |
| `week2_attacker_size_stage1_cnn_seed7_5k_updates.svg` | Single-run sweep after Stage-1 CNN-Eve training. |
| `week2_attacker_size_stage1_mlp_partial_seed7_20k_updates.svg` | Partial repeated sweep after Stage-1 MLP-Eve training; Bob was unreliable. |
| `week3_attacker_size_stage1_mlp_seed7_20k_updates.svg` | Reliable repeated sweep after Stage-1 MLP-Eve training. |
| `week3_attacker_size_stage1_cnn_seed7_20k_updates.svg` | Reliable repeated sweep after Stage-1 CNN-Eve training. |

## Preliminary, unmatched-capacity pilot results

Held-out Eve bit accuracy (50% is random guessing):

| Fresh attacker | 16-bit accuracy | 32-bit accuracy |
| --- | ---: | ---: |
| CNN | 57.9% | 54.8% |
| MLP | 83.5% | 60.3% |
| BiLSTM | 88.6% | 72.9% |
| Transformer | 50.0% | 49.9% |

The BiLSTM substantially recovers the plaintext at both lengths, so these CNN-trained systems are **not generally secure**. The Transformer result means only that this particular Transformer configuration did not learn an attack in that run; it is not evidence of security.

## 32-bit five-seed sweep (pre-matching baseline)

Alice and Bob were trained against a CNN Eve for 50,000 adversarial steps for each seed (7-11). Each frozen Alice/Bob pair was then attacked by fresh CNN, MLP, BiLSTM, and Transformer Eves trained for 20,000 steps. Every metric below is measured on newly generated held-out plaintext/ciphertext pairs.

| Model | Mean bit accuracy | Standard deviation | Mean Hamming error |
| --- | ---: | ---: | ---: |
| Fresh CNN Eve | 56.36% | 1.93% | 13.96 / 32 |
| Fresh MLP Eve | 59.93% | 6.06% | 12.82 / 32 |
| Fresh BiLSTM Eve | 66.52% | 17.16% | 10.71 / 32 |
| Fresh Transformer Eve | 75.34% | 23.57% | 7.89 / 32 |
| Bob | 100.00% | 0.00% | n/a |

Random guessing is 50% bit accuracy and 16 wrong bits per 32-bit message. Bob learned reliable communication, but the frozen learned systems were repeatedly broken by fresh non-CNN attackers. The Transformer had the strongest average attack, while its large standard deviation shows that the learned code and attack success are strongly seed-dependent.

This is evidence of **attacker-architecture overfitting**: training Alice and Bob against one CNN Eve does not create robust secrecy against other learned attackers. These initial numbers used unmatched attacker capacities, so they are a pre-matching baseline rather than the final comparison.

The current notebook now uses a capacity-matched post-hoc suite: CNN (23,073 parameters), MLP (24,864), BiLSTM (19,441), and Transformer (18,209). It keeps the same attack data, optimizer, and step budget for all four attackers. Rerun the five-seed sweep before treating cross-architecture differences as final.

## Current work: full 4x4 Eve-architecture sweep

The new [`adversarial_neural_cryptography_4x4_sweep.ipynb`](adversarial_neural_cryptography_4x4_sweep.ipynb) is a standalone Google Colab experiment. It uses the same larger 19,409-parameter CNN for Alice and Bob, and varies Eve in both roles:

| Role | Architectures |
| --- | --- |
| Eve used while training Alice/Bob | CNN, MLP, BiLSTM, Transformer |
| Fresh post-hoc attacker | CNN, MLP, BiLSTM, Transformer |

This produces a 4x4 matrix: each row is the Eve architecture Alice/Bob learned against, and each column is the architecture of a new attacker trained against that frozen pair. The main question is whether a learned communication scheme that withstands one Eve family is still vulnerable to a different family.

The Eve architectures range from roughly 18k to 25k parameters, so they are in the same capacity range as Alice and Bob. The notebook is optimized for a Colab T4 or L4 GPU with mixed precision and fixed-shape CUDA optimizations. It uses one seed per run (`42` or `7`), 20,000 adversarial-training steps per training-Eve row, and 8,000 steps per attacker cell. Alice/Bob receive two in-game Eve updates per step and an elevated privacy-loss weight to make hiding the plaintext a first-class optimization goal. It prints progress, the held-out result tables, and two graphs inline; it does not save checkpoints, plots, CSVs, or other result files.

In Colab, choose a T4/L4 GPU, set `SEED` in the first code cell, then run its two code cells in order. If a T4 runs out of memory, lower `BATCH_SIZE` from `8192` to `4096`.

Run the notebook after changes with:

```powershell
python build_adversarial_neural_cryptography_notebook.py
```

This is a research evaluation, not production encryption and not a replacement for AES or TLS. The setup is inspired by [Abadi and Andersen (2016)](https://arxiv.org/abs/1610.06918).

## ResNet-18 Alice/Bob transfer-learning experiment

[`scripts/resnet_alice_bob_experiment.ipynb`](scripts/resnet_alice_bob_experiment.ipynb) is a self-contained Google Colab notebook testing the professor's pretrained-model suggestion. Open the notebook in Colab, select a GPU runtime, and run its cells in order. No companion Python file is required.

The notebook uses the genuine `torchvision` ResNet-18 initialized from official ImageNet pretrained weights; it does not train a new ResNet from scratch. For 64-bit cryptography, Alice receives a 64-bit message plus a 64-bit key, while Bob receives a 64-value ciphertext plus the 64-bit key. These 128-value inputs become 8x16 three-channel images resized to ResNet's standard 224x224 input. The pretrained stem through `layer3` remains frozen, and only `layer4` plus the new 64-value output head are fine-tuned. Alice and Bob train against a CNN Eve, then the frozen pair is attacked independently by fresh CNN, MLP, BiLSTM, Transformer, and Extreme Learning Machine (ELM) attackers.

Nothing is written to Colab's temporary filesystem. Training progress is printed both readably and as `LOG_JSON` records. Each held-out result is printed as `RESULT_JSON`, followed by a complete results table, a one-row attack matrix, and a heatmap. Saving the executed Colab notebook preserves all of these outputs.

An A100 is recommended for the fastest run, although an L4 is fully supported. The notebook detects the accelerator and selects batch size 256 on A100, 128 on L4, and 64 on other CUDA GPUs. A100 and L4 runs use BF16 mixed precision, TF32 matrix operations, and channels-last ResNet tensors. Held-out evaluation is chunked by the selected batch size to avoid GPU-memory spikes.
