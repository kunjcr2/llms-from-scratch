# Results Record: Cross-Architecture Neural Cryptography

## What was measured

Alice encrypts a binary plaintext using a shared binary key, Bob reconstructs it using the ciphertext and key, and Eve/Trudy tries to reconstruct it from ciphertext alone. Results below report held-out attacker bit accuracy; 50% is random guessing. Higher attacker accuracy means more plaintext information was recovered.

These are research experiments, not production cryptography or a replacement for AES/TLS.

## 1. Preliminary CNN-trained pilots

### Setup

- Alice and Bob: standard CNN / mix-and-transform architecture.
- Training attacker: CNN Eve.
- Fresh attackers after Alice/Bob were frozen: CNN, MLP, BiLSTM, and Transformer.
- Message lengths: 16 and 32 bits.
- These attackers were not capacity-matched, so this is a pilot baseline only.

| Fresh attacker | 16-bit accuracy | 32-bit accuracy |
| --- | ---: | ---: |
| CNN | 57.9% | 54.8% |
| MLP | 83.5% | 60.3% |
| BiLSTM | 88.6% | 72.9% |
| Transformer | 50.0% | 49.9% |

### Observation

The BiLSTM recovered substantial information at both message lengths. The Transformer result only means that this particular configuration did not learn an attack in that run; it is not evidence that the learned cipher was secure.

## 2. Five-seed 32-bit CNN-trained sweep

### Setup

- Alice and Bob: CNN / mix-and-transform architecture.
- Training attacker: CNN Eve.
- Message/key/ciphertext length: 32 bits.
- Alice/Bob adversarial training: 50,000 steps for each seed (7-11).
- Fresh attackers: CNN, MLP, BiLSTM, and Transformer, each trained from scratch for 20,000 steps.
- Evaluation: newly generated held-out plaintext/ciphertext pairs.
- Important limitation: attacker capacities were not yet matched.

| Model | Mean bit accuracy | Standard deviation | Mean Hamming error |
| --- | ---: | ---: | ---: |
| Fresh CNN Eve | 56.36% | 1.93% | 13.96 / 32 |
| Fresh MLP Eve | 59.93% | 6.06% | 12.82 / 32 |
| Fresh BiLSTM Eve | 66.52% | 17.16% | 10.71 / 32 |
| Fresh Transformer Eve | 75.34% | 23.57% | 7.89 / 32 |
| Bob | 100.00% | 0.00% | n/a |

### Observation

Bob communicated reliably, but fresh non-CNN attackers repeatedly decoded above chance. The strongest average attacker was the Transformer, although its large standard deviation shows that success was strongly seed-dependent. This is evidence that training only against one CNN Eve did not create a robust scheme against other learned attackers.

## 3. Capacity-matched 32-bit 4x4 architecture sweep

### Setup

- Alice and Bob: 19,409-parameter CNNs.
- Message/key/ciphertext length: 32 bits.
- Eve architectures in both roles: CNN, MLP, BiLSTM, Transformer.
- Training Eve: one architecture used while adversarially training Alice/Bob.
- Post-hoc Eve: a fresh attacker architecture trained against each frozen Alice/Bob pair.
- Approximate attacker parameter counts: CNN 23,073; MLP 24,864; BiLSTM 19,441; Transformer 18,209.
- One sweep shown below; 20,000 Alice/Bob training steps per row and 8,000 fresh-attacker steps per cell.
- The current notebook supports seed 42 or 7, but multi-seed repetition has not yet been completed for this matrix.

| Training Eve \\ fresh attacker | CNN | MLP | BiLSTM | Transformer |
| --- | ---: | ---: | ---: | ---: |
| CNN | 54.23% | 63.35% | 50.93% | 53.37% |
| MLP | 52.82% | 57.65% | 51.33% | 52.30% |
| BiLSTM | 59.00% | 60.75% | 57.00% | 58.46% |
| Transformer | 54.49% | 56.30% | 51.29% | 53.47% |

### Observation

Most cells are close to chance, but the MLP reaches 63.35% against the pair trained with a CNN attacker. This suggests that attacker architecture can matter even when the models are in a similar parameter range. Because this is currently a single-seed result, it needs repeated runs before making a strong architecture-level claim.

## 4. Pretrained ResNet-18 Alice/Bob, 64-bit experiments

### Setup

- Alice and Bob: genuine ImageNet-pretrained ResNet-18 models, not ResNets trained from scratch.
- Input: 64-bit message plus 64-bit key, reshaped into an 8x16 three-channel image and resized to 224x224.
- Fine-tuning: pretrained stem through `layer3` frozen; `layer4` and a new 64-value output head trainable.
- Training attacker: CNN Eve.
- Fresh attackers: CNN, MLP, BiLSTM, Transformer, and Extreme Learning Machine (ELM).
- Evaluation: held-out 64-bit plaintext/ciphertext pairs.
- Key limitation: this changes both the architecture size and the input representation, so it cannot isolate the effect of size alone.

| Fresh attacker | Earlier run | Longer Alice/Bob training run |
| --- | ---: | ---: |
| CNN | 71.8% | 64.7% |
| MLP | 70.4% | 64.2% |
| BiLSTM | 52.8% | 50.0% |
| Transformer | 65.9% | 63.2% |
| ELM | 63.6% | 64.2% |

### Observation

Several fresh attackers remained clearly above chance in both runs. Longer Alice/Bob training reduced the CNN, MLP, and Transformer attack scores in these recorded runs, but did not remove the vulnerability. This is a preliminary comparison only: it does not show that larger attackers are consistently stronger, or that longer training alone caused the change.

## Current interpretation and next controlled comparison

The completed results show that Bob can learn reliable communication while fresh attackers with different architectures can still recover information. The most useful next experiment is a controlled multi-seed study that changes one factor at a time:

1. attacker capacity;
2. attacker architecture; and
3. Alice/Bob architecture or input representation.

This will help separate the effect of model size from the effect of architecture, training duration, and the ResNet image-shaped representation.

## Reference

Abadi and Andersen's evaluation also resets and retrains Eve after Alice and Bob are trained; they treat a fresh Eve with a substantial advantage as evidence that a learned scheme is non-robust. Their paper reports experiments at 16, 32, and 64 bits, but does not establish that a larger attacker is always stronger: [Learning to Protect Communications with Adversarial Neural Cryptography (2016)](https://arxiv.org/abs/1610.06918).
