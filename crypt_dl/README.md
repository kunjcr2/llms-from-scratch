# Cross-Architecture Neural Cryptography

This is an adversarial-neural-cryptography experiment based on the Alice–Bob–Eve setup. Alice and Bob are trained against a CNN Eve, then frozen. Fresh attacker architectures are trained from scratch on ciphertext produced by the frozen Alice.

The purpose is to test **attacker-architecture overfitting**: does a learned cipher only fool the CNN it trained against?

## Pilot results

Held-out Eve bit accuracy (50% is random guessing):

| Fresh attacker | 16-bit accuracy | 32-bit accuracy |
| --- | ---: | ---: |
| CNN | 57.9% | 54.8% |
| MLP | 83.5% | 60.3% |
| BiLSTM | 88.6% | 72.9% |
| Transformer | 50.0% | 49.9% |

The BiLSTM substantially recovers the plaintext at both lengths, so these CNN-trained systems are **not generally secure**. The Transformer result means only that this particular Transformer configuration did not learn an attack in that run; it is not evidence of security.

## 32-bit five-seed sweep

Alice and Bob were trained against a CNN Eve for 50,000 adversarial steps for each seed (7-11). Each frozen Alice/Bob pair was then attacked by fresh CNN, MLP, BiLSTM, and Transformer Eves trained for 20,000 steps. Every metric below is measured on newly generated held-out plaintext/ciphertext pairs.

| Model | Mean bit accuracy | Standard deviation | Mean Hamming error |
| --- | ---: | ---: | ---: |
| Fresh CNN Eve | 56.36% | 1.93% | 13.96 / 32 |
| Fresh MLP Eve | 59.93% | 6.06% | 12.82 / 32 |
| Fresh BiLSTM Eve | 66.52% | 17.16% | 10.71 / 32 |
| Fresh Transformer Eve | 75.34% | 23.57% | 7.89 / 32 |
| Bob | 100.00% | 0.00% | n/a |

Random guessing is 50% bit accuracy and 16 wrong bits per 32-bit message. Bob learned reliable communication, but the frozen learned systems were repeatedly broken by fresh non-CNN attackers. The Transformer had the strongest average attack, while its large standard deviation shows that the learned code and attack success are strongly seed-dependent.

This is evidence of **attacker-architecture overfitting**: training Alice and Bob against one CNN Eve does not create robust secrecy against other learned attackers. It is a pilot result; a formal comparison should capacity-match and tune all attacker families, then report confidence intervals over more seeds.

Run the notebook after changes with:

```powershell
python build_adversarial_neural_cryptography_notebook.py
```

This is a research evaluation, not production encryption and not a replacement for AES or TLS. The setup is inspired by [Abadi and Andersen (2016)](https://arxiv.org/abs/1610.06918).
