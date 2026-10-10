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

Run the notebook after changes with:

```powershell
python build_adversarial_neural_cryptography_notebook.py
```

This is a research evaluation, not production encryption and not a replacement for AES or TLS. The setup is inspired by [Abadi and Andersen (2016)](https://arxiv.org/abs/1610.06918).
