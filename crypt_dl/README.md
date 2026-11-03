# Cross-Architecture Neural Cryptography

Alice and Bob learn to communicate using a shared key while Eve tries to recover the plaintext from ciphertext alone. This research project tests whether the learned communication survives fresh attackers. It is not production cryptography.

Previous experiments, measurements, and figures are in [RESULTS.md](RESULTS.md) and [img/](img/). The setup builds on [Abadi and Andersen (2016)](https://arxiv.org/abs/1610.06918).

## CNN+MLP Eve experiment

[`scripts/hybrid_eve_mlp_study.py`](scripts/hybrid_eve_mlp_study.py) trains separate Alice–Bob pairs against a CNN Eve, an MLP Eve, or both together. After training, it freezes each pair and tests fresh **MLP attackers only** at roughly 5k and 80k parameters. The default study uses three pair seeds and three attacker seeds. A pair is attacked only if Bob reaches at least 99% bit accuracy on held-out validation and test messages.

The script appends configuration, pair, attack, and summary records to a JSONL file. Each condition gets four total Eve updates per Alice–Bob update. These are new baselines; earlier Week 3 runs used two Eve updates.

## Run with Colab CLI

Use the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) from Linux, macOS, or WSL on Windows. Run one condition and pair seed per session. This WSL example runs `hybrid` with seed `7` (on Linux or macOS, change the `cd` path):

```bash
cd /mnt/c/Users/kunjs/OneDrive/Projects/llms-from-scratch
mkdir -p crypt_dl/runs
colab run --gpu A100 --keep -s crypt-hybrid-7 --timeout 14400 \
  crypt_dl/scripts/hybrid_eve_mlp_study.py \
  --conditions hybrid --pair-seeds 7 \
  --output /content/crypt-hybrid-7.jsonl
colab download -s crypt-hybrid-7 \
  /content/crypt-hybrid-7.jsonl crypt_dl/runs/crypt-hybrid-7.jsonl
colab stop -s crypt-hybrid-7
```

Repeat for `cnn`, `mlp`, and `hybrid` with seeds `7`, `8`, and `9`, changing the session and output names. Download the JSONL before stopping the session, including if the run ended early. `--keep` leaves the runtime allocated until `colab stop`.
