# Cross-Architecture Neural Cryptography

Alice and Bob learn to communicate using a shared key while Eve tries to recover the plaintext from ciphertext alone. This research project tests whether the learned communication survives fresh attackers. It is not production cryptography.

Previous experiments, measurements, and figures are in [RESULTS.md](RESULTS.md) and [img/](img/). The setup builds on [Abadi and Andersen (2016)](https://arxiv.org/abs/1610.06918).

## CNN+MLP Eve experiment

[`scripts/hybrid_eve_mlp_study.ipynb`](scripts/hybrid_eve_mlp_study.ipynb) trains separate Alice–Bob pairs against a CNN Eve, an MLP Eve, or both together. After training, it freezes each pair and tests fresh **MLP attackers only** at roughly 5k and 80k parameters. Each run uses attacker seeds 101, 102, and 103. The notebook attacks a pair only if Bob meets `BOB_MIN` on held-out validation and test messages; the notebook default is 99%.

The notebook appends configuration, pair, attack, and summary records to a JSONL file. Each condition gets four total Eve updates per Alice–Bob update, split two per Eve in the hybrid condition. These are new baselines; earlier Week 3 runs used two Eve updates.

### Seed 7 results

[`results/hybrid_seed7.jsonl`](results/hybrid_seed7.jsonl) combines the hybrid run and the CNN/MLP run. Both runs set `BOB_MIN = 0.95`. The table gives held-out test accuracy; attacker values are means across three fresh starts after 20,000 updates.

| Stage-1 Eve | Bob bit accuracy | Bob full-message accuracy | Fresh 5k MLP bit accuracy | Fresh 80k MLP bit accuracy |
| --- | ---: | ---: | ---: | ---: |
| Hybrid CNN + MLP | 96.08% | 28.22% | 56.94% | 67.39% |
| CNN | 50.01% | 0% | skipped | skipped |
| MLP | 99.98% | 99.53% | 56.42% | 67.90% |

The CNN pair did not learn reliable communication and therefore was not attacked. The hybrid pair meets the project's 96% bit-accuracy criterion but falls short of the notebook's 99% default and has much lower full-message accuracy than the MLP pair. These are one-pair-per-condition results; repeat with additional pair seeds before drawing a general comparison.

## Run in Google Colab

Open the notebook in Colab and select an A100 or L4 GPU runtime. In its first code cell, set `CONDITIONS` to `("cnn",)`, `("mlp",)`, or `("hybrid",)` and `PAIR_SEEDS` to one seed. Set `BOB_MIN` explicitly and keep it consistent across comparable runs. Change `RUN_NAME` for each run, then run the cells from top to bottom. The notebook starts with `("hybrid",)` and seed `7` as an example.

Results are appended to `/content/<RUN_NAME>.jsonl`. Use the last cell to download them, including after an interruption while the runtime is still connected. The first cell selects batch size 16,384 on A100 and 8,192 on L4; keep it consistent across the conditions you compare.
