# Cross-Architecture Neural Cryptography

This folder studies whether Alice and Bob can learn a keyed communication scheme that remains difficult for a newly trained attacker to decode. Alice receives a binary message and key and emits a ciphertext; Bob receives the ciphertext and key and reconstructs the message; Eve receives only the ciphertext and tries to recover the message. The project is an experiment in learned communication and adversarial training, not a production cryptographic construction.

The main question is: **how do Eve's architecture, capacity, optimization time, and training setup affect the plaintext leakage of a frozen Alice-Bob pair?** The evaluation follows the fresh-attacker idea in [Abadi and Andersen (2016)](https://arxiv.org/abs/1610.06918): after Stage 1, Alice and Bob are frozen and a new attacker is trained from scratch.

## Objective and protocol

Experiments use 32-bit (with earlier 16- and 64-bit pilots) binary messages and keys.

1. **Stage 1:** train Alice and Bob to communicate while an Eve model is trained from ciphertext alone. Depending on the experiment, Eve is a CNN, MLP, Transformer, or a CNN+MLP pair.
2. **Freeze:** keep Alice and Bob fixed after training.
3. **Stage 2:** train fresh attackers from new random initializations using ciphertext only. The attacker never receives the key.
4. **Evaluate:** measure Bob and attacker performance on held-out messages, keys, and (where specified) fresh random streams.

An attacker above the 50% bit-accuracy baseline has recovered some plaintext information. A high attacker score is evidence that the learned scheme is not robust to that attacker; it is not a formal cryptanalytic security result.

## Architectures

- **Alice and Bob:** primarily `CryptCNN` models, approximately 40,609 parameters each in the Week 3-4 studies.
- **Stage-1 Eve:** ciphertext-only CNN, MLP, Transformer, or separate CNN and MLP Eves in the hybrid condition.
- **Fresh Stage-2 attackers:** CNN, MLP, bidirectional LSTM, Transformer, and, in a pilot, XGBoost bit classifiers.
- **Capacity studies:** fresh neural attackers are targeted at approximately 5k, 10k, 20k, 40k, and 80k parameters. The actual counts are recorded in `RESULTS.md` because they differ by architecture.

The hybrid condition uses two separate Eves. Each gets two updates per Alice-Bob update, for four total Eve updates; Alice is trained against the better-performing Eve on that batch. This is not a single network that combines CNN and MLP layers.

## Experimental setup and metrics

The exact configuration varies by study and is recorded with the results. Common settings in the repeated studies are:

- 32-bit messages and keys, freshly sampled during training and evaluation.
- A frozen Alice-Bob pair evaluated before accepting attacker results. The project uses 96% Bob bit accuracy as its communication criterion; some older notebooks/runs use a 95% or 99% gate, so the recorded run configuration matters.
- Fresh attackers initialized with seeds 101, 102, and 103 in the repeated size studies.
- Attacker checkpoints at 5,000 and/or 20,000 updates, with held-out validation used for checkpoint selection where stated.
- A 50% bit-accuracy chance baseline for binary plaintext bits.

Reported metrics are:

- **Bob bit accuracy:** fraction of message bits Bob reconstructs correctly.
- **Bob full-message accuracy:** fraction of complete messages reconstructed with every bit correct.
- **Attacker bit accuracy:** plaintext bits recovered by a fresh attacker from ciphertext alone.
- **Attacker full-message accuracy:** complete plaintext messages recovered exactly.
- **Variation:** standard deviation across attacker initializations or pair seeds, explicitly labeled in the tables.

Bit and full-message accuracy must be read together: for example, 96.08% correct bits can still correspond to only 28.22% completely correct 32-bit messages.

## Completed and recorded experiments

The detailed per-run data, limitations, and figures are in [`RESULTS.md`](RESULTS.md). The following experiments have recorded outputs; “completed” does not mean every planned replication was completed.

| Study | Recorded result | Status and limitation |
| --- | --- | --- |
| CNN-trained pilots and five-seed 32-bit sweep | Fresh non-CNN attackers, especially BiLSTM and Transformer, often decoded above chance after CNN-Eve training. | Completed pilot baseline; attacker capacities were not matched. |
| Capacity-matched architecture sweep | A single-seed 4x4 comparison of CNN, MLP, BiLSTM, and Transformer attackers. | Completed single sweep; needs multi-seed replication. |
| 64-bit ResNet-18 Alice-Bob pilot | Longer Alice-Bob training reduced some recorded attack scores but did not remove leakage. | Completed pilot; not a causal or general architecture comparison. |
| Single-pair attacker-size sweeps | At 5,000 attacker updates, size effects were mostly flat; with 20,000 updates, MLP leakage increased substantially with capacity on one frozen pair. | Recorded; the interrupted run used a pair with only 87.50% Bob bit accuracy and is not a valid successful-communication result. |
| Repeated size study after MLP-Eve Stage 1 | On a reliable pair, MLP showed a strong capacity effect, Transformer a smaller non-monotonic effect, CNN was approximately flat, and BiLSTM rose modestly through 40k. | 57 of 60 conditions completed; the 80k BiLSTM conditions were missing. One pair seed only. |
| Repeated size study after CNN-Eve Stage 1 | Fresh MLPs rose from 62.11% at 5k to 80.89% at 80k bits and recovered some complete messages. | All recorded attackers completed; one pair seed only. |
| Repeated size study after Transformer-Eve Stage 1 | Fresh MLPs rose from 55.60% at 5k to 71.10% at 80k bits; initialization variability was substantial. | All recorded attackers completed; Bob passed the project's 96% criterion but was below the older 99% gate. |
| Three-seed CNN/MLP/hybrid comparison | Fresh 80k MLPs beat 5k MLPs on every successful MLP-only and hybrid pair. The hybrid mean was 67.49% versus 68.85% for MLP-only; this does not show a privacy advantage. | Seeds 7-9 recorded. CNN-only Bob communication failed on two seeds; the successful seed was fully broken by fresh MLPs. |
| XGBoost follow-up | Across MLP-Eve pairs, XGBoost recovered 52.58-53.95% bits and a same-data fresh MLP recovered 52.96-54.37%; neither recovered a complete message. | Completed small pilot, not matched to the larger 20,000-update attacker studies. Checkpoints are saved for larger follow-up attacks. |

The defensible overall finding is that fresh attackers can recover plaintext information from these learned schemes, and that attacker capacity can matter substantially after sufficient optimization, especially for MLP attackers. The data do not establish a universal architecture ranking, a monotonic size law for every attacker family, or cryptographic secrecy. Stage-1 Eve choice also interacts with communication reliability and leakage, and the three-seed hybrid comparison does not show a consistent benefit from adding a CNN Eve to an MLP Eve.

## Ongoing or not-yet-completed work

These are open follow-ups identified by the recorded notes, not completed results:

- Repeat the Stage-1 Eve and fresh-attacker comparisons across more independently trained, reliable Alice-Bob pairs.
- Finish or replace the missing 80k BiLSTM conditions and report them separately from the completed 57-condition study.
- Test whether a stronger 80k MLP Eve during Stage 1 reduces leakage.
- Investigate the anomalous CNN-only seed-9 run by saving checkpoints and ciphertext samples, testing key dependence, and retraining a fresh CNN Eve against the frozen pair.
- Use the saved Alice-Bob checkpoints for larger attacks after the small XGBoost follow-up.

Until these are run, avoid describing the current observations as a general security claim or as proof that one Stage-1 architecture is intrinsically better.

## Reproduction and file map

- [`RESULTS.md`](RESULTS.md): chronological experiment record, tables, figures, caveats, and current interpretation.
- [`scripts/hybrid_eve_mlp_study.ipynb`](scripts/hybrid_eve_mlp_study.ipynb): three-condition CNN, MLP, and hybrid Stage-1 study with fresh MLP attacks.
- [`scripts/attacker_size_repeated_a100.ipynb`](scripts/attacker_size_repeated_a100.ipynb): repeated fresh-attacker size study on an A100.
- [`scripts/mlp_eve_xgboost_followup.ipynb`](scripts/mlp_eve_xgboost_followup.ipynb): MLP-Eve pairs attacked with XGBoost and a fresh MLP.
- Other notebooks in `scripts/`: early pilots, capacity sweeps, ResNet-18 experiments, and follow-ups.
- `results/`: checked-in JSONL/JSON measurements. `img/`: plots referenced by `RESULTS.md`.
- [`POST_WK4.md`](POST_WK4.md): the Week 4 research update and planned next steps.

For Colab runs, set the condition, pair seed, run name, and `BOB_MIN` explicitly before executing a notebook top to bottom. Keep batch size, update budget, attacker seeds, held-out split, and the communication gate consistent when comparing conditions. The checked-in hybrid notebook defaults to a 99% gate, while the Week 4 recorded runs used `BOB_MIN = 0.95` and were interpreted with the project's 96% criterion; do not silently mix these settings.
