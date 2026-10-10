import json


nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}


def md(text):
    # The source file intentionally stays ASCII-friendly; normalize copied typographic glyphs.
    text = (text.replace("â€“", "-").replace("â€”", "-").replace("â€™", "'")
                .replace("â†“", chr(0x2193)).replace("â†™", chr(0x2199)).replace("â†˜", chr(0x2198)))
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})


def code(text):
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(keepends=True)})


md(r'''
# Adversarial Neural Cryptography (PyTorch)

This small, self-contained notebook teaches the basic *Alice–Bob–Eve* experiment. Alice learns to turn a binary message and a secret binary key into a continuous-valued ciphertext. Bob receives that ciphertext plus the same key and learns to reconstruct the message. Eve sees only the ciphertext and tries to recover the message.

This is an educational adversarial-learning demonstration—not a secure cipher and not a replacement for established cryptography such as AES.
''')

md(r'''
## 1. What adversarial neural cryptography is

The central idea is to train a communicating pair against an eavesdropper. Alice and Bob want reliable communication, while Eve wants to decode without the key. Their opposing objectives make this a three-player optimization problem.

```text
Plaintext + Secret Key
          ↓
        Alice
          ↓
      Ciphertext
       ↙       ↘
Bob + Key       Eve
   ↓             ↓
Plaintext      Guess
```

## 2. Alice, Bob, and Eve roles

* **Alice** receives a 16-bit plaintext and a 16-bit secret key, and emits a 16-number ciphertext.
* **Bob** receives the ciphertext and the key, so he can learn to reconstruct the plaintext.
* **Eve** receives only ciphertext. Omitting the key is essential: otherwise this would not model an eavesdropper.
''')

md('## 3. Imports and reproducibility')
code(r'''
# Only standard teaching dependencies: PyTorch, NumPy, and Matplotlib.
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

# Fixed seeds make the initialization and random training batches reproducible.
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Select the best accelerator available, while remaining fully CPU compatible.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
''')

md('## 4. Configuration')
code(r'''
MESSAGE_BITS = 32
KEY_BITS = 32
CIPHER_BITS = 32

# These match the scale of the original paper rather than the earlier MLP demo.
# A T4/L4/A100 handles this small Conv1D model comfortably.  Lower BATCH_SIZE to
# 1024 if Colab reports an out-of-memory error.
BATCH_SIZE = 4096
TRAIN_STEPS = 50_000
EVE_UPDATES = 2           # Paper: one Alice/Bob update, then two Eve updates.
POSTHOC_EVE_RESTARTS = 3  # The paper uses several independently reset attackers.
FRESH_EVE_STEPS = 20_000
EVAL_SIZE = 16_384
LOG_EVERY = 100

LR_ALICE_BOB = 8e-4
LR_EVE = 1e-3
RANDOM_ERROR = MESSAGE_BITS / 2  # Expected absolute bit error for a random guess.

assert MESSAGE_BITS == KEY_BITS == CIPHER_BITS
print(f"Message/key/ciphertext width: {MESSAGE_BITS}; random-error target: {RANDOM_ERROR:.1f} bits")
''')

md('## 5. Random message and key generation')
code(r'''
def random_batch(batch_size):
    """Return independent balanced -1/+1 bit tensors, each shaped [batch_size, 16]."""
    message = 2 * torch.randint(0, 2, (batch_size, MESSAGE_BITS), device=device).float() - 1
    key = 2 * torch.randint(0, 2, (batch_size, KEY_BITS), device=device).float() - 1
    assert message.shape == (batch_size, MESSAGE_BITS)
    assert key.shape == (batch_size, KEY_BITS)
    return message, key

message, key = random_batch(3)
print("message shape:", tuple(message.shape), "key shape:", tuple(key.shape))
''')

md('## 6–8. Alice, Bob, and Eve models')
code(r'''
class MixAndTransform(nn.Module):
    """Paper-faithful 1-D 'mix & transform' network for Alice, Bob, or Eve.

    A dense layer first lets the model learn *which* input positions should interact.
    Four Conv1D layers then transform those learned local mixtures.  The layer specs
    are [kernel, input channels, output channels] = [4,1,2], [2,2,4], [1,4,4],
    [1,4,1], with strides 1, 2, 1, 1, as reported by Abadi & Andersen (2016).
    """
    def __init__(self, input_bits):
        super().__init__()
        self.mix = nn.Linear(input_bits, 2 * MESSAGE_BITS)
        self.conv1 = nn.Conv1d(1, 2, kernel_size=4, stride=1, padding="same")
        self.conv2 = nn.Conv1d(2, 4, kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(4, 4, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(4, 1, kernel_size=1, stride=1)

    def forward(self, x):
        # [B, input_bits] -> [B, 32] -> [B, 1, 32].  Conv2 halves length to 16.
        x = self.mix(x).unsqueeze(1)
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        x = torch.tanh(self.conv4(x))       # continuous ciphertext/bit estimate in [-1, 1]
        return x.squeeze(1)                 # [B, 16]


# Alice and Bob each receive 32 values. Eve receives only 16 ciphertext values.
alice = MixAndTransform(MESSAGE_BITS + KEY_BITS).to(device)
bob = MixAndTransform(CIPHER_BITS + KEY_BITS).to(device)
eve = MixAndTransform(CIPHER_BITS).to(device)

with torch.no_grad():
    m, k = random_batch(4)
    c = alice(torch.cat([m, k], dim=1))
    assert c.shape == (4, CIPHER_BITS)
    assert bob(torch.cat([c, k], dim=1)).shape == (4, MESSAGE_BITS)
    assert eve(c).shape == (4, MESSAGE_BITS)
print("All CNN shape checks passed.")
''')

md(r'''
## 9. Loss functions

This notebook follows the paper's L1 reconstruction-error objective. Inputs and
outputs use -1/+1 bits; for interpretation, we map an output \(z\in[-1,1]\) to
\((z+1)/2\in[0,1]\). The per-message reconstruction error is the sum of absolute
bit errors, so a random balanced guess has expected error \(N/2=8\) for \(N=16\).

\[
L_{Bob}=\left\|\frac{Bob(Alice(m,k),k)+1}{2}-\frac{m+1}{2}\right\|_1,\qquad
L_{Eve}=\left\|\frac{Eve(c)+1}{2}-\frac{m+1}{2}\right\|_1.
\]

For Alice and Bob, the differentiable adversarial objective is

\[
L_{AB}=L_{Bob}+\frac{(N/2-L_{Eve})^2}{(N/2)^2}.
\]

The squared term pushes Eve's *differentiable reconstruction error* toward random guessing rather than merely making Eve consistently wrong (which Eve could undo by flipping every bit). We never use thresholded accuracy as a loss because thresholding is not differentiable.
''')
code(r'''
def finite(*values):
    """Fail loudly rather than silently continuing if adversarial training becomes unstable."""
    assert all(torch.isfinite(value).all().item() for value in values), "NaN or infinity detected"

def bit_accuracy(outputs, targets):
    """The sign boundary at zero maps continuous [-1,1] outputs back to -1/+1 bits."""
    predicted_bits = torch.where(outputs >= 0, 1.0, -1.0)
    return (predicted_bits == targets).float().mean()

def full_message_accuracy(outputs, targets):
    """A message is correct only if every one of its 16 thresholded bits is correct."""
    predicted_bits = torch.where(outputs >= 0, 1.0, -1.0)
    return (predicted_bits == targets).all(dim=1).float().mean()

def hamming_error(outputs, targets):
    """Mean count of incorrect bits per 16-bit message (0 is perfect; 8 is random)."""
    predicted_bits = torch.where(outputs >= 0, 1.0, -1.0)
    return (predicted_bits != targets).float().sum(dim=1).mean()

def reconstruction_error(outputs, targets):
    """Differentiable expected number of wrong bits: 0 perfect, 8 random, 16 opposite."""
    return (torch.abs((outputs + 1) / 2 - (targets + 1) / 2).sum(dim=1)).mean()

opt_ab = torch.optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=LR_ALICE_BOB)
opt_eve = torch.optim.Adam(eve.parameters(), lr=LR_EVE)
''')

md('## 10. Adversarial training loop')
code(r'''
# These lists hold true measurements, not hand-entered results.
history = {name: [] for name in ["step", "bob_loss", "eve_loss", "bob_acc", "eve_acc"]}

for step in range(1, TRAIN_STEPS + 1):
    # ---- Eve-only updates ----
    # Detaching c prevents Eve's optimizer from changing Alice while Eve learns.
    for _ in range(EVE_UPDATES):
        message, key = random_batch(BATCH_SIZE)
        with torch.no_grad():
            ciphertext = alice(torch.cat([message, key], dim=1))
        opt_eve.zero_grad(set_to_none=True)  # clear gradients left by the preceding update
        eve_loss = reconstruction_error(eve(ciphertext.detach()), message)
        finite(eve_loss)
        eve_loss.backward()
        opt_eve.step()

    # ---- Alice/Bob update ----
    message, key = random_batch(BATCH_SIZE)
    ciphertext = alice(torch.cat([message, key], dim=1))
    bob_logits = bob(torch.cat([ciphertext, key], dim=1))

    # Freeze Eve's parameters, but do not detach its output: gradients must flow to Alice.
    for parameter in eve.parameters():
        parameter.requires_grad_(False)
    eve_logits_for_ab = eve(ciphertext)
    bob_loss = reconstruction_error(bob_logits, message)
    eve_loss_for_ab = reconstruction_error(eve_logits_for_ab, message)
    ab_loss = bob_loss + (RANDOM_ERROR - eve_loss_for_ab).pow(2) / RANDOM_ERROR**2
    finite(ciphertext, bob_loss, eve_loss_for_ab, ab_loss)

    opt_ab.zero_grad(set_to_none=True)  # gradients must be cleared before each optimizer update
    ab_loss.backward()
    opt_ab.step()
    for parameter in eve.parameters():
        parameter.requires_grad_(True)

    if step % LOG_EVERY == 0 or step == 1:
        # Measure on this freshly generated batch. Accuracy is a metric, not an optimization loss.
        with torch.no_grad():
            history["step"].append(step)
            history["bob_loss"].append(bob_loss.item())
            history["eve_loss"].append(eve_loss_for_ab.item())
            history["bob_acc"].append(bit_accuracy(bob_logits, message).item())
            history["eve_acc"].append(bit_accuracy(eve_logits_for_ab, message).item())
        # A dependency-free progress bar: it updates only at logging intervals,
        # so it remains readable in Colab rather than printing every training step.
        progress = step / TRAIN_STEPS
        width = 24
        bar = "#" * int(progress * width) + "-" * (width - int(progress * width))
        print(f"[{bar}] {progress:6.1%} | step {step:>6,}/{TRAIN_STEPS:,} | "
              f"Bob error {bob_loss.item():5.2f}, acc {bit_accuracy(bob_logits, message).item():6.2%} | "
              f"Eve error {eve_loss_for_ab.item():5.2f}, acc {bit_accuracy(eve_logits_for_ab, message).item():6.2%}")

print("Adversarial training finished; all recorded losses were finite.")
''')

md('## 11. Training curves')
code(r'''
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
axes[0, 0].plot(history["step"], history["bob_loss"], label="Bob L1 error")
axes[0, 0].set_title("Bob reconstruction loss")
axes[0, 1].plot(history["step"], history["eve_loss"], color="tab:orange", label="Original Eve L1 error")
axes[0, 1].axhline(RANDOM_ERROR, color="gray", ls="--", lw=1, label="random error")
axes[0, 1].set_title("Original Eve reconstruction loss")
axes[1, 0].plot(history["step"], history["bob_acc"], color="tab:green")
axes[1, 0].axhline(.5, color="gray", ls="--", lw=1)
axes[1, 0].set_title("Bob bit accuracy")
axes[1, 1].plot(history["step"], history["eve_acc"], color="tab:red")
axes[1, 1].axhline(.5, color="gray", ls="--", lw=1)
axes[1, 1].set_title("Original Eve bit accuracy")
for ax in axes.flat:
    ax.set_xlabel("training step")
    ax.grid(alpha=.25)
axes[0, 0].set_ylabel("expected wrong bits")
axes[1, 0].set_ylabel("accuracy")
fig.tight_layout()
plt.show()
''')

md('## 12. Evaluation of Bob and the original Eve')
code(r'''
@torch.no_grad()
def evaluate_decoder(decoder, ciphertext, message, key=None):
    """Compute every requested held-out metric for Bob (key given) or Eve (no key)."""
    logits = decoder(torch.cat([ciphertext, key], dim=1)) if key is not None else decoder(ciphertext)
    return {
        "bit_accuracy": bit_accuracy(logits, message).item(),
        "full_message_accuracy": full_message_accuracy(logits, message).item(),
        "hamming_error": hamming_error(logits, message).item(),
        "reconstruction_error": reconstruction_error(logits, message).item(),
    }

# This is a genuinely new held-out random test set.
test_message, test_key = random_batch(EVAL_SIZE)
test_ciphertext = alice(torch.cat([test_message, test_key], dim=1))
bob_metrics = evaluate_decoder(bob, test_ciphertext, test_message, test_key)

original_eve_metrics = evaluate_decoder(eve, test_ciphertext, test_message)

def print_metrics(label, values):
    print(f"{label}: bit accuracy={values['bit_accuracy']:.2%}; "
          f"full-message accuracy={values['full_message_accuracy']:.2%}; "
          f"average Hamming error={values['hamming_error']:.2f}/16; "
          f"continuous reconstruction error={values['reconstruction_error']:.2f}/16")

print_metrics("Bob (held-out)", bob_metrics)
print_metrics("Original Eve (held-out)", original_eve_metrics)
''')

md('## 13. Freezing Alice and Bob')
code(r'''
# Freeze both learned communication networks completely before mounting a new attack.
# eval() fixes their inference behavior; requires_grad_(False) prevents any parameter updates.
alice.eval()
bob.eval()
for model in (alice, bob):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
assert not any(p.requires_grad for p in alice.parameters())
assert not any(p.requires_grad for p in bob.parameters())
print("Alice and Bob are frozen.")
''')

md('## 14. Creating and training a fresh post-hoc Eve')
code(r'''
# A new random attacker tests whether Alice learned something that only fooled the Eve
# encountered during training. New examples are generated by frozen Alice on every step.
fresh_history = {"step": [], "loss": [], "acc": []}
fresh_eve_metrics = None
best_fresh_eve_metrics = None

# A post-hoc attack is deliberately stronger than the earlier one: independently
# reinitialize Eve several times and report the best held-out attacker.
for restart in range(POSTHOC_EVE_RESTARTS):
    fresh_eve = MixAndTransform(CIPHER_BITS).to(device)
    fresh_opt = torch.optim.Adam(fresh_eve.parameters(), lr=LR_EVE)
    for step in range(1, FRESH_EVE_STEPS + 1):
        attack_message, attack_key = random_batch(BATCH_SIZE)
        with torch.no_grad():
            attack_ciphertext = alice(torch.cat([attack_message, attack_key], dim=1))
        fresh_opt.zero_grad(set_to_none=True)
        fresh_logits = fresh_eve(attack_ciphertext)
        fresh_loss = reconstruction_error(fresh_logits, attack_message)
        finite(fresh_loss)
        fresh_loss.backward()
        fresh_opt.step()
        # Keep a curve for the first fresh attacker; all restarts affect the final result.
        if restart == 0 and (step % LOG_EVERY == 0 or step == 1):
            with torch.no_grad():
                fresh_history["step"].append(step)
                fresh_history["loss"].append(fresh_loss.item())
                fresh_history["acc"].append(bit_accuracy(fresh_logits, attack_message).item())
    candidate = evaluate_decoder(fresh_eve, test_ciphertext, test_message)
    print_metrics(f"Fresh Eve restart {restart + 1}", candidate)
    if best_fresh_eve_metrics is None or candidate["bit_accuracy"] > best_fresh_eve_metrics["bit_accuracy"]:
        best_fresh_eve_metrics = candidate

fresh_eve_metrics = best_fresh_eve_metrics
print_metrics("Fresh Eve (held-out)", fresh_eve_metrics)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(fresh_history["step"], fresh_history["loss"], color="tab:orange")
axes[0].axhline(RANDOM_ERROR, color="gray", ls="--", lw=1)
axes[0].set(title="Fresh Eve attack error (restart 1)", xlabel="attack training step", ylabel="expected wrong bits")
axes[1].plot(fresh_history["step"], fresh_history["acc"], color="tab:red")
axes[1].axhline(.5, color="gray", ls="--", lw=1)
axes[1].set(title="Fresh Eve bit accuracy", xlabel="attack training step", ylabel="accuracy")
for ax in axes: ax.grid(alpha=.25)
fig.tight_layout()
plt.show()
''')

md('## 15. Cross-architecture post-hoc Eve attacks')
code(r'''
# The earlier fresh Eve has the same CNN architecture as the training Eve.  That is
# necessary but not sufficient: a learned cipher could hide a pattern from a CNN
# while leaving a global or sequential pattern for another kind of network.
#
# This cell keeps Alice and Bob frozen and gives three *new* attacker architectures
# the same ciphertext-only attack task and the same number of optimization steps.
# Their parameter counts are not identical, so this is an exploratory comparison;
# a formal study should tune and capacity-match every family across multiple seeds.
CROSS_ARCH_EVE_STEPS = 20_000

class MLPEve(nn.Module):
    """A general fixed-vector attacker: no convolutional locality assumption."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CIPHER_BITS, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, MESSAGE_BITS), nn.Tanh(),
        )
    def forward(self, ciphertext):
        return self.net(ciphertext)

class BiLSTMEve(nn.Module):
    """A bidirectional sequential attacker: each guessed bit sees the whole ciphertext."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=48, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.readout = nn.Linear(96, 1)
    def forward(self, ciphertext):
        sequence, _ = self.lstm(ciphertext.unsqueeze(-1))  # [B, 16] -> [B, 16, 1]
        return torch.tanh(self.readout(sequence).squeeze(-1))

class TransformerEve(nn.Module):
    """A self-attention attacker that can directly compare every ciphertext position."""
    def __init__(self):
        super().__init__()
        width = 64
        self.input_projection = nn.Linear(1, width)
        self.position = nn.Parameter(torch.zeros(1, CIPHER_BITS, width))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4,
                                           dim_feedforward=128, dropout=0.0,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.readout = nn.Linear(width, 1)
    def forward(self, ciphertext):
        x = self.input_projection(ciphertext.unsqueeze(-1)) + self.position
        return torch.tanh(self.readout(self.encoder(x)).squeeze(-1))


attackers = {
    "CNN (fresh)": MixAndTransform(CIPHER_BITS),
    "MLP": MLPEve(),
    "BiLSTM": BiLSTMEve(),
    "Transformer": TransformerEve(),
}
cross_arch_metrics = {}

for name, attacker in attackers.items():
    attacker = attacker.to(device)
    attacker.train()
    attack_optimizer = torch.optim.Adam(attacker.parameters(), lr=LR_EVE)
    for step in range(1, CROSS_ARCH_EVE_STEPS + 1):
        attack_message, attack_key = random_batch(BATCH_SIZE)
        with torch.no_grad():
            attack_ciphertext = alice(torch.cat([attack_message, attack_key], dim=1))
        attack_optimizer.zero_grad(set_to_none=True)
        attack_loss = reconstruction_error(attacker(attack_ciphertext), attack_message)
        finite(attack_loss)
        attack_loss.backward()
        attack_optimizer.step()
        if step % 2_000 == 0:
            print(f"{name:>13}: {step:>6,}/{CROSS_ARCH_EVE_STEPS:,} steps | "
                  f"training error {attack_loss.item():.2f}/16")

    attacker.eval()
    cross_arch_metrics[name] = evaluate_decoder(attacker, test_ciphertext, test_message)
    print_metrics(f"{name} attack (held-out)", cross_arch_metrics[name])

labels = list(cross_arch_metrics)
attack_accuracies = [cross_arch_metrics[name]["bit_accuracy"] for name in labels]
plt.figure(figsize=(8, 4))
bars = plt.bar(labels, attack_accuracies, color=["tab:orange", "tab:blue", "tab:purple", "tab:brown"])
plt.axhline(.5, color="gray", ls="--", lw=1, label="random guessing")
plt.ylim(0, 1.05)
plt.ylabel("held-out bit accuracy")
plt.title("Cross-architecture attacks on frozen Alice/Bob")
plt.legend()
for bar, value in zip(bars, attack_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, value + .02, f"{value:.1%}", ha="center")
plt.tight_layout()
plt.show()
''')

md('## 15. Final comparison')
code(r'''
labels = ["Bob", "Original Eve", "Fresh Eve"]
accuracies = [bob_metrics["bit_accuracy"], original_eve_metrics["bit_accuracy"], fresh_eve_metrics["bit_accuracy"]]
colors = ["tab:green", "tab:red", "tab:orange"]
plt.figure(figsize=(7, 4))
bars = plt.bar(labels, accuracies, color=colors)
plt.axhline(.5, color="gray", ls="--", lw=1, label="random guessing")
plt.ylim(0, 1.05)
plt.ylabel("held-out bit accuracy")
plt.title("Final held-out decoding comparison")
plt.legend()
for bar, value in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, value + .02, f"{value:.1%}", ha="center")
plt.tight_layout()
plt.show()
''')

md('## 16. What the results mean')
code(r'''
if bob_metrics["bit_accuracy"] < 0.90:
    interpretation = ("Bob did not decode reliably, so this run did not establish a successful learned "
                      "communication system. Try more training steps or tune the hyperparameters.")
elif fresh_eve_metrics["bit_accuracy"] > original_eve_metrics["bit_accuracy"] + 0.05:
    interpretation = ("The fresh Eve attacks better than the original Eve. This suggests the original "
                      "attacker overfit to the adversarial game rather than Alice learning broadly useful secrecy.")
elif (original_eve_metrics["bit_accuracy"] < 0.58 and fresh_eve_metrics["bit_accuracy"] < 0.58):
    interpretation = ("Both Eves remain near random guessing on this test. That is empirical robustness "
                      "evidence for this particular experiment, not a cryptographic proof.")
else:
    interpretation = ("Bob communicates well, but at least one attacker recovers meaningfully more than "
                      "random bits. The learned transformation should not be considered secret.")
print(interpretation)
''')

md(r'''
## 17. Limitations

This experiment is intentionally small. Its continuous ciphertext, finite network capacity, and limited attacker training do **not** provide formal secrecy. A more powerful, better-tuned, or differently structured Eve can often learn an attack; the post-hoc Eve is specifically included to reveal that risk. Results vary with seeds and hyperparameters. Do not use this system to protect real data—use reviewed, standard cryptographic primitives such as AES with authenticated encryption instead.
''')

md('## 18. Overnight 32-bit, five-seed cross-architecture sweep')
code(r'''
# Seed 7 is the run already completed by this notebook. This cell runs four more
# independent 32-bit CNN-trained Alice/Bob systems, then prints and displays all
# five-run results directly in the notebook output.

assert MESSAGE_BITS == KEY_BITS == CIPHER_BITS == 32, "The overnight sweep is configured for 32-bit messages."
OVERNIGHT_EXTRA_SEEDS = [8, 9, 10, 11]

def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_attack_one_seed(seed):
    """Train against a CNN Eve, freeze Alice/Bob, then train each fresh attacker."""
    set_experiment_seed(seed)
    run_alice = MixAndTransform(MESSAGE_BITS + KEY_BITS).to(device)
    run_bob = MixAndTransform(CIPHER_BITS + KEY_BITS).to(device)
    run_eve = MixAndTransform(CIPHER_BITS).to(device)
    run_opt_ab = torch.optim.Adam(list(run_alice.parameters()) + list(run_bob.parameters()), lr=LR_ALICE_BOB)
    run_opt_eve = torch.optim.Adam(run_eve.parameters(), lr=LR_EVE)

    print(f"\n{'=' * 18} Seed {seed}: CNN adversarial training {'=' * 18}")
    for step in range(1, TRAIN_STEPS + 1):
        # Eve-only updates: detached ciphertext keeps Alice fixed during these updates.
        for _ in range(EVE_UPDATES):
            message, key = random_batch(BATCH_SIZE)
            with torch.no_grad():
                ciphertext = run_alice(torch.cat([message, key], dim=1))
            run_opt_eve.zero_grad(set_to_none=True)
            eve_loss = reconstruction_error(run_eve(ciphertext.detach()), message)
            finite(eve_loss)
            eve_loss.backward()
            run_opt_eve.step()

        # Alice/Bob update: Eve's weights are frozen while its gradient still reaches Alice.
        message, key = random_batch(BATCH_SIZE)
        ciphertext = run_alice(torch.cat([message, key], dim=1))
        bob_output = run_bob(torch.cat([ciphertext, key], dim=1))
        for parameter in run_eve.parameters():
            parameter.requires_grad_(False)
        eve_output = run_eve(ciphertext)
        bob_loss = reconstruction_error(bob_output, message)
        eve_loss_for_ab = reconstruction_error(eve_output, message)
        ab_loss = bob_loss + (RANDOM_ERROR - eve_loss_for_ab).pow(2) / RANDOM_ERROR**2
        finite(bob_loss, eve_loss_for_ab, ab_loss)
        run_opt_ab.zero_grad(set_to_none=True)
        ab_loss.backward()
        run_opt_ab.step()
        for parameter in run_eve.parameters():
            parameter.requires_grad_(True)

        if step % 5_000 == 0 or step == 1:
            print(f"Seed {seed} | {step:>6,}/{TRAIN_STEPS:,} | Bob {bit_accuracy(bob_output, message).item():.2%} | "
                  f"CNN Eve {bit_accuracy(eve_output, message).item():.2%}")

    # Freeze Alice and Bob and evaluate every attacker on exactly the same held-out set.
    run_alice.eval()
    run_bob.eval()
    for model in (run_alice, run_bob):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    held_message, held_key = random_batch(EVAL_SIZE)
    with torch.no_grad():
        held_ciphertext = run_alice(torch.cat([held_message, held_key], dim=1))
    run_bob_metrics = evaluate_decoder(run_bob, held_ciphertext, held_message, held_key)

    run_attackers = {
        "CNN (fresh)": MixAndTransform(CIPHER_BITS),
        "MLP": MLPEve(),
        "BiLSTM": BiLSTMEve(),
        "Transformer": TransformerEve(),
    }
    run_attack_metrics = {}
    for name, attacker in run_attackers.items():
        attacker = attacker.to(device)
        attack_optimizer = torch.optim.Adam(attacker.parameters(), lr=LR_EVE)
        for _ in range(CROSS_ARCH_EVE_STEPS):
            attack_message, attack_key = random_batch(BATCH_SIZE)
            with torch.no_grad():
                attack_ciphertext = run_alice(torch.cat([attack_message, attack_key], dim=1))
            attack_optimizer.zero_grad(set_to_none=True)
            attack_loss = reconstruction_error(attacker(attack_ciphertext), attack_message)
            finite(attack_loss)
            attack_loss.backward()
            attack_optimizer.step()
        attacker.eval()
        run_attack_metrics[name] = evaluate_decoder(attacker, held_ciphertext, held_message)
        print(f"Seed {seed} | {name:>13}: held-out bit accuracy {run_attack_metrics[name]['bit_accuracy']:.2%}")
        del attacker, attack_optimizer

    del run_alice, run_bob, run_eve, run_opt_ab, run_opt_eve, run_attackers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_bob_metrics, run_attack_metrics

# Include the current notebook's Seed 7 result, then run Seeds 8-11.
all_bob_rows = [{"seed": SEED, **bob_metrics}]
all_attack_rows = [{"seed": SEED, "attacker": name, **metrics} for name, metrics in cross_arch_metrics.items()]
for seed in OVERNIGHT_EXTRA_SEEDS:
    seed_bob_metrics, seed_attack_metrics = train_and_attack_one_seed(seed)
    all_bob_rows.append({"seed": seed, **seed_bob_metrics})
    all_attack_rows.extend({"seed": seed, "attacker": name, **metrics}
                           for name, metrics in seed_attack_metrics.items())

seeds = [row["seed"] for row in all_bob_rows]
attacker_names = ["CNN (fresh)", "MLP", "BiLSTM", "Transformer"]
summary_rows = []
for name in attacker_names:
    rows = [row for row in all_attack_rows if row["attacker"] == name]
    accuracies = np.array([row["bit_accuracy"] for row in rows])
    hamming = np.array([row["hamming_error"] for row in rows])
    summary_rows.append({"attacker": name, "mean_bit_accuracy": float(accuracies.mean()),
                         "std_bit_accuracy": float(accuracies.std(ddof=1)),
                         "mean_hamming_error": float(hamming.mean()),
                         "std_hamming_error": float(hamming.std(ddof=1))})
mean_bob_accuracy = np.mean([row["bit_accuracy"] for row in all_bob_rows])
std_bob_accuracy = np.std([row["bit_accuracy"] for row in all_bob_rows], ddof=1)
print("\n" + "=" * 73)
print("32-bit cross-architecture results across five seeds")
print(f"{'Attacker':<16} {'Mean accuracy':>15} {'Std. dev.':>12} {'Mean Hamming':>15}")
for row in summary_rows:
    print(f"{row['attacker']:<16} {row['mean_bit_accuracy']:>14.2%} {row['std_bit_accuracy']:>11.2%} "
          f"{row['mean_hamming_error']:>14.2f}/32")
print(f"Bob              {mean_bob_accuracy:>14.2%} {std_bob_accuracy:>11.2%} (communication baseline)")

# Graph 1 shows whether an attack is stable or dependent on a lucky random seed.
fig, ax = plt.subplots(figsize=(9, 4.8))
for name in attacker_names:
    rows = sorted((row for row in all_attack_rows if row["attacker"] == name), key=lambda row: row["seed"])
    ax.plot(seeds, [row["bit_accuracy"] for row in rows], marker="o", linewidth=2, label=name)
ax.axhline(.5, color="gray", ls="--", lw=1, label="random guessing")
ax.set(xlabel="random seed", ylabel="held-out Eve bit accuracy", ylim=(0, 1.02),
       title="32-bit cross-architecture attacks across seeds")
ax.grid(alpha=.25)
ax.legend(ncol=2)
fig.tight_layout()
plt.show()

# Graph 2 is the presentation-ready aggregate: mean and one-standard-deviation bars.
labels = [row["attacker"] for row in summary_rows] + ["Bob"]
means = [row["mean_bit_accuracy"] for row in summary_rows] + [mean_bob_accuracy]
errors = [row["std_bit_accuracy"] for row in summary_rows] + [std_bob_accuracy]
fig, ax = plt.subplots(figsize=(9, 4.8))
bars = ax.bar(labels, means, yerr=errors, capsize=5,
              color=["tab:orange", "tab:blue", "tab:purple", "tab:brown", "tab:green"])
ax.axhline(.5, color="gray", ls="--", lw=1, label="random guessing")
ax.set(ylabel="held-out bit accuracy (mean ± 1 std)", ylim=(0, 1.08),
       title="Five-seed summary: frozen 32-bit Alice/Bob")
ax.legend()
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + .025, f"{mean:.1%}", ha="center")
fig.tight_layout()
plt.show()
''')

with open("adversarial_neural_cryptography_demo.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Wrote adversarial_neural_cryptography_demo.ipynb")
