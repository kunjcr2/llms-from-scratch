"""Build a standalone Colab notebook for the 4x4 Eve-architecture sweep."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        markdown(
            """
# Fast 4x4 Eve Architecture Sweep

Run this notebook in Google Colab with a **T4** or **L4** GPU. It keeps Alice and Bob as the same CNN architecture, then runs every combination of:

```text
Eve used during Alice/Bob training  x  fresh Eve used to attack afterward
CNN, MLP, BiLSTM, Transformer       x  CNN, MLP, BiLSTM, Transformer
```

Run the two code cells below in order. The second cell prints progress, the final 4x4 table, and two inline graphs. It writes no checkpoints, CSVs, or plots.
"""
        ),
        code(
            r'''
# Run this setup cell first.
import random
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from IPython.display import display

SEED = 42  # Choose exactly one: 42 or 7.
assert SEED in (42, 7)
ARCHITECTURES = ["CNN", "MLP", "BiLSTM", "Transformer"]

# Tuned for a full sweep on a Colab T4/L4. If a T4 runs out of memory, use 4096.
MESSAGE_BITS = KEY_BITS = CIPHER_BITS = 32
BATCH_SIZE = 8192 if torch.cuda.is_available() else 512
TRAIN_STEPS = 20_000
ATTACK_STEPS = 8_000
EVE_UPDATES = 2
EVAL_SIZE = 16_384
LOG_EVERY = 500
LR_ALICE_BOB = 8e-4
LR_EVE = 1e-3
RANDOM_ERROR = MESSAGE_BITS / 2
PRIVACY_WEIGHT = 8.0

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

USE_AMP = device == "cuda"
try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
except AttributeError:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

def autocast_context():
    return torch.autocast("cuda", dtype=torch.float16) if USE_AMP else nullcontext()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def derived_seed(*values):
    result = 17
    for value in values:
        result = (result * 1_000_003 + int(value)) % (2**31 - 1)
    return result

def random_batch(batch_size):
    message = 2 * torch.randint(0, 2, (batch_size, MESSAGE_BITS), device=device).float() - 1
    key = 2 * torch.randint(0, 2, (batch_size, KEY_BITS), device=device).float() - 1
    return message, key

def evaluation_batch(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    message = 2 * torch.randint(0, 2, (EVAL_SIZE, MESSAGE_BITS), generator=generator).float() - 1
    key = 2 * torch.randint(0, 2, (EVAL_SIZE, KEY_BITS), generator=generator).float() - 1
    return message.to(device), key.to(device)

def reconstruction_error(outputs, targets):
    return (torch.abs((outputs + 1) / 2 - (targets + 1) / 2).sum(dim=1)).mean()

def bits(outputs):
    return torch.where(outputs >= 0, 1.0, -1.0)

def bit_accuracy(outputs, targets):
    return (bits(outputs) == targets).float().mean()

def full_message_accuracy(outputs, targets):
    return (bits(outputs) == targets).all(dim=1).float().mean()

def hamming_error(outputs, targets):
    return (bits(outputs) != targets).float().sum(dim=1).mean()

def optimizer_step(loss, optimizer):
    optimizer.zero_grad(set_to_none=True)
    if USE_AMP:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

class AliceBobCNN(nn.Module):
    """A 19,409-parameter CNN used for both Alice and Bob."""
    def __init__(self, input_bits):
        super().__init__()
        # 64 input values -> 128 positions -> 64 -> 32 output positions.
        self.mix = nn.Linear(input_bits, 128)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding="same")
        self.conv2 = nn.Conv1d(32, 48, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(48, 48, kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(48, 1, kernel_size=1)

    def forward(self, x):
        x = self.mix(x).unsqueeze(1)
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return torch.tanh(self.conv4(x)).squeeze(1)

class CNNEve(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(CIPHER_BITS, 192)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=3)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, ciphertext):
        x = self.mix(ciphertext).unsqueeze(1)
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return torch.tanh(self.conv4(x).squeeze(1))

class MLPEve(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 24, num_layers=2, batch_first=True, bidirectional=True)
        self.readout = nn.Linear(48, 1)

    def forward(self, ciphertext):
        sequence, _ = self.lstm(ciphertext.unsqueeze(-1))
        return torch.tanh(self.readout(sequence).squeeze(-1))

class TransformerEve(nn.Module):
    def __init__(self):
        super().__init__()
        width = 32
        self.input_projection = nn.Linear(1, width)
        self.position = nn.Parameter(torch.zeros(1, CIPHER_BITS, width))
        layer = nn.TransformerEncoderLayer(width, 4, 64, dropout=0.0,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.readout = nn.Linear(width, 1)

    def forward(self, ciphertext):
        x = self.input_projection(ciphertext.unsqueeze(-1)) + self.position
        return torch.tanh(self.readout(self.encoder(x)).squeeze(-1))

def make_eve(name):
    return {
        "CNN": CNNEve,
        "MLP": MLPEve,
        "BiLSTM": BiLSTMEve,
        "Transformer": TransformerEve,
    }[name]()

def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())

alice_bob_parameters = parameter_count(AliceBobCNN(MESSAGE_BITS + KEY_BITS))
eve_parameter_counts = {name: parameter_count(make_eve(name)) for name in ARCHITECTURES}
assert alice_bob_parameters >= 15_000
assert all(15_000 <= count <= 26_000 for count in eve_parameter_counts.values())

print(f"Using {device}; AMP={USE_AMP}; batch size={BATCH_SIZE:,}")
print(f"Full sweep: four training Eves x four attackers; seed={SEED}")
print(f"Alice parameters: {alice_bob_parameters:,}; Bob parameters: {alice_bob_parameters:,}")
print("Eve parameters:", eve_parameter_counts)
print(f"Privacy-loss weight: {PRIVACY_WEIGHT:g}; in-game Eve updates per step: {EVE_UPDATES}")
'''
        ),
        code(
            r'''
# Run this cell second. It runs the entire 4x4 sweep and prints the results.
pair_rows, attack_rows = [], []

for trainer_index, training_eve_name in enumerate(ARCHITECTURES):
    set_seed(derived_seed(SEED, trainer_index))
    alice = AliceBobCNN(MESSAGE_BITS + KEY_BITS).to(device)
    bob = AliceBobCNN(CIPHER_BITS + KEY_BITS).to(device)
    training_eve = make_eve(training_eve_name).to(device)
    ab_optimizer = torch.optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=LR_ALICE_BOB)
    eve_optimizer = torch.optim.Adam(training_eve.parameters(), lr=LR_EVE)

    print(f"\n{'=' * 16} Train Alice/Bob vs {training_eve_name} Eve {'=' * 16}")
    alice.train(); bob.train(); training_eve.train()
    for step in range(1, TRAIN_STEPS + 1):
        message, key = random_batch(BATCH_SIZE)
        with torch.no_grad(), autocast_context():
            ciphertext = alice(torch.cat([message, key], dim=1))
        with autocast_context():
            eve_loss = reconstruction_error(training_eve(ciphertext), message)
        optimizer_step(eve_loss, eve_optimizer)

        message, key = random_batch(BATCH_SIZE)
        for parameter in training_eve.parameters():
            parameter.requires_grad_(False)
        with autocast_context():
            ciphertext = alice(torch.cat([message, key], dim=1))
            bob_output = bob(torch.cat([ciphertext, key], dim=1))
            eve_output = training_eve(ciphertext)
            bob_loss = reconstruction_error(bob_output, message)
            eve_loss_for_ab = reconstruction_error(eve_output, message)
            privacy_loss = (RANDOM_ERROR - eve_loss_for_ab).pow(2) / RANDOM_ERROR**2
            ab_loss = bob_loss + PRIVACY_WEIGHT * privacy_loss
        optimizer_step(ab_loss, ab_optimizer)
        for parameter in training_eve.parameters():
            parameter.requires_grad_(True)

        if step == 1 or step % LOG_EVERY == 0 or step == TRAIN_STEPS:
            print(f"{training_eve_name:>11} | {step:>6,}/{TRAIN_STEPS:,} | "
                  f"Bob {bit_accuracy(bob_output, message).item():.2%} "
                  f"(L1 {bob_loss.item():.2f}) | "
                  f"in-game Eve {bit_accuracy(eve_output, message).item():.2%} "
                  f"(L1 {eve_loss_for_ab.item():.2f})")

    alice.eval(); bob.eval(); training_eve.eval()
    for model in (alice, bob):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    test_message, test_key = evaluation_batch(derived_seed(SEED, trainer_index, 99))
    with torch.inference_mode(), autocast_context():
        test_ciphertext = alice(torch.cat([test_message, test_key], dim=1))
        bob_output = bob(torch.cat([test_ciphertext, test_key], dim=1))
        in_game_output = training_eve(test_ciphertext)
    pair_rows.append({
        "training_eve": training_eve_name,
        "bob_bit_accuracy": bit_accuracy(bob_output, test_message).item(),
        "in_game_eve_bit_accuracy": bit_accuracy(in_game_output, test_message).item(),
        "bob_full_message_accuracy": full_message_accuracy(bob_output, test_message).item(),
        "in_game_eve_full_message_accuracy": full_message_accuracy(in_game_output, test_message).item(),
    })

    for attacker_index, attacker_name in enumerate(ARCHITECTURES):
        set_seed(derived_seed(SEED, trainer_index, attacker_index, 123))
        attacker = make_eve(attacker_name).to(device)
        attacker_optimizer = torch.optim.Adam(attacker.parameters(), lr=LR_EVE)
        attacker.train()
        print(f"  Attack {training_eve_name} pair with {attacker_name} Eve")
        for step in range(1, ATTACK_STEPS + 1):
            attack_message, attack_key = random_batch(BATCH_SIZE)
            with torch.no_grad(), autocast_context():
                attack_ciphertext = alice(torch.cat([attack_message, attack_key], dim=1))
            with autocast_context():
                attack_output = attacker(attack_ciphertext)
                attack_loss = reconstruction_error(attack_output, attack_message)
            optimizer_step(attack_loss, attacker_optimizer)
            if step == 1 or step % LOG_EVERY == 0 or step == ATTACK_STEPS:
                print(f"    {attacker_name:<11} {step:>5,}/{ATTACK_STEPS:,} | "
                      f"attack {bit_accuracy(attack_output, attack_message).item():.2%}")

        attacker.eval()
        with torch.inference_mode(), autocast_context():
            test_output = attacker(test_ciphertext)
        attack_rows.append({
            "training_eve": training_eve_name,
            "posthoc_attacker": attacker_name,
            "bit_accuracy": bit_accuracy(test_output, test_message).item(),
            "full_message_accuracy": full_message_accuracy(test_output, test_message).item(),
            "hamming_error": hamming_error(test_output, test_message).item(),
        })
        print(f"    FINAL {training_eve_name:>11} -> {attacker_name:<11}: "
              f"{attack_rows[-1]['bit_accuracy']:.2%}")
        del attacker, attacker_optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    del alice, bob, training_eve, ab_optimizer, eve_optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

pair_table = pd.DataFrame(pair_rows)
attack_table = pd.DataFrame(attack_rows)
attack_matrix = attack_table.pivot(index="training_eve", columns="posthoc_attacker", values="bit_accuracy")
attack_matrix = attack_matrix.reindex(index=ARCHITECTURES, columns=ARCHITECTURES)

display(pair_table.style.format({
    "bob_bit_accuracy": "{:.2%}",
    "in_game_eve_bit_accuracy": "{:.2%}",
    "bob_full_message_accuracy": "{:.2%}",
    "in_game_eve_full_message_accuracy": "{:.2%}",
}).set_caption("Held-out Bob and in-game Eve"))
display(attack_matrix.style.format("{:.2%}").set_caption(
    "Held-out Eve bit accuracy: training Eve (rows) x fresh attacker (columns); 50% = random"
))

fig, ax = plt.subplots(figsize=(8.5, 5.5))
image = ax.imshow(attack_matrix.to_numpy(), vmin=.5, vmax=.7, cmap="magma")
ax.set_xticks(range(4), ARCHITECTURES)
ax.set_yticks(range(4), ARCHITECTURES)
ax.set_xlabel("Fresh post-hoc Eve architecture")
ax.set_ylabel("Eve architecture used while training Alice/Bob")
ax.set_title("Full Eve architecture attack matrix")
for row in range(4):
    for column in range(4):
        value = attack_matrix.iloc[row, column]
        ax.text(column, row, f"{value:.1%}", ha="center", va="center",
                color="white" if value < .60 else "black")
fig.colorbar(image, ax=ax, label="held-out Eve bit accuracy (50% to 70% scale)")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5))
locations = np.arange(4)
ax.bar(locations - .18, pair_table["bob_bit_accuracy"], .36, label="Bob", color="tab:green")
ax.bar(locations + .18, pair_table["in_game_eve_bit_accuracy"], .36,
       label="in-game Eve", color="tab:red")
ax.axhline(.5, color="gray", linestyle="--", linewidth=1, label="random guessing")
ax.set_xticks(locations, ARCHITECTURES)
ax.set_ylim(0, 1.08)
ax.set_ylabel("held-out bit accuracy")
ax.set_title("Bob reliability by training Eve")
ax.legend()
fig.tight_layout()
plt.show()
'''
        ),
    ],
}

output = Path(__file__).with_name("adversarial_neural_cryptography_4x4_sweep.ipynb")
output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {output}")
