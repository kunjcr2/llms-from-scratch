r"""Compare CNN, MLP, and CNN+MLP Stage-1 Eves using fresh MLP attacks.

Run from the repository root in Linux/macOS or WSL (Colab CLI is not native
Windows software). Example for the hybrid condition and pair seed 7:

    mkdir -p crypt_dl/runs
    colab run --gpu A100 --keep -s crypt-hybrid-7 --timeout 14400 \
      crypt_dl/scripts/hybrid_eve_mlp_study.py \
      --conditions hybrid --pair-seeds 7 \
      --output /content/crypt-hybrid-7.jsonl
    colab download -s crypt-hybrid-7 \
      /content/crypt-hybrid-7.jsonl crypt_dl/runs/crypt-hybrid-7.jsonl
    colab stop -s crypt-hybrid-7

Repeat for cnn, mlp, and hybrid with pair seeds 7, 8, and 9; change the session
and output names each time. Download before stopping, including after an
interrupted run. The --keep option retains the remote session until stopped.
Each pair uses only fresh MLP attackers. JSONL results are appended immediately.
"""

import argparse
import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


BITS = 32
CNN_WIDTH = 54  # 39,757 parameters
MLP_STAGE_WIDTHS = (133, 218)  # 40,609 parameters


class CryptCNN(nn.Module):
    """Original 40,609-parameter Alice/Bob architecture from the size study."""

    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(64, 192)
        self.conv1 = nn.Conv1d(1, 48, 5, padding="same")
        self.conv2 = nn.Conv1d(48, 96, 4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(96, 48, 2, stride=3)
        self.conv4 = nn.Conv1d(48, 1, 1)

    def forward(self, inputs):
        x = self.mix(inputs).unsqueeze(1)
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return torch.tanh(self.conv4(x).squeeze(1))


class CNNEve(nn.Module):
    def __init__(self, width=CNN_WIDTH):
        super().__init__()
        self.mix = nn.Linear(32, 128)
        self.conv1 = nn.Conv1d(1, width, 5, padding="same")
        self.conv2 = nn.Conv1d(width, 2 * width, 4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(2 * width, width, 2, stride=2)
        self.conv4 = nn.Conv1d(width, 1, 1)

    def forward(self, ciphertext):
        x = self.mix(ciphertext).unsqueeze(1)
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return torch.tanh(self.conv4(x).squeeze(1))


class MLPEve(nn.Module):
    def __init__(self, width, hidden2=None):
        super().__init__()
        hidden2 = width if hidden2 is None else hidden2
        self.net = nn.Sequential(
            nn.Linear(BITS, width), nn.ReLU(),
            nn.Linear(width, hidden2), nn.ReLU(),
            nn.Linear(hidden2, BITS), nn.Tanh(),
        )

    def forward(self, ciphertext):
        return self.net(ciphertext)


def count(model):
    return sum(p.numel() for p in model.parameters())


def mlp_for_target(target):
    # Equal hidden widths: parameters = width**2 + 66*width + 32.
    width = min(range(8, 321), key=lambda w: abs(w * w + 66 * w + 32 - target))
    return MLPEve(width), width


def batch(n, rng, device):
    message = torch.randint(0, 2, (n, BITS), generator=rng, device=device).float() * 2 - 1
    key = torch.randint(0, 2, (n, BITS), generator=rng, device=device).float() * 2 - 1
    return message, key


def l1_error(prediction, message):
    return ((prediction.float() - message).abs() / 2).sum(dim=1).mean()


def precision(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def score(model, inputs, labels, device, chunk_size):
    wrong = total_l1 = full = 0.0
    for start in range(0, len(labels), chunk_size):
        target = labels[start:start + chunk_size]
        with precision(device):
            prediction = model(inputs[start:start + chunk_size])
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite evaluation prediction")
        errors = ((prediction >= 0) != (target >= 0)).sum(1)
        wrong += errors.sum().item()
        full += (errors == 0).sum().item()
        total_l1 += ((prediction.float() - target).abs() / 2).sum().item()
    n = len(labels)
    return dict(bit_accuracy=1 - wrong / (n * BITS), hamming_error=wrong / n,
                full_message_accuracy=full / n, reconstruction_l1=total_l1 / n,
                messages=n)


@torch.inference_mode()
def held_out(alice, bob, n, seed, device, chunk_size):
    rng = torch.Generator(device=device).manual_seed(seed)
    messages, keys = batch(n, rng, device)
    parts = []
    for start in range(0, n, chunk_size):
        with precision(device):
            parts.append(alice(torch.cat((messages[start:start + chunk_size],
                                          keys[start:start + chunk_size]), 1)))
    ciphertext = torch.cat(parts)
    bob_score = score(bob, torch.cat((ciphertext, keys), 1), messages, device, chunk_size)
    return (ciphertext, messages), bob_score


def append_record(path, record):
    print(record["type"].upper() + "_JSON " + json.dumps(record, allow_nan=False), flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()


def set_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def train_pair(condition, pair_seed, args, device, output):
    # Initialize Alice/Bob identically across conditions for a given pair seed.
    set_seed(pair_seed, device)
    alice = CryptCNN().to(device)
    bob = CryptCNN().to(device)
    eves = {}
    if condition in ("cnn", "hybrid"):
        set_seed(10_000 + pair_seed, device)
        eves["cnn"] = CNNEve().to(device)
    if condition in ("mlp", "hybrid"):
        set_seed(20_000 + pair_seed, device)
        eves["mlp"] = MLPEve(*MLP_STAGE_WIDTHS).to(device)

    ab_opt = torch.optim.Adam([*alice.parameters(), *bob.parameters()], lr=args.ab_lr)
    eve_opts = {name: torch.optim.Adam(eve.parameters(), lr=args.eve_lr)
                for name, eve in eves.items()}
    # Separate streams make Alice/Bob batches and each Eve's batches identical
    # for the same pair seed across single and hybrid conditions.
    ab_rng = torch.Generator(device=device).manual_seed(100_000 + pair_seed)
    eve_rngs = {name: torch.Generator(device=device).manual_seed(
        (110_000 if name == "cnn" else 120_000) + pair_seed) for name in eves}
    started = time.perf_counter()
    print(f"PAIR_START condition={condition} seed={pair_seed} steps={args.pair_steps}", flush=True)

    eve_names = tuple(eves)
    updates_per_model = {name: sum(index % len(eve_names) == offset
                                   for index in range(args.total_eve_updates))
                         for offset, name in enumerate(eve_names)}
    for step in range(1, args.pair_steps + 1):
        for index in range(args.total_eve_updates):
            name = eve_names[index % len(eve_names)]
            eve = eves[name]
            message, key = batch(args.batch_size, eve_rngs[name], device)
            with torch.no_grad(), precision(device):
                ciphertext = alice(torch.cat((message, key), 1))
            with precision(device):
                eve_loss = l1_error(eve(ciphertext), message)
            eve_opts[name].zero_grad(set_to_none=True)
            eve_loss.backward()
            eve_opts[name].step()

        message, key = batch(args.batch_size, ab_rng, device)
        for eve in eves.values():
            eve.requires_grad_(False)
        with precision(device):
            ciphertext = alice(torch.cat((message, key), 1))
            bob_loss = l1_error(bob(torch.cat((ciphertext, key), 1)), message)
            eve_errors = {name: l1_error(eve(ciphertext), message)
                          for name, eve in eves.items()}
            # Optimize against the Eve that reconstructs the plaintext best on
            # this batch. The same formula with one Eve reproduces the baseline.
            best_eve_error = torch.stack(tuple(eve_errors.values())).min()
            privacy_loss = ((BITS / 2 - best_eve_error) / (BITS / 2)).square()
            total_loss = bob_loss + args.privacy_weight * privacy_loss
        ab_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        ab_opt.step()
        for eve in eves.values():
            eve.requires_grad_(True)

        if step == 1 or step % args.log_every == 0 or step == args.pair_steps:
            values = [total_loss.item(), bob_loss.item(),
                      *(error.item() for error in eve_errors.values())]
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"Nonfinite Stage-1 loss at step {step}")
            print(f"PAIR_STEP condition={condition} seed={pair_seed} step={step} "
                  f"bob_l1={bob_loss.item():.4f} "
                  f"eve_l1={{{', '.join(f'{k}: {v.item():.4f}' for k, v in eve_errors.items())}}} "
                  f"minutes={(time.perf_counter()-started)/60:.1f}", flush=True)

    alice.eval().requires_grad_(False)
    bob.eval().requires_grad_(False)
    for eve in eves.values():
        eve.eval().requires_grad_(False)
    validation, bob_validation = held_out(alice, bob, args.validation_size,
                                          200_000 + pair_seed, device, args.eval_batch_size)
    test, bob_test = held_out(alice, bob, args.test_size,
                              300_000 + pair_seed, device, args.eval_batch_size)
    reliable = (bob_validation["bit_accuracy"] >= args.bob_min and
                bob_test["bit_accuracy"] >= args.bob_min)
    record = dict(type="pair", condition=condition, pair_seed=pair_seed,
                  pair_steps=args.pair_steps, batch_size=args.batch_size,
                  total_eve_updates=args.total_eve_updates,
                  eve_updates_per_model=updates_per_model,
                  alice_params=count(alice), bob_params=count(bob),
                  eve_params={name: count(eve) for name, eve in eves.items()},
                  objective="bob_l1 + weight * ((16 - min_eve_l1)^2 / 16^2)",
                  bob_min=args.bob_min, bob_validation=bob_validation,
                  bob_test=bob_test, reliable_bob=reliable,
                  stage1_eve_test={name: score(eve, *test, device, args.eval_batch_size)
                                   for name, eve in eves.items()})
    append_record(output, record)
    return alice, validation, test, reliable


def attack(alice, validation, test, condition, pair_seed, target, seed,
           args, device, output):
    set_seed(seed, device)
    attacker, width = mlp_for_target(target)
    attacker = attacker.to(device)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=args.attack_lr)
    # Identical attack examples across conditions, capacities, and initializations.
    rng = torch.Generator(device=device).manual_seed(400_000 + pair_seed)
    checkpoints = sorted(set((args.early_attack_step, args.attack_steps)))
    print(f"ATTACK_START condition={condition} pair={pair_seed} target={target} "
          f"actual={count(attacker)} seed={seed}", flush=True)
    for step in range(1, args.attack_steps + 1):
        message, key = batch(args.batch_size, rng, device)
        with torch.no_grad(), precision(device):
            ciphertext = alice(torch.cat((message, key), 1))
        with precision(device):
            loss = l1_error(attacker(ciphertext), message)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step in checkpoints:
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"Nonfinite attacker loss at step {step}")
            print(f"ATTACK_STEP condition={condition} pair={pair_seed} target={target} "
                  f"seed={seed} step={step} train_l1={loss.item():.4f}", flush=True)
        if step in checkpoints:
            attacker.eval()
            record = dict(type="attack", condition=condition, pair_seed=pair_seed,
                          family="mlp", target_params=target, actual_params=count(attacker),
                          width=width, attacker_seed=seed, step=step,
                          validation=score(attacker, *validation, device, args.eval_batch_size),
                          test=score(attacker, *test, device, args.eval_batch_size))
            append_record(output, record)
            attacker.train()


def summarize(path, final_step):
    """Average attacker starts within each pair before averaging pairs."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_pair = {}
    for record in records:
        if record["type"] == "attack" and record["step"] == final_step:
            key = (record["condition"], record["target_params"], record["pair_seed"])
            by_pair.setdefault(key, []).append(record["test"]["bit_accuracy"])
    by_condition = {}
    for (condition, target, pair_seed), scores in by_pair.items():
        by_condition.setdefault((condition, target), []).append(
            dict(pair_seed=pair_seed, mean_bit_accuracy=statistics.mean(scores),
                 attacker_starts=len(scores)))
    for (condition, target), pairs in sorted(by_condition.items()):
        pair_means = [pair["mean_bit_accuracy"] for pair in pairs]
        append_record(path, dict(type="summary", condition=condition,
                      target_params=target, step=final_step, pairs=sorted(
                          pairs, key=lambda pair: pair["pair_seed"]),
                      pair_count=len(pairs), mean_bit_accuracy=statistics.mean(pair_means),
                      sd_across_pairs=(statistics.stdev(pair_means)
                                       if len(pair_means) > 1 else None)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/content/hybrid_eve_mlp_results.jsonl"))
    parser.add_argument("--conditions", nargs="+", choices=("cnn", "mlp", "hybrid"),
                        default=("cnn", "mlp", "hybrid"))
    parser.add_argument("--pair-seeds", nargs="+", type=int, default=(7, 8, 9))
    parser.add_argument("--attacker-seeds", nargs="+", type=int, default=(101, 102, 103))
    parser.add_argument("--target-sizes", nargs="+", type=int, default=(5_000, 80_000))
    parser.add_argument("--pair-steps", type=int, default=15_000)
    parser.add_argument("--attack-steps", type=int, default=20_000)
    parser.add_argument("--early-attack-step", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--total-eve-updates", type=int, default=4,
                        help="Total Eve updates per Alice/Bob step, shared round-robin")
    parser.add_argument("--bob-min", type=float, default=0.99)
    parser.add_argument("--privacy-weight", type=float, default=8.0)
    parser.add_argument("--ab-lr", type=float, default=1e-3)
    parser.add_argument("--eve-lr", type=float, default=1e-3)
    parser.add_argument("--attack-lr", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=int, default=16_384)
    parser.add_argument("--test-size", type=int, default=65_536)
    parser.add_argument("--eval-batch-size", type=int, default=2_048)
    parser.add_argument("--log-every", type=int, default=1_000)
    args = parser.parse_args()
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Conditions must be unique")
    if len(set(args.pair_seeds)) != len(args.pair_seeds):
        parser.error("Pair seeds must be unique")
    if len(set(args.attacker_seeds)) != len(args.attacker_seeds):
        parser.error("Attacker seeds must be unique")
    if len(set(args.target_sizes)) != len(args.target_sizes):
        parser.error("Target sizes must be unique")
    if not (0.5 < args.bob_min <= 1):
        parser.error("--bob-min must be above 0.5 and no greater than 1")
    if not 1 <= args.early_attack_step <= args.attack_steps:
        parser.error("Early attack step must be between 1 and attack steps")
    for name in ("pair_steps", "attack_steps", "batch_size", "total_eve_updates",
                 "validation_size", "test_size", "eval_batch_size", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(size < 1 for size in args.target_sizes):
        parser.error("Target sizes must be positive")
    if "hybrid" in args.conditions and args.total_eve_updates % 2:
        parser.error("--total-eve-updates must be even for the hybrid condition")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new --output")
    if not args.output.parent.exists():
        parser.error(f"Output directory does not exist: {args.output.parent}")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This study requires a CUDA GPU; request one via colab run --gpu A100")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    if count(CryptCNN()) != 40_609 or count(CNNEve()) != 39_757:
        raise AssertionError("Model sizes changed from the Week 3 baseline")
    if count(MLPEve(*MLP_STAGE_WIDTHS)) != 40_609:
        raise AssertionError("Stage-1 MLP size changed from the Week 3 baseline")
    print(f"DEVICE {torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    append_record(args.output, dict(type="config", conditions=args.conditions,
                  pair_seeds=args.pair_seeds, attacker_seeds=args.attacker_seeds,
                  target_sizes=args.target_sizes, pair_steps=args.pair_steps,
                  attack_steps=args.attack_steps, early_attack_step=args.early_attack_step,
                  batch_size=args.batch_size, total_eve_updates=args.total_eve_updates,
                  bob_min=args.bob_min, privacy_weight=args.privacy_weight,
                  ab_lr=args.ab_lr, eve_lr=args.eve_lr, attack_lr=args.attack_lr,
                  validation_size=args.validation_size, test_size=args.test_size,
                  gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__))
    for pair_seed in args.pair_seeds:
        for condition in args.conditions:
            alice, validation, test, reliable = train_pair(
                condition, pair_seed, args, device, args.output)
            if reliable:
                for target in args.target_sizes:
                    for seed in args.attacker_seeds:
                        attack(alice, validation, test, condition, pair_seed,
                               target, seed, args, device, args.output)
            else:
                print(f"SKIP_ATTACK condition={condition} pair={pair_seed} "
                      f"Bob failed the prespecified {args.bob_min:.1%} reliability threshold",
                      flush=True)
            del alice, validation, test
            torch.cuda.empty_cache()
    summarize(args.output, args.attack_steps)
    print(f"DONE results={args.output}", flush=True)


if __name__ == "__main__":
    main()
