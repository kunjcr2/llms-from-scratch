from pathlib import Path

import matplotlib.pyplot as plt

methods = [
    "Weighted SFT",
    "SFT (Full)",
    "SFT",
    "RFT → GRPO @ 300",
    "RFT",
    "Base",
    "GRPO",
    "GSPO @ 200",
    "Dr. GRPO @ 200",
]

# Approximate values from the report are represented by their displayed estimate.
accuracy = [24.0, 30.0, 32.30, 36.54, 37.4, 42.08, 45.03, 45.72, 46.02]
labels = ["24.26%", "27.44%", "32.30%", "36.54%", "37.4%", "42.08%", "45.03%", "45.72%", "46.02%"]

colors = ["#7f8c8d"] * len(methods)
colors[6] = "#0d7ee7"       # GRPO
colors[7] = "#0d7ee7"       # GSPO @ 200
colors[8] = "#0d7ee7"       # Dr. GRPO @ 200

fig, ax = plt.subplots(figsize=(16, 9))
x = range(len(methods))
bars = ax.bar(x, accuracy, color=colors)

ax.set_xticks(x, methods, rotation=45, ha="right")
ax.tick_params(axis="x", labelsize=12, pad=8)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylim(0, 50)
ax.set_ylabel("Accuracy (%)", fontsize=20, labelpad=14)
ax.set_title("Qwen2.5-0.5B Post-Training on GSM8K (LoRA, unless mentioned)", loc="center", pad=24, fontsize=24, weight="bold")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

for bar, label in zip(bars, labels):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, label, ha="center", va="bottom", fontsize=16)

ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Full GSM8K test split · 1,319 problems · greedy decoding · exact numeric-answer accuracy", fontsize=12, color="dimgray")
plt.tight_layout()
output_dir = Path(__file__).resolve().parent
plt.savefig(output_dir / "qwen25-05b-gsm8k-results.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "qwen25-05b-gsm8k-results.svg", bbox_inches="tight")
