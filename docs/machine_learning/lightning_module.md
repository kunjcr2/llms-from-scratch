# PyTorch Lightning: Quick Start (for a \~50M param model or smaller)

## 📚 Docs & Tutorials

- Official docs (stable): [https://lightning.ai/docs/pytorch/stable/](https://lightning.ai/docs/pytorch/stable/) ([Lightning AI][1])
- Trainer API (what you’ll use most): [https://lightning.ai/docs/pytorch/stable/common/trainer.html](https://lightning.ai/docs/pytorch/stable/common/trainer.html) ([Lightning AI][2])
- “Lightning in 15 minutes” quickstart: [https://lightning.ai/docs/pytorch/stable/starter/introduction.html](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) ([Lightning AI][3])
- Tutorials index (DataModules, CIFAR, etc.): [https://lightning.ai/docs/pytorch/stable/tutorials.html](https://lightning.ai/docs/pytorch/stable/tutorials.html) ([Lightning AI][4])

### 🎥 Relevant Videos

- Lightning AI (official) playlists: [https://www.youtube.com/@PyTorchLightning/playlists](https://www.youtube.com/@PyTorchLightning/playlists) ([YouTube][5])
- PyTorch Lightning tutorials playlist (community): [https://www.youtube.com/playlist?list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP](https://www.youtube.com/playlist?list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP) ([YouTube][6])
- “Into Generative AI with PyTorch Lightning 2.0” (talk): [https://www.youtube.com/watch?v=pfdeWgNup2Y](https://www.youtube.com/watch?v=pfdeWgNup2Y) ([YouTube][7])

---

## ✅ Install & Import

```bash
# CPU-only (ok for dev)
pip install "pytorch-lightning>=2.0" torch torchvision torchtext

# GPU (CUDA wheels vary by CUDA version; adjust as needed)
# Example for CUDA 12.x:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install "pytorch-lightning>=2.0"
```

```python
# imports you’ll use 90% of the time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

import lightning as L  # PyTorch Lightning 2.x uses the 'lightning' package
from lightning.pytorch.loggers import WandbLogger  # optional
```

---

## 🧩 Minimal Lightning Example (classification)

A tiny example you can scale up later (swap the model with your \~50M param net).

```python
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning as L

# ----- Toy Dataset (replace with real) -----
def make_toy_dataset(n=10_000, d=32):
    X = torch.randn(n, d)
    y = (X.mean(dim=1) > 0).long()
    return TensorDataset(X, y)

# ----- LightningModule -----
class TinyClassifier(L.LightningModule):
    def __init__(self, in_dim=32, hidden=128, n_classes=2, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
        self.crit = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.crit(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss  = self.crit(logits, y)
        acc   = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# ----- DataModule (optional but clean) -----
class ToyDataModule(L.LightningDataModule):
    def __init__(self, batch_size=256):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        ds = make_toy_dataset()
        n = len(ds); n_train = int(0.8 * n)
        self.train_ds, self.val_ds = random_split(ds, [n_train, n - n_train])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

if __name__ == "__main__":
    L.seed_everything(42)
    model = TinyClassifier(in_dim=32, hidden=512)  # bump hidden to ~50M with deeper/wider nets later
    dm = ToyDataModule(batch_size=512)

    # Optional: W&B logging (remove if not using)
    # from lightning.pytorch.loggers import WandbLogger
    # logger = WandbLogger(project="lightning-quickstart")

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=10,
        # logger=logger,
    )
    trainer.fit(model, datamodule=dm)
```

---

## 🚀 Train It (real-world pattern)

### Option A — Local quick run

```bash
python train.py
```

### Option B — With mixed precision + deterministic seed

```bash
python train.py --precision 16-mixed --seed 42
```

### Option C — Multi-GPU (DDP) when you’re ready

```bash
python -m torch.distributed.run --nproc_per_node=2 train.py
# Or let Lightning handle it:
# L.Trainer(accelerator="gpu", devices=2, strategy="ddp")
```

---

## 📝 Notes for Scaling Up (\~50M params)

- Start with this template, then **swap `TinyClassifier`** for your actual model (e.g., a Transformer).
- Keep **mixed precision** on (`precision="16-mixed"`) to fit bigger models.
- Use **checkpointing & early stopping** via Lightning callbacks once you move beyond toy data.
- Trainer docs for knobs you’ll tune (epochs, strategy, precision, callbacks): see Trainer API. ([Lightning AI][2])

---

[1]: https://lightning.ai/docs/pytorch/stable/?utm_source=chatgpt.com "Welcome to ⚡ PyTorch Lightning — PyTorch Lightning 2.5.4 documentation"
[2]: https://lightning.ai/docs/pytorch/stable/common/trainer.html?utm_source=chatgpt.com "Trainer — PyTorch Lightning 2.5.4 documentation"
[3]: https://lightning.ai/docs/pytorch/stable/starter/introduction.html?utm_source=chatgpt.com "Lightning in 15 minutes — PyTorch Lightning 2.5.4 documentation"
[4]: https://lightning.ai/docs/pytorch/stable/tutorials.html?utm_source=chatgpt.com "PyTorch Lightning Tutorials — PyTorch Lightning 2.5.4 documentation"
[5]: https://www.youtube.com/%40PyTorchLightning/playlists?utm_source=chatgpt.com "Lightning AI - YouTube"
[6]: https://www.youtube.com/playlist?list=PLhhyoLH6IjfyL740PTuXef4TstxAK6nGP&utm_source=chatgpt.com "PyTorch Lightning Tutorials - YouTube"
[7]: https://www.youtube.com/watch?v=pfdeWgNup2Y&utm_source=chatgpt.com "Into Generative AI with PyTorch Lightning 2.0 - YouTube"
