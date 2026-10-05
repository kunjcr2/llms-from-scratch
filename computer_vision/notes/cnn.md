# 🧠 MNIST CNN Model (PyTorch)

A high-performing Convolutional Neural Network designed for **MNIST handwritten digit classification**.  
This model stacks multiple convolutional blocks with **BatchNorm**, **Dropout**, and **Adaptive Average Pooling** for stable and accurate learning.

---

## 🏗️ Architecture

```python
import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28x28 -> 14x14
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14x14 -> 7x7
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> (B, 256, 1, 1)
            nn.Flatten(),             # -> (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x
```

---

## 📊 Model Summary

| Layer (type)         | Output Shape  | Param # |
| -------------------- | ------------- | ------- |
| Conv2d (1→32)        | (32, 28, 28)  | 288     |
| BatchNorm2d (32)     | (32, 28, 28)  | 64      |
| Conv2d (32→64)       | (64, 28, 28)  | 18,432  |
| BatchNorm2d (64)     | (64, 28, 28)  | 128     |
| MaxPool2d            | (64, 14, 14)  | 0       |
| Dropout (0.25)       | -             | 0       |
| Conv2d (64→128)      | (128, 14, 14) | 73,728  |
| BatchNorm2d (128)    | (128, 14, 14) | 256     |
| Conv2d (128→256)     | (256, 14, 14) | 294,912 |
| BatchNorm2d (256)    | (256, 14, 14) | 512     |
| MaxPool2d            | (256, 7, 7)   | 0       |
| Dropout (0.25)       | -             | 0       |
| AdaptiveAvgPool2d(1) | (256, 1, 1)   | 0       |
| Flatten              | (256)         | 0       |
| Linear (256→128)     | (128)         | 32,768  |
| Linear (128→10)      | (10)          | 1,280   |
| **Total Params**     | **≈422,506**  |         |

---

## 🧩 Key Features

- **Batch Normalization** → stabilizes and accelerates training
- **Dropout** → prevents overfitting
- **Adaptive Average Pooling** → input-size independent
- **Compact yet high-capacity** → perfect for small datasets

---

## ⚡ Training Tip

Use `Adam` with a small learning rate for best performance:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

Trains to **>99% accuracy on MNIST** in ~10–15 epochs on a single GPU.

---

**Author:** Kunj Shah
**Framework:** PyTorch
**Dataset:** MNIST

---
