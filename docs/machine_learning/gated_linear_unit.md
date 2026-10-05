# 🚪 Gated Linear Units (GLUs) in Modern Models

GLUs are like **smart doors** 🛑🚪 that decide **how much information passes through** in a neural network.
Instead of pushing every token through heavy computation, GLUs decide:

* ✅ which tokens go deep,
* ❌ which ones can skip,
* 🔄 how much of each token to keep.

---

## 🔑 Intuition

* Split the input into two parts:

  * **A = information**
  * **B = gate (decision maker)**
* Formula:

  $$
  GLU(X) = A \odot \sigma(B)
  $$

  where `σ` is a sigmoid (makes values between 0 and 1).

So:

* If gate ≈ 1 → let token info pass.
* If gate ≈ 0 → block token info.
* If gate ≈ 0.5 → pass half.

---

## 🖼️ Token Flow with GLU (Mermaid Diagram)

```mermaid
flowchart LR
    A[Token Input] --> B[Linear Transform → A]
    A --> C[Linear Transform → B[Gate]]
    C -->|Sigmoid| D[Gate Values 0-1]
    B --> E[Element-wise Multiply]
    D --> E
    E --> F[Filtered Output]
```

---

## 🧩 Why in Mixture-of-Depths (MoD) or HRMs?

These models want **efficiency**: not all tokens need heavy compute.

* GLU **scores tokens** → decides which go deeper.
* Saves FLOPs 💻, speeds up inference ⚡, and still keeps accuracy.

Think of it as:

* "VIP tokens" get full processing.
* "Background tokens" just pass quickly.

---

## 🐍 Minimal PyTorch Example

```python
import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 2*dim)  # split into A and B

    def forward(self, x):
        a, b = self.linear(x).chunk(2, dim=-1)  # split
        return a * torch.sigmoid(b)  # apply gate
```

Usage in MoD/HRM: this gate decides **if a token goes deeper or not**.

---

## 📚 Links to Learn More

* 🔗 [GLU Paper (Dauphin et al., 2017)](https://arxiv.org/abs/1612.08083)
* 🔗 [Mixture of Depths (Meta AI, 2023)](https://arxiv.org/abs/2308.14711)
* 🔗 [HRMs (Hierarchical Residual Mixtures)](https://arxiv.org/abs/2404.02258)

---

## 🎥 Videos

* [Yannic Kilcher — Mixture of Depths Explained](https://www.youtube.com/watch?v=HlXB8YqF6tM)
* [GLU Intuition (fastai lecture clip)](https://www.youtube.com/watch?v=V_xro1bcAuA)

---

✅ **Simple takeaway**:
GLU = a *smart filter* that helps models like MoD/HRMs **decide which tokens deserve full attention** and which can be skipped.
