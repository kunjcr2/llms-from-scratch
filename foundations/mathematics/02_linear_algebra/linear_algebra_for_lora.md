# Linear Algebra for Deep Learning — Session Notes
### Kunj Shah | A10 Networks Prep | April 2026

---

## 1. Vectors

### What is a vector?
A list of numbers representing a point or direction in space.

```
v = [2, 3]  →  go 2 units right, 3 units up
```

In DL — every token is a vector:
```
"king" = [0.2, -0.4, 0.8, ..., 0.1]   # 768 numbers in BERT
```

### Norms — measuring size of a vector

| Norm | Formula | Intuition | DL Use |
|------|---------|-----------|--------|
| L1 | $\sum |v_i|$ | Manhattan distance — grid walking | Sparsity, pruning |
| L2 | $\sqrt{\sum v_i^2}$ | Straight line distance | Gradient clipping, weight decay |
| L∞ | $\max(|v_i|)$ | Largest absolute value | Adversarial robustness |

**Example:**
$$v = [3, -4]$$
$$\|v\|_1 = |3| + |-4| = 7$$
$$\|v\|_2 = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$$
$$\|v\|_\infty = \max(3, 4) = 4$$

### Gradient Clipping — why L2 matters

During training, gradients can explode. Fix:

$$g \leftarrow g \times \frac{\text{max\_norm}}{\|g\|}$$

**Example:**
$$g = [3, 4], \quad \|g\| = 5, \quad \text{max\_norm} = 1.0$$
$$g_{\text{clipped}} = [0.6, 0.8], \quad \|g_{\text{clipped}}\| = 1.0$$

Direction preserved. Only magnitude shrinks.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### L2 Regularization vs L2 Loss

**L2 Loss (MSE)** — measures how wrong predictions are:
$$L = \frac{1}{n} \sum (y_{\text{pred}} - y_{\text{true}})^2$$

**L2 Regularization (Weight Decay)** — penalizes large weights:
$$L_{\text{total}} = L_{\text{task}} + \lambda \|w\|_2^2$$

In code:
```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
# weight_decay = lambda
```

---

## 2. Dot Product, Cosine Similarity, Orthogonality

### Dot Product

**Mechanical:**
$$a \cdot b = a_1 b_1 + a_2 b_2 + \ldots + a_n b_n$$

**Example:**
$$a = [1, 2, 3], \quad b = [4, 5, 6]$$
$$a \cdot b = 4 + 10 + 18 = 32$$

**Geometric:**
$$a \cdot b = \|a\| \cdot \|b\| \cdot \cos(\theta)$$

| Angle | cos(θ) | Meaning |
|-------|--------|---------|
| 0° | 1 | Fully aligned |
| 90° | 0 | Perpendicular |
| 180° | -1 | Opposite directions |

**In attention:**
$$\text{score} = Q \cdot K^T$$
High score = query and key aligned = attend more to that token.

### Cosine Similarity

Raw dot product depends on magnitude. Fix — normalize:

$$\text{cosine\_sim}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|} = \cos(\theta)$$

Range: $[-1, 1]$. Pure direction, ignores magnitude.

**In LLMs:**
```
cosine("king", "queen") → high  ✓
cosine("king", "apple") → low   ✓
```
This is how vector databases find similar embeddings.

### Orthogonality

$$a \cdot b = 0 \implies \text{vectors are perpendicular (orthogonal)}$$

**Orthogonal matrix Q:**
$$Q^T Q = I$$

Every column is orthogonal to every other column AND has unit length (orthonormal).

**Key property:** Orthogonal matrices ONLY rotate — they never stretch or squish. Lengths and angles preserved.

**Example — 45° rotation:**
$$Q = \begin{bmatrix} 0.707 & -0.707 \\ 0.707 & 0.707 \end{bmatrix}$$

Check:
$$\text{col}_1 \cdot \text{col}_2 = (0.707)(-0.707) + (0.707)(0.707) = 0 \checkmark$$
$$\|\text{col}_1\| = \sqrt{0.707^2 + 0.707^2} = 1 \checkmark$$

**Why this matters:** In SVD, $U$ and $V^T$ are orthogonal matrices — pure rotation, no distortion.

---

## 3. Matrices

### What is a matrix?
A grid of numbers. More importantly — **a transformation.**

$$A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}, \quad v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$Av = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

Stretched 2× in x, 3× in y. Every linear layer in a neural network is this:
$$\text{output} = W \times \text{input}$$

### Rank — the most important concept

**Rank = number of linearly independent rows/columns = actual dimensionality of information.**

**Example of rank-deficient matrix:**
$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

Row 2 = 2 × Row 1. Only 1 unique direction. **Rank = 1.**

**Why rank matters for LoRA:**

Full fine-tuning:
$$\Delta W \text{ is } 4096 \times 4096 = 16\text{M parameters}$$

LoRA exploits that $\Delta W$ is approximately low rank:
$$\Delta W \approx B \times A$$
$$B: 4096 \times r, \quad A: r \times 4096, \quad r = 8$$
$$\text{Parameters} = 4096 \times 8 + 8 \times 4096 = 65{,}536 \quad (256\times \text{ fewer})$$

Information passes through a bottleneck of size $r$ → result can only have rank $r$.

---

## 4. Eigenvalues and Eigenvectors

### The core question

For a matrix $A$, is there a vector that **only gets stretched, never rotated?**

$$A \mathbf{v} = \lambda \mathbf{v}$$

- $\mathbf{v}$ = eigenvector (special direction)
- $\lambda$ = eigenvalue (how much it stretches)

### How to find eigenvalues

**Step 1:** Rearrange the equation:
$$(A - \lambda I)\mathbf{v} = 0$$

**Step 2:** For non-zero $\mathbf{v}$, matrix must be singular:
$$\det(A - \lambda I) = 0$$

**Step 3:** Solve the polynomial.

**Example:**
$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

$$\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 2 = 0$$

$$\lambda^2 - 7\lambda + 10 = 0 \implies (\lambda - 5)(\lambda - 2) = 0$$

$$\boxed{\lambda_1 = 5, \quad \lambda_2 = 2}$$

### How to find eigenvectors

For each eigenvalue, solve $(A - \lambda I)\mathbf{v} = 0$:

**For $\lambda_1 = 5$:**
$$\begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \mathbf{v} = 0 \implies v_1 = v_2 \implies \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**For $\lambda_2 = 2$:**
$$\begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix} \mathbf{v} = 0 \implies v_2 = -2v_1 \implies \mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$

### Eigendecomposition

Stack eigenvectors as columns → $Q$. Eigenvalues on diagonal → $\Lambda$:

$$A = Q \Lambda Q^{-1}$$

$$Q = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$$

**Intuition:** Any matrix = rotate → stretch → rotate back.

### What eigenvalues tell you

| Eigenvalue | Meaning |
|------------|---------|
| $\lambda > 1$ | Stretches that direction |
| $\lambda < 1$ | Shrinks that direction |
| $\lambda = 0$ | Collapses that direction — information lost |
| $\lambda < 0$ | Flips direction |

### DL uses of eigenvalues

**1. Gradient explosion/vanishing:**
$$A^n = Q \Lambda^n Q^{-1}$$
If any $\lambda > 1$ → raises to power $n$ → explodes. This is the actual math behind gradient explosion.

**2. Loss landscape — Hessian:**
$$H = \frac{d^2 L}{dw^2}$$
Large eigenvalues of $H$ = sharp curvature = sensitive directions. Adam approximates this.

**3. LoRA rank selection:**
```python
U, S, Vt = torch.linalg.svd(delta_W)
plt.plot(S.cpu().numpy())
# where curve drops off → choose rank r there
```

### Limitation of eigendecomposition

**Only works on square matrices.**

Most weight matrices in transformers can be rectangular. → Need SVD.

---

## 5. SVD — Singular Value Decomposition

### The fix for eigendecomposition's limitation

Works on **any** matrix. Any shape.

$$A = U \Sigma V^T$$

For a matrix $A$ of shape $m \times n$:

| Matrix | Shape | What it is |
|--------|-------|------------|
| $U$ | $m \times m$ | Left singular vectors — orthogonal (rotation) |
| $\Sigma$ | $m \times n$ | Singular values on diagonal — always positive, sorted |
| $V^T$ | $n \times n$ | Right singular vectors — orthogonal (rotation) |

**Same story as eigendecomposition:**
$$V^T \rightarrow \text{rotate input} \rightarrow \Sigma \rightarrow \text{stretch} \rightarrow U \rightarrow \text{rotate output}$$

Any matrix = rotate → stretch → rotate.

### Singular values

$$\Sigma = \begin{bmatrix} \sigma_1 & 0 \\ 0 & \sigma_2 \\ 0 & 0 \end{bmatrix}, \quad \sigma_1 \geq \sigma_2 \geq 0$$

Large $\sigma$ → that direction carries a lot of information.
Small $\sigma$ → almost nothing.
Zero $\sigma$ → completely useless.

**Connection to eigenvalues:**
$$\sigma_i = \sqrt{\lambda_i(A^T A)}$$

SVD is eigendecomposition applied to $A^T A$.

### Rank-r approximation

Keep only top $r$ singular values:

$$A \approx \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

**Example:**
```python
A = torch.tensor([[3., 2.], [2., 3.], [1., 1.]])
U, S, Vt = torch.linalg.svd(A, full_matrices=False)
# S = [5.47, 1.00]
# Rank-1 captures 5.47/(5.47+1.00) = 84% of information
```

---

## 6. LoRA — Full Story from First Principles

### The problem

Fine-tuning a 7B parameter LLM:
- Storing gradients = 28GB
- Adam optimizer states = 56GB extra
- Not feasible for most setups

### The key observation

Researchers found: the weight update matrix $\Delta W$ during fine-tuning is **approximately low rank**.

Plot of singular values of $\Delta W$:
```
σ values:
|█
| █
|  █
|   ██
|     ████████████  ← near zero, useless
└──────────────────
```

Out of 4096 directions, only ~8 carry meaningful updates.

### The math

Instead of learning full $\Delta W$:
$$\Delta W \approx BA$$
$$B: d \times r, \quad A: r \times d, \quad r \ll d$$

Information bottleneck of size $r$ → result has rank $r$.

### Architecture

```
Input x (8, 5, 4096)
        │
        ├─────────────────────────┐
        │                         │
        ▼                         ▼
   W_q (frozen)              A: 4096×r
   4096×4096                      │
        │                    bottleneck (r)
        │                         │
        │                    B: r×4096
        │                         │
        └──────────┬──────────────┘
                   ▼
            output = Wx + BAx
```

Both paths take original input $x$. Results are **added**, not chained.

### Initialization

$$A \sim \mathcal{N}(0, 1), \quad B = 0$$

At start: $BA = 0$. Model starts exactly as pretrained. No noise added. Training begins stably.

### Scaling

$$\text{output} = Wx + \frac{\alpha}{r} BAx$$

$\alpha$ controls update magnitude. Dividing by $r$ keeps scale consistent regardless of rank chosen.

### During training vs inference

**Training — always separate:**
$$x \xrightarrow{A \ (4096 \times r)} (r,) \xrightarrow{B \ (r \times 4096)} (4096,)$$

Two separate matrix multiplications. A and B updated independently via gradients.

**Inference — merge once:**
$$W_{\text{new}} = W + \frac{\alpha}{r} BA$$

Single matrix. Zero extra latency.

### Why SVD proves LoRA is optimal

By the Eckart-Young theorem — the best rank-$r$ approximation to any matrix in Frobenius norm is given by keeping the top $r$ singular values in SVD.

$$\Delta W \approx U_r \Sigma_r V_r^T = BA$$

No other rank-$r$ matrix is closer to $\Delta W$ than $BA$. SVD mathematically guarantees LoRA is the optimal low-rank approximation.

### Parameter count

| Method | Parameters | Memory |
|--------|-----------|--------|
| Full fine-tuning | $4096 \times 4096 = 16.7\text{M}$ | ~67MB per layer |
| LoRA $r=8$ | $2 \times 4096 \times 8 = 65\text{K}$ | ~262KB per layer |
| Reduction | **256×** | **256×** |

---

## 7. Quick Reference — Key Formulas

$$\text{Dot product:} \quad a \cdot b = \sum a_i b_i = \|a\|\|b\|\cos\theta$$

$$\text{Cosine similarity:} \quad \frac{a \cdot b}{\|a\|\|b\|}$$

$$\text{L2 norm:} \quad \|v\| = \sqrt{\sum v_i^2}$$

$$\text{Eigenvalue equation:} \quad Av = \lambda v$$

$$\text{Find eigenvalues:} \quad \det(A - \lambda I) = 0$$

$$\text{Eigendecomposition:} \quad A = Q\Lambda Q^{-1}$$

$$\text{SVD:} \quad A = U\Sigma V^T$$

$$\text{LoRA forward:} \quad \text{output} = Wx + \frac{\alpha}{r}BAx$$

$$\text{Gradient clipping:} \quad g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|}$$

$$\text{Weight decay:} \quad L_{\text{total}} = L_{\text{task}} + \lambda\|w\|_2^2$$

---

*Next session: Optimization — Adam, AdamW, LR schedules, and Probability & Stats*