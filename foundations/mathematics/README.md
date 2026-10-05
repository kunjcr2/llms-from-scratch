# 🧮 Math for Machine Learning — From Scratch to R&D

> **Goal**: Build deep, intuitive, and practical understanding of the three mathematical pillars of AI/ML — starting from first principles and connecting every concept to what happens inside real models.

---

## Why Math for ML?

You don't need math to *use* PyTorch. But you need it to:

- **Debug** training failures (why is my loss exploding? why is it stuck?)
- **Design** new architectures or losses with intention
- **Read** research papers without drowning
- **Tune** hyperparameters with understanding rather than guessing
- **Innovate** in R&D — you can't build new things from a black box

Math is the *language* ML is written in. These notes teach you to speak it fluently.

---

## 📁 Structure

| Folder | Topics | Key ML Connections |
|--------|--------|-------------------|
| [`01_statistics_probability/`](./01_statistics_probability/README.md) | Distributions, Bayes, MLE, Regression | Loss functions, model assumptions, probabilistic inference |
| [`02_linear_algebra/`](./02_linear_algebra/README.md) | Vectors, Matrices, Eigenvalues, SVD, PCA | Data representation, weight matrices, dimensionality reduction |
| [`03_calculus/`](./03_calculus/README.md) | Derivatives, Gradients, Chain Rule, Optimization | Backpropagation, gradient descent, loss landscapes |
| [`04_second_order_ml/`](./04_second_order_ml/README.md) | Jacobians, Hessians, Taylor expansions, EWC, JVPs | Continual learning, curvature, second-order approximations |

---

## 🗺️ Recommended Learning Path

```
Statistics & Probability  ──►  Linear Algebra  ──►  Calculus  ──►  Second-Order ML
        │                            │                   │                 │
   Understanding data           Data as vectors     Training models   Curvature, local
   and distributions            and transformations  via optimization  approximations, EWC
```

You can go through each section independently, but this order builds the best foundation.

---

## 📦 What's Inside Each Section

Each section contains:

- **`README.md`** — Full theoretical coverage with:
  - Intuitive explanations in plain English
  - Rigorous LaTeX math
  - Visual intuitions and analogies
  - Explicit ML/DL connections for every concept

- **`.ipynb` notebook** — Hands-on code with:
  - From-scratch implementations in `numpy`
  - Visualizations with `matplotlib`
  - Real ML examples with `sklearn`, `torch`
  - Exercises and experiments

---

## 🔧 Requirements

```bash
pip install numpy matplotlib pandas scipy scikit-learn torch
```

---

## 📖 References & Going Deeper

- [Mathematics for Machine Learning (Deisenroth et al.)](https://mml-book.github.io/) — Free PDF
- [Introduction to Statistical Learning](https://www.statlearning.com/) — Free PDF  
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability)
