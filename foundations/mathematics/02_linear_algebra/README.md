# 🔢 Linear Algebra for Machine Learning

> *"The data is the message. Linear algebra is the language."*

Linear algebra is the language of data. Data comes in tables, images, sequences — all of which are naturally represented as vectors and matrices. Understanding how to work with these objects, and what operations on them mean geometrically, is the foundation for almost all quantitative work.

---

## Table of Contents

1. [Scalars, Vectors, Matrices, Tensors](#1-scalars-vectors-matrices-tensors)
2. [Vector Operations — Geometry of Data](#2-vector-operations--geometry-of-data)
3. [Matrix Operations](#3-matrix-operations)
4. [The Dot Product & Linear Transformations](#4-the-dot-product--linear-transformations)
5. [Norms — Measuring Size](#5-norms--measuring-size)
6. [Systems of Linear Equations](#6-systems-of-linear-equations)
7. [Determinants](#7-determinants)
8. [Matrix Inverses](#8-matrix-inverses)
9. [Matrix Rank & Linear Independence](#9-matrix-rank--linear-independence)
10. [Eigenvalues & Eigenvectors](#10-eigenvalues--eigenvectors)
11. [Singular Value Decomposition (SVD)](#11-singular-value-decomposition-svd)
12. [Principal Component Analysis (PCA)](#12-principal-component-analysis-pca)

---

## 1. Scalars, Vectors, Matrices, Tensors

### The Hierarchy

| Object | Dimensions | Example |
|--------|-----------|--------|
| **Scalar** $a \in \mathbb{R}$ | 0D | A temperature reading, a price |
| **Vector** $\mathbf{v} \in \mathbb{R}^n$ | 1D | A person's measurements, a data row |
| **Matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$ | 2D | A spreadsheet, a greyscale image |
| **Tensor** $\mathcal{T} \in \mathbb{R}^{d_1 \times \ldots \times d_k}$ | $k$D | A colour image (height × width × 3) |

### Vectors as Points and Directions

A vector $\mathbf{v} = [v_1, v_2, \ldots, v_n]^T$ can mean two things:
- A **point** in $n$-dimensional space
- A **direction** with magnitude (an arrow pointing somewhere)

Think of it this way: if someone gives you directions ("go 3 blocks east, 2 blocks north"), that's a vector-as-direction. If they say "meet me at coordinates (3, 2)", that's a vector-as-point. The math is the same; the interpretation depends on context.

---

## 2. Vector Operations — Geometry of Data

### Addition & Scalar Multiplication

$$\mathbf{u} + \mathbf{v} = [u_1+v_1, u_2+v_2, \ldots]^T$$
$$c\mathbf{v} = [cv_1, cv_2, \ldots]^T$$

Geometrically: addition is "tip-to-tail" — place the tail of $\mathbf{v}$ at the tip of $\mathbf{u}$, and you land at $\mathbf{u} + \mathbf{v}$. Scalar multiplication stretches the arrow ($c > 1$), shrinks it ($0 < c < 1$), or flips it ($c < 0$).

### Linear Combinations & Span

A **linear combination** of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$:

$$\mathbf{u} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

The **span** is the set of all possible linear combinations — every point you can reach by mixing those vectors with any weights.

**Intuition**: Think of $\mathbf{v}_1$ and $\mathbf{v}_2$ as two directions you're allowed to move. If you can move any amount in each direction, the span is everything you can possibly reach. Two non-parallel vectors in 3D don't span all of 3D — they only span a flat plane through the origin. Add a third vector pointing out of that plane, and now you can reach anywhere in 3D. The span "fills up" more of space as you add more independent directions.

---

## 3. Matrix Operations

### Transpose

$$(\mathbf{A}^T)_{ij} = A_{ji}$$

Flips rows and columns. If $\mathbf{A}$ is $m \times n$, then $\mathbf{A}^T$ is $n \times m$. Think of it as rotating the matrix 90° along its diagonal. Crucial for: weight matrices, attention, convolutions.

### Matrix Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:

$$(\mathbf{AB})_{ij} = \sum_{l=1}^k A_{il} B_{lj}$$

The $(i,j)$ entry of $\mathbf{AB}$ is the **dot product** of row $i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$.

**Intuition**: Matrix multiplication is function composition. If $\mathbf{B}$ transforms space one way, and $\mathbf{A}$ transforms it another way, then $\mathbf{AB}$ is "do $\mathbf{B}$ first, then $\mathbf{A}$." This is why order matters — doing two transformations in the wrong order gives a different result, just like rotating then reflecting is different from reflecting then rotating.

**NOT commutative**: $\mathbf{AB} \neq \mathbf{BA}$ in general.

### Key Properties

$$(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T \quad \text{(order reverses)}$$
$$(\mathbf{ABC})^T = \mathbf{C}^T \mathbf{B}^T \mathbf{A}^T$$

The reversal makes sense dimensionally: if $\mathbf{AB}$ is $m \times n$, then $(\mathbf{AB})^T$ must be $n \times m$. And $\mathbf{B}^T\mathbf{A}^T$ has shape $(n \times k)(k \times m) = n \times m$. ✓

---

## 4. The Dot Product & Linear Transformations

### Dot Product

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

Where $\theta$ is the angle between the vectors.

- $\mathbf{u} \cdot \mathbf{v} > 0$: vectors point in similar directions
- $\mathbf{u} \cdot \mathbf{v} = 0$: **orthogonal** (perpendicular)
- $\mathbf{u} \cdot \mathbf{v} < 0$: vectors point in opposite directions

**Intuition**: The dot product measures "how much of $\mathbf{u}$ is going in the direction of $\mathbf{v}$". Imagine projecting one vector onto the other — the dot product is the length of that shadow, scaled by the other vector's length. If the two vectors are perpendicular, there's no shadow at all → dot product is zero.

### A Matrix as a Transformation

A matrix $\mathbf{A}$ **transforms** input vectors to output vectors. It can:
- Rotate
- Scale
- Reflect
- Project
- Any combination

**Intuition**: To understand what any matrix does, watch what it does to the unit axes. $\mathbf{A}\mathbf{e}_1$ tells you where the first basis vector lands. $\mathbf{A}\mathbf{e}_2$ tells you where the second lands. The columns of $\mathbf{A}$ are exactly the new locations of the original axes. Every vector in space gets carried along for the ride — it's like a stretchy, rotatable grid.

A sequence of matrix operations is a sequence of transformations of space. $\mathbf{ABC}\mathbf{x}$ means: first transform by $\mathbf{C}$, then $\mathbf{B}$, then $\mathbf{A}$.

### Cosine Similarity

Normalising the dot product gives a measure of **angular similarity** between two vectors:

$$\cos(\theta) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

This removes magnitude from the comparison — only the **direction** matters. Two vectors can have completely different magnitudes but point in the same direction — cosine similarity tells you they're identical in direction. That's why it's used to compare text embeddings: the length of the embedding vector is arbitrary, but the direction encodes meaning.

---

## 5. Norms — Measuring Size

A norm is just a way of measuring "how big" a vector is. Different norms measure "bigness" differently.

### L2 Norm (Euclidean Length)

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2} = \sqrt{\mathbf{v}^T \mathbf{v}}$$

The straight-line distance from the origin to the point $\mathbf{v}$. This is what you mean by "length" in everyday life. It penalizes large components **quadratically** — a single large value dominates.

### L1 Norm

$$\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$$

The "taxicab" or "Manhattan" distance — how far you'd travel if you could only move along grid lines (no diagonals). Unlike L2, it treats each component equally — a vector with ten moderate values costs as much as a vector with one large value. This property makes L1 **sparsity-promoting**: in optimization, minimizing L1 norm tends to push most components to exactly zero, leaving only a few non-zero ones.

**L1 vs L2 in practice:**
- Use **L2** when you want to shrink all weights smoothly (ridge regression, weight decay)
- Use **L1** when you want sparse solutions — most weights exactly zero (LASSO, sparse coding)

### General $L_p$ Norm

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p \right)^{1/p}$$

As $p$ increases, the norm becomes increasingly dominated by the **largest component**. At $p = \infty$, the L∞ norm is just $\max_i |v_i|$ — only the biggest element matters.

### Frobenius Norm (for Matrices)

$$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$$

The matrix equivalent of the L2 norm — just treat all entries as a single long vector and take its length. It measures the "total energy" of a matrix. When you take the SVD, the Frobenius norm relates cleanly to the singular values: $\|\mathbf{A}\|_F = \sqrt{\sum_i \sigma_i^2}$.

---

## 6. Systems of Linear Equations

### The Problem

$$\mathbf{Ax} = \mathbf{b}$$

Where $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

You're asking: "What vector $\mathbf{x}$ does $\mathbf{A}$ transform into $\mathbf{b}$?"

**Intuition**: Each row of $\mathbf{Ax} = \mathbf{b}$ is one constraint — a hyperplane (line in 2D, plane in 3D, etc.) in $\mathbf{x}$-space. Your solution $\mathbf{x}$ must lie on *all* of these hyperplanes simultaneously.

Three possible outcomes:

1. **Unique solution** ($m = n$, $\mathbf{A}$ is invertible): All the hyperplanes intersect at exactly one point. $\mathbf{b}$ is in the column space of $\mathbf{A}$, and $\mathbf{A}$'s columns are independent enough to "reach" it from one direction only.

2. **No solution** ($\mathbf{b}$ not in the column space of $\mathbf{A}$) — *overdetermined*: You have more equations than unknowns. Think of fitting a line through 100 points — no line passes through all of them exactly. The hyperplanes don't all share a common intersection. The best you can do is find the $\mathbf{x}$ that minimizes leftover error (least squares).

3. **Infinitely many solutions** — *underdetermined*: You have fewer equations than unknowns, or some equations are redundant. Some directions in $\mathbf{x}$-space are completely unconstrained. Imagine one equation in 3 unknowns — it defines a plane, and every point on that plane is a valid solution.

---

## 7. Determinants

The determinant $\det(\mathbf{A})$ measures the **signed volume scaling factor** of the transformation $\mathbf{A}$.

**Intuition**: Take the unit square in 2D (area = 1). Apply matrix $\mathbf{A}$ to every point in it. The resulting parallelogram has area $|\det(\mathbf{A})|$. The sign tells you whether the orientation was preserved (positive) or flipped (negative — like a mirror reflection).

For a $2\times2$ matrix:

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

Think of the two columns as vectors $\begin{pmatrix}a\\c\end{pmatrix}$ and $\begin{pmatrix}b\\d\end{pmatrix}$. The determinant is the area of the parallelogram they form.

### Key Properties

- $\det(\mathbf{A}) = 0$: Matrix is **singular** (collapses space, not invertible). The transformation squashes all of space down into a lower-dimensional subspace — like folding a 3D cube flat onto a piece of paper. Once collapsed, you can't undo it. There's no inverse.
- $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$: Two transformations together scale volume by the product of their individual scaling factors.
- $\det(\mathbf{A}^T) = \det(\mathbf{A})$: Transposing doesn't change volume scaling.
- $\det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A})$: If $\mathbf{A}$ doubles volume, $\mathbf{A}^{-1}$ halves it.

---

## 8. Matrix Inverses

For a square matrix $\mathbf{A}$, the inverse $\mathbf{A}^{-1}$ **undoes** the transformation $\mathbf{A}$:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

**Intuition**: If $\mathbf{A}$ rotates space 30° clockwise, then $\mathbf{A}^{-1}$ rotates it 30° counterclockwise. If $\mathbf{A}$ stretches in some direction, $\mathbf{A}^{-1}$ squashes it back. The inverse literally reverses everything $\mathbf{A}$ did.

Inverse exists **iff** $\det(\mathbf{A}) \neq 0$ — because if the determinant is zero, $\mathbf{A}$ collapsed space into fewer dimensions, and there's no way to un-collapse it (you lost information).

### Pseudoinverse (Moore-Penrose)

When $\mathbf{A}$ is not square (you can't have a true inverse) or is singular (true inverse doesn't exist):

$$\mathbf{A}^+ = \mathbf{V}\mathbf{\Sigma}^+ \mathbf{U}^T \quad \text{(from SVD)}$$

**Intuition**: The pseudoinverse is the best possible attempt at an inverse when a true inverse doesn't exist. It gives you two guarantees:
- Among all $\mathbf{x}$ that minimize $\|\mathbf{Ax} - \mathbf{b}\|_2$ (**least-squares** — as close to a solution as possible), it picks the one with the **smallest norm** (minimum extra "stuff" in the solution).

In overdetermined systems (more equations than unknowns), $\mathbf{x} = \mathbf{A}^+\mathbf{b}$ is the best-fit solution. In underdetermined systems (more unknowns than equations), it finds the solution that doesn't make up unnecessary extra components.

---

## 9. Matrix Rank & Linear Independence

### Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly independent** if:

$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = \cdots = c_k = 0$$

No vector can be written as a linear combination of the others.

**Intuition**: A set of vectors is linearly dependent if one of them is "redundant" — it doesn't add any new direction that you couldn't already reach from the others. For example, $[1, 0]$, $[0, 1]$, and $[2, 3]$ are dependent because $[2, 3] = 2[1,0] + 3[0,1]$. The third vector adds no new information about what directions you can reach.

### Rank

The **rank** of $\mathbf{A}$ = number of linearly independent rows = number of linearly independent columns.

$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$

- **Full rank**: $\text{rank} = \min(m, n)$ — every row and column carries genuinely new information; no redundancy.
- **Rank deficient**: $\text{rank} < \min(m, n)$ — some rows/columns are linear combinations of others; they're telling you things you already knew.

**Why rank matters**: The rank tells you the *true dimensionality* of the transformation. A $1000 \times 1000$ matrix with rank 5 only spans a 5-dimensional subspace despite its size — it lives in 5D, not 1000D. This is the key insight behind dimensionality reduction: real-world data is often low rank (the true degrees of freedom are far fewer than the number of features).

---

## 10. Eigenvalues & Eigenvectors

### Definition

For a square matrix $\mathbf{A}$, a nonzero vector $\mathbf{v}$ is an **eigenvector** if:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

$\lambda$ is the corresponding **eigenvalue**.

**Intuition**: Most vectors get rotated *and* stretched when multiplied by $\mathbf{A}$. But eigenvectors are **special directions** that the transformation only stretches (by factor $\lambda$) — they don't rotate at all. They stay pointed in the same direction (or opposite if $\lambda < 0$). If you were standing on an eigenvector direction, the transformation would feel like just a zoom-in or zoom-out, not a rotation.

$\lambda > 1$: the transformation *amplifies* in that direction.
$0 < \lambda < 1$: it *shrinks* in that direction.
$\lambda < 0$: it flips and scales.
$\lambda = 0$: that direction collapses to zero (the matrix is singular).

### Computing Eigenvalues

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0 \quad \text{(characteristic equation)}$$

We're asking: for what $\lambda$ does $(\mathbf{A} - \lambda\mathbf{I})$ collapse space? That's when its determinant is zero — i.e., when $\lambda$ is just right so that $\mathbf{A}$'s transformation and "$\lambda \times \text{identity}$" cancel each other out in some direction.

### Eigendecomposition

If $\mathbf{A}$ has $n$ linearly independent eigenvectors:

$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{-1}$$

Where $\mathbf{Q}$ has eigenvectors as columns, $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

**Intuition**: This decomposes $\mathbf{A}$ into three steps: (1) rotate to the eigenvector coordinate system ($\mathbf{Q}^{-1}$), (2) scale each axis independently ($\mathbf{\Lambda}$), (3) rotate back ($\mathbf{Q}$). In its own "natural coordinates", $\mathbf{A}$ is just axis-aligned stretching.

### Symmetric Matrices (Spectral Theorem)

If $\mathbf{A} = \mathbf{A}^T$ (symmetric):
- All eigenvalues are **real**
- Eigenvectors are **orthogonal**: $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$

The covariance matrix $\Sigma$ is always symmetric — so its eigenvectors are guaranteed to be orthogonal and its eigenvalues real. This makes PCA well-defined.

---

## 11. Singular Value Decomposition (SVD)

The **SVD** is the most fundamental matrix factorisation. Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

Where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: orthogonal matrix — **left singular vectors** (output directions)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: diagonal matrix — **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ (scaling factors)
- $\mathbf{V}^T \in \mathbb{R}^{n \times n}$: orthogonal matrix — **right singular vectors** (input directions)

### Intuition

Every matrix is just three operations: a rotation (or reflection) in the input space ($\mathbf{V}^T$), an axis-aligned stretch ($\mathbf{\Sigma}$), and a rotation in the output space ($\mathbf{U}$).

Think of it as: "Here are $r$ special directions in the input space ($\mathbf{V}$). When this matrix acts on those directions, it stretches them by factors $\sigma_1 \geq \sigma_2 \geq \cdots$ and sends them to corresponding output directions ($\mathbf{U}$). Everything else gets squashed to zero."

The singular values $\sigma_i$ measure how much "action" the matrix has in each of its special directions. A large $\sigma_1$ means there's one direction the matrix strongly amplifies. Many similar-sized singular values means the matrix acts fairly uniformly across all directions.

### Best Rank-$k$ Approximation (Eckart-Young Theorem)

The best rank-$k$ approximation of $\mathbf{A}$ in Frobenius norm:

$$\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

**Intuition**: Each term $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ is a rank-1 matrix — an outer product of two vectors. Think of it as one "layer" of the matrix, capturing one direction of variation. The SVD sorts these layers by importance ($\sigma_1$ is largest). Keeping only the top $k$ layers gives you the closest rank-$k$ matrix to $\mathbf{A}$. Like image compression: keep only the most important "modes" and discard the noise.

---

## 12. Principal Component Analysis (PCA)

PCA finds the directions (principal components) in your data that carry the most **variance** — i.e., the directions along which your data is most spread out.

**Intuition first**: Imagine a cloud of data points. Most clouds don't spread equally in all directions — there's usually one direction where the cloud is elongated, and another where it's compressed. PCA finds those directions. The first principal component is the single direction that captures the most spread. The second is the best direction orthogonal to the first. And so on. By projecting onto the top $k$ directions, you keep the $k$ most informative dimensions and throw away the rest.

### Algorithm

**Step 1**: Centre the data: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{x}}$

(Subtract the mean of each feature so the cloud is centred at the origin — PCA finds directions of spread around the mean, so you need to remove the mean first.)

**Step 2**: Compute covariance matrix: $\mathbf{C} = \frac{1}{N-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$

($\mathbf{C}_{ij}$ = how much feature $i$ and feature $j$ vary together. The diagonal is variance of each feature.)

**Step 3**: Eigen-decompose: $\mathbf{C} = \mathbf{Q\Lambda Q}^T$

(The eigenvectors of $\mathbf{C}$ are the principal directions. **Why?** Because the direction that maximizes variance $\mathbf{w}^T\mathbf{C}\mathbf{w}$ subject to $\|\mathbf{w}\|=1$ is the top eigenvector — this is a standard result in constrained optimization via Lagrange multipliers.)

**Step 4**: Sort by eigenvalue (descending): $\lambda_1 \geq \lambda_2 \geq \cdots$

(The eigenvalue $\lambda_i$ tells you *how much* variance is captured by that direction. Larger eigenvalue = more spread = more important.)

**Step 5**: Project onto top-$k$ eigenvectors: $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{Q}_k$

(Each data point gets new coordinates in the principal component space — you've compressed from $n$ features to $k$.)

### Connection to SVD

$$\tilde{\mathbf{X}} = \mathbf{U\Sigma V}^T \implies \mathbf{C} = \frac{1}{N-1}\mathbf{V\Sigma}^2\mathbf{V}^T$$

The principal components are the right singular vectors $\mathbf{V}$, and the eigenvalues of $\mathbf{C}$ are $\sigma_i^2 / (N-1)$. In practice, it's numerically more stable to run PCA via SVD than to explicitly form and eigendecompose $\mathbf{C}$.

### Explained Variance

The fraction of variance explained by the first $k$ components:

$$\text{Explained Variance Ratio}_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^n \sigma_i^2}$$

This is your score for how much information you kept. If $k=2$ gives you 95% explained variance, you've compressed the data down to 2D while preserving 95% of its structure.

---

## 📓 Notebook

Open [`linear_algebra.ipynb`](./linear_algebra.ipynb) for hands-on code covering all of the above.
