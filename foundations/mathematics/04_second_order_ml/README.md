# Second-Order Ideas for ML

> *This section explains the chain of ideas that leads from ordinary derivatives to Elastic Weight Consolidation (EWC). The core theme is simple: once you understand how a function changes locally, you can approximate it, reason about curvature, and build useful machine learning algorithms from that approximation.*

---

## Why This Section Exists

In deep learning, students often learn gradient descent first and stop there. That gets you far, but many important ideas in modern ML need one extra layer of mathematical maturity:

- How does a loss change when there are many parameters?
- How do we approximate a complicated loss near a good solution?
- How do we measure which parameters are "important" for an old task?
- Why do papers avoid explicitly storing full Jacobians or Hessians?

This note answers those questions in one ladder of concepts.

---

## The Ladder

Each step depends on the one before it:

1. **Partial derivative**: derivative with respect to one variable while holding the others fixed
2. **Gradient**: all partial derivatives collected into one vector
3. **Jacobian**: the derivative of a vector-valued function, written as a matrix
4. **Hessian**: second derivatives of a scalar-valued function, written as a matrix
5. **1D Taylor series**: approximate a function using value, slope, curvature, and higher derivatives
6. **Multivariable Taylor series**: the same idea in many dimensions, using gradient and Hessian
7. **EWC**: a continual-learning method that uses the multivariable Taylor idea on the old task loss
8. **JVP trick**: a way to get Jacobian-related information without storing the full Jacobian

If one of these is blurry, the next one becomes harder. So we will build carefully.

---

## A Running Example

To keep things concrete, we will often use the function

$$
f(x, y) = x^2 + 3xy + 2y^2
$$

This is a simple scalar-valued function of two variables. It is not a neural network, but mathematically it behaves like a tiny loss function with two parameters.

Later, when we talk about neural networks, you can mentally replace $(x, y)$ by a parameter vector

$$
\theta = (\theta_1, \theta_2, \dots, \theta_n).
$$

---

## 1. Partial Derivative

### The Main Idea

If a function depends on many variables, we can ask:

- "What happens if only $x$ changes?"
- "What happens if only $y$ changes?"

That is exactly what a **partial derivative** measures.

For a function $f(x_1, x_2, \dots, x_n)$, the partial derivative with respect to $x_i$ is

$$
\frac{\partial f}{\partial x_i}.
$$

It means: vary $x_i$, freeze all the other inputs, and measure the instantaneous rate of change.

### Why "partial"?

Because you are not looking at the total change under all variables moving together. You are looking at one coordinate direction at a time.

### Example

For

$$
f(x, y) = x^2 + 3xy + 2y^2,
$$

the partial derivative with respect to $x$ is

$$
\frac{\partial f}{\partial x} = 2x + 3y.
$$

Reason:

- derivative of $x^2$ with respect to $x$ is $2x$
- derivative of $3xy$ with respect to $x$ is $3y$ because $y$ is treated as a constant
- derivative of $2y^2$ with respect to $x$ is $0$ because it does not involve $x$

Similarly,

$$
\frac{\partial f}{\partial y} = 3x + 4y.
$$

### Intuition in ML

Suppose $f$ is the loss and $x,y$ are parameters.

- $\frac{\partial f}{\partial x}$ tells you how sensitive the loss is to parameter $x$
- $\frac{\partial f}{\partial y}$ tells you how sensitive the loss is to parameter $y$

This is the first local sensitivity measure in optimization.

---

## 2. Gradient

### The Main Idea

If partial derivatives tell you the effect of moving along each coordinate separately, the **gradient** collects all of them into one object.

For a scalar-valued function

$$
f: \mathbb{R}^n \to \mathbb{R},
$$

the gradient is

$$
\nabla f(\mathbf{x}) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}.
$$

### Example

For our running function,

$$
\nabla f(x,y) =
\begin{bmatrix}
2x + 3y \\
3x + 4y
\end{bmatrix}.
$$

### What the Gradient Means Geometrically

The gradient points in the direction of **steepest increase**.

That means:

- if you move a tiny step in the gradient direction, the function increases as fast as possible
- if you move a tiny step in the negative gradient direction, the function decreases as fast as possible

This is why gradient descent uses

$$
\theta \leftarrow \theta - \eta \nabla_\theta L(\theta).
$$

### What the Gradient Means in ML

If $L(\theta)$ is the loss of a neural network, then

$$
\nabla_\theta L(\theta)
$$

is the vector telling you how each parameter should move to reduce loss most quickly to first order.

This is the basic engine of training.

### Important Shape Fact

If $\theta \in \mathbb{R}^n$, then the gradient is also an object with $n$ entries.

So:

- scalar loss
- vector of derivatives

That shape pattern matters because the Jacobian generalizes it.

---

## 3. Jacobian

### Why Gradient Is Not Enough

The gradient works for functions that output a single number:

$$
f: \mathbb{R}^n \to \mathbb{R}.
$$

But many functions in ML output vectors:

$$
\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m.
$$

Examples:

- a layer maps an input vector to an output vector
- a model maps features to logits
- a softmax function maps logits to class probabilities

Now one derivative number is not enough, and one gradient vector is not enough either. Each output component depends on each input component.

### Definition

The **Jacobian** of

$$
\mathbf{g}(\mathbf{x}) =
\begin{bmatrix}
g_1(\mathbf{x}) \\
g_2(\mathbf{x}) \\
\vdots \\
g_m(\mathbf{x})
\end{bmatrix}
$$

is the matrix

$$
J(\mathbf{x}) =
\begin{bmatrix}
\frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} & \cdots & \frac{\partial g_1}{\partial x_n} \\
\frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2} & \cdots & \frac{\partial g_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial g_m}{\partial x_1} & \frac{\partial g_m}{\partial x_2} & \cdots & \frac{\partial g_m}{\partial x_n}
\end{bmatrix}.
$$

### What It Means

Each row is the gradient of one output component.

So the Jacobian tells you:

- how output 1 changes with all inputs
- how output 2 changes with all inputs
- ...
- how output $m$ changes with all inputs

### Small Example

Let

$$
\mathbf{g}(x,y) =
\begin{bmatrix}
x^2 + y \\
xy
\end{bmatrix}.
$$

Then

$$
J(x,y) =
\begin{bmatrix}
2x & 1 \\
y & x
\end{bmatrix}.
$$

### Why It Matters in Deep Learning

Neural networks are compositions of vector-valued maps. Backpropagation repeatedly applies Jacobian-like operations through layers.

At a high level:

- forward pass: vectors move forward
- backward pass: derivatives move backward

The Jacobian is the formal object that connects those two.

### Gradient as a Special Case

If the output dimension is 1, the Jacobian becomes a single row, which is essentially the transpose of the gradient.

So:

- gradient = derivative of scalar-valued function
- Jacobian = derivative of vector-valued function

---

## 4. Hessian

### The Main Idea

The gradient tells you the local slope.
The **Hessian** tells you how that slope itself changes.

So the Hessian measures **curvature**.

For a scalar-valued function

$$
f: \mathbb{R}^n \to \mathbb{R},
$$

the Hessian is the matrix of second partial derivatives:

$$
H(\mathbf{x}) =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}.
$$

### Example

From

$$
f(x, y) = x^2 + 3xy + 2y^2,
$$

we already found

$$
\nabla f(x,y) =
\begin{bmatrix}
2x + 3y \\
3x + 4y
\end{bmatrix}.
$$

Differentiate again:

$$
H(x,y) =
\begin{bmatrix}
\frac{\partial}{\partial x}(2x+3y) & \frac{\partial}{\partial y}(2x+3y) \\
\frac{\partial}{\partial x}(3x+4y) & \frac{\partial}{\partial y}(3x+4y)
\end{bmatrix}
=
\begin{bmatrix}
2 & 3 \\
3 & 4
\end{bmatrix}.
$$

### What the Entries Mean

- diagonal entries: curvature with respect to each variable alone
- off-diagonal entries: interaction between variables

The off-diagonal terms are extremely important in ML. They tell you that parameters do not act independently. Changing parameter $x$ can affect how sensitive the loss is to parameter $y$.

### Why Hessians Matter

The Hessian tells you whether a point is locally:

- bowl-shaped: likely a local minimum
- hill-shaped: likely a local maximum
- saddle-shaped: mixed curvature

This is why second-order methods look at Hessians. They do not just ask "which way is downhill?" They ask "what is the local geometry of the surface?"

### Important ML Interpretation

If a parameter direction has large curvature, a small step can change the loss a lot.
If a parameter direction has small curvature, you can move more without changing the loss much.

This is exactly the kind of information EWC tries to use.

---

## 5. 1D Taylor Series

### The Core Idea

Suppose a function is complicated, but you only care about behavior near a point $a$.

Instead of using the full function, you can approximate it using derivatives at $a$.

For a one-variable function, the Taylor expansion around $a$ is

$$
f(x) = f(a)
+ f'(a)(x-a)
+ \frac{1}{2}f''(a)(x-a)^2
+ \frac{1}{3!}f^{(3)}(a)(x-a)^3
+ \cdots
$$

### What Each Term Means

- $f(a)$: the value at the expansion point
- linear term: local slope
- quadratic term: local curvature
- cubic and higher terms: finer local shape

### Why This Is Powerful

A hard function can be replaced locally by an easier polynomial.

Near $a$:

- first-order approximation uses slope
- second-order approximation uses slope plus curvature

### First-Order Approximation

$$
f(x) \approx f(a) + f'(a)(x-a)
$$

This is just the tangent line approximation.

### Second-Order Approximation

$$
f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2
$$

This adds curvature, so it is much better when the function bends.

### Example

Take

$$
f(x) = x^2.
$$

Around $a=1$:

- $f(1)=1$
- $f'(x)=2x$, so $f'(1)=2$
- $f''(x)=2$, so $f''(1)=2$

Then

$$
f(x) \approx 1 + 2(x-1) + \frac{1}{2}\cdot 2(x-1)^2
= 1 + 2(x-1) + (x-1)^2.
$$

For a quadratic function like $x^2$, the second-order Taylor approximation is exact.

### ML Connection

Optimization methods often pretend the loss near the current point looks like:

- a line if using first-order information
- a quadratic bowl if using second-order information

This local approximation viewpoint is one of the most useful mental models in machine learning.

---

## 6. Multivariable Taylor Series

Now we generalize Taylor expansion from one variable to many.

This is the mathematical bridge from derivatives to EWC.

### Second-Order Expansion

For a scalar-valued function $f(\mathbf{x})$, expanded around a point $\mathbf{a}$:

$$
f(\mathbf{x}) \approx
f(\mathbf{a})
+ \nabla f(\mathbf{a})^T(\mathbf{x}-\mathbf{a})
+ \frac{1}{2}(\mathbf{x}-\mathbf{a})^T H(\mathbf{a})(\mathbf{x}-\mathbf{a}).
$$

This is the most important formula in this whole note.

### Read It Slowly

Let $\mathbf{d} = \mathbf{x} - \mathbf{a}$ be a small displacement from the point $\mathbf{a}$.

Then

- $f(\mathbf{a})$ is the base value
- $\nabla f(\mathbf{a})^T \mathbf{d}$ is the linear change from slope
- $\frac{1}{2}\mathbf{d}^T H(\mathbf{a}) \mathbf{d}$ is the quadratic correction from curvature

### Why the Hessian Appears

In one dimension, curvature is measured by $f''(a)$.
In many dimensions, curvature has to describe:

- bending along coordinate directions
- interactions between directions

That is exactly what the Hessian encodes.

### If We Are at a Minimum

Suppose $\mathbf{a}$ is a local minimum of the function. Then usually

$$
\nabla f(\mathbf{a}) = \mathbf{0}.
$$

So the approximation simplifies to

$$
f(\mathbf{x}) \approx
f(\mathbf{a})
+ \frac{1}{2}(\mathbf{x}-\mathbf{a})^T H(\mathbf{a})(\mathbf{x}-\mathbf{a}).
$$

This says:

> Near a minimum, the function behaves approximately like a quadratic form.

That statement is central in optimization and continual learning.

### Why This Matters for Neural Networks

Let $L_A(\theta)$ be the loss for Task A.
After training on Task A, assume we have found parameters $\theta_A^*$ that work well.

Then near $\theta_A^*$,

$$
L_A(\theta) \approx
L_A(\theta_A^*)
+ \frac{1}{2}(\theta-\theta_A^*)^T H_A (\theta-\theta_A^*),
$$

where

$$
H_A = \nabla_\theta^2 L_A(\theta_A^*).
$$

That means the old task loss increases quadratically as we move away from the old optimum, with the Hessian telling us which directions are dangerous.

This is exactly the idea behind EWC.

---

## 7. EWC = Step 6 Applied to Task A's Loss

EWC stands for **Elastic Weight Consolidation**.

It is a continual-learning method designed to reduce **catastrophic forgetting**.

### The Problem

You train on Task A and get good parameters $\theta_A^*$.
Then you train on Task B.

If you optimize only for Task B, gradient descent may move the parameters far away from values that were important for Task A. Performance on Task A collapses. That is catastrophic forgetting.

### The Key Idea

Do not allow the parameters to move freely away from the Task A solution in directions that are important for Task A.

To formalize "important", use a local quadratic approximation of Task A's loss around $\theta_A^*$.

### Start from the Multivariable Taylor Approximation

Around $\theta_A^*$,

$$
L_A(\theta) \approx
L_A(\theta_A^*)
+ \nabla L_A(\theta_A^*)^T(\theta-\theta_A^*)
+ \frac{1}{2}(\theta-\theta_A^*)^T H_A(\theta-\theta_A^*).
$$

If $\theta_A^*$ is a well-trained point, then

$$
\nabla L_A(\theta_A^*) \approx 0.
$$

So

$$
L_A(\theta) \approx
L_A(\theta_A^*)
+ \frac{1}{2}(\theta-\theta_A^*)^T H_A(\theta-\theta_A^*).
$$

This says the increase in Task A loss caused by moving to new parameters $\theta$ is approximately quadratic.

### Interpretation

If a direction has high curvature under Task A, then moving in that direction increases the old loss quickly.

So those parameters or directions should be protected.

If a direction has low curvature, then the model can move there with little harm to Task A.

### Diagonal Approximation

For large neural networks, the full Hessian is enormous and hard to compute or store.

So EWC uses a diagonal approximation:

$$
H_A \approx \operatorname{diag}(F_1, F_2, \dots, F_n),
$$

where the diagonal entries are usually estimated with the **Fisher information**.

Then the quadratic form becomes

$$
\frac{1}{2}\sum_i F_i(\theta_i - \theta_{A,i}^*)^2.
$$

### Final EWC Objective

When learning Task B, instead of minimizing only $L_B(\theta)$, EWC minimizes

$$
L_{\text{EWC}}(\theta)
=
L_B(\theta)
+ \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_{A,i}^*)^2.
$$

### Read This Loss Carefully

- $L_B(\theta)$ says: learn the new task
- $(\theta_i - \theta_{A,i}^*)^2$ says: do not drift too far from the old solution
- $F_i$ says: penalize drift more strongly for parameters that mattered more to Task A
- $\lambda$ sets the overall strength of memory preservation

### Why the Name "Elastic"?

Because each parameter is attached to its old value by something like a spring.

- weak spring if the parameter was not important
- strong spring if the parameter was important

So the model can adapt, but not equally in all directions.

### Conceptual Summary

EWC is not a random regularizer.
It is a mathematically motivated regularizer coming from a **second-order local approximation** of the old task loss.

That is the conceptual punchline:

> EWC is step 6 applied to Task A's loss, with a practical diagonal curvature approximation.

---

## 8. Why Jacobians Blow Up Storage

Before the trick, we need to understand the problem.

### Shape Explosion

Suppose

$$
\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m.
$$

Its Jacobian has shape

$$
m \times n.
$$

If both $m$ and $n$ are large, this matrix becomes huge.

### Example in Deep Learning

Imagine:

- output dimension $m = 10{,}000$
- parameter count $n = 10^7$

Then the Jacobian has

$$
10{,}000 \times 10^7 = 10^{11}
$$

entries.

That is completely impractical to materialize in memory.

Even if each entry used only 4 bytes, this would require hundreds of gigabytes.

### But Do We Really Need the Full Jacobian?

Usually, no.

In practice we often only need:

- $J\mathbf{v}$ for some vector $\mathbf{v}$, called a **Jacobian-vector product (JVP)**
- $J^T\mathbf{v}$ for some vector $\mathbf{v}$, called a **vector-Jacobian product (VJP)**

These are far cheaper than constructing $J$ explicitly.

---

## 9. The JVP Trick

### Definition

Given a function

$$
\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m
$$

and a vector $\mathbf{v} \in \mathbb{R}^n$, the JVP is

$$
J(\mathbf{x})\mathbf{v}.
$$

This gives the directional change of the output when the input moves in direction $\mathbf{v}$.

### Directional Derivative View

A JVP can be understood as

$$
J(\mathbf{x})\mathbf{v}
=
\left.\frac{d}{d\epsilon}\mathbf{g}(\mathbf{x} + \epsilon \mathbf{v})\right|_{\epsilon=0}.
$$

This is a very important identity.

It says:

> Instead of building the whole Jacobian, just ask how the function changes along one chosen direction.

That directional derivative is exactly the Jacobian times a vector.

### Tiny Example

Let

$$
\mathbf{g}(x,y) =
\begin{bmatrix}
x^2 + y \\
xy
\end{bmatrix},
\qquad
J(x,y) =
\begin{bmatrix}
2x & 1 \\
y & x
\end{bmatrix}.
$$

Take direction

$$
\mathbf{v} =
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}.
$$

Then

$$
J(x,y)\mathbf{v}
=
\begin{bmatrix}
2xv_1 + v_2 \\
yv_1 + xv_2
\end{bmatrix}.
$$

We got the effect of the Jacobian on a direction without needing to use it as a giant standalone object in a realistic large-scale setting.

### Why This Helps in ML

Many second-order computations need things like:

- Hessian-vector products
- Fisher-vector products
- sensitivity along a direction
- linearized network behavior

These can often be expressed using JVPs and VJPs, avoiding explicit Jacobian construction.

### Important Practical Distinction

In deep learning libraries:

- **forward-mode autodiff** naturally computes JVPs
- **reverse-mode autodiff** naturally computes VJPs

Backpropagation is fundamentally a reverse-mode idea, which is why $J^T\mathbf{v}$ shows up constantly in training.

### Connection to Hessian-Vector Products

A Hessian is also too large to store for modern models.
But many algorithms only need

$$
H\mathbf{v}.
$$

Using autodiff tricks, this can be computed without forming $H$ explicitly, just as JVPs avoid forming $J$ explicitly.

So the big pattern is:

- full derivative objects are mathematically clean
- derivative-vector products are computationally practical

That is a very common theme in modern ML systems.

---

## Putting the Whole Story Together

Here is the full conceptual chain in one place:

1. A **partial derivative** tells you how a scalar changes when one variable moves.
2. The **gradient** collects all those sensitivities for a scalar-valued function.
3. The **Jacobian** extends this to vector-valued outputs.
4. The **Hessian** measures curvature of a scalar-valued function.
5. A **1D Taylor series** approximates a function locally using derivatives.
6. A **multivariable Taylor series** uses gradient and Hessian to approximate a loss near a point.
7. **EWC** uses that local quadratic approximation around the Task A optimum to prevent forgetting.
8. Because full Jacobians and Hessians are huge, practical ML uses **JVPs/VJPs and vector-product tricks** instead of materializing the full matrices.

That is the ladder.

---

## What a Third-Year ML Student Should Remember

If you remember only five things, remember these:

1. The gradient is the multivariable version of slope.
2. The Hessian tells you which parameter directions are stiff and which are flexible.
3. Taylor expansion is the tool that converts local derivative information into a usable approximation of a function.
4. EWC is a second-order idea: protect old-task-important directions using a quadratic penalty.
5. In real deep learning, we almost never store full Jacobians or Hessians because derivative-vector products give the needed information much more cheaply.

---

## Suggested Next Questions

Once this note feels clear, the natural next questions are:

- Why is the Fisher information a reasonable substitute for the Hessian in EWC?
- What is the difference between full EWC, online EWC, and diagonal EWC?
- How do Hessian-vector products work in autodiff?
- Why is reverse-mode autodiff ideal for neural network training?
- How does the Gauss-Newton matrix relate to the Hessian?

Those are the right follow-up questions once the ladder here has clicked.
