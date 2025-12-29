# Introduction to Optimization for Machine Learning

> This lecture covers the theory of optimization and implements gradient descent from scratch.

---

## What is Optimization?

In ML, **optimization** means finding the best values for model parameters that minimize a cost/loss function.

### Linear Regression Example

- Data points are scattered, and we fit a line: $\hat{y} = mx + c$
- **m** = slope, **c** = intercept (the parameters)
- Starting with random values of m and c, optimization finds the optimal values
- A good model passes close to the data points; a bad model is far away

> The goal: Minimize the cost function to get a model that approximates the data well.

---

## Cost Functions

A **cost function** (or loss function) measures how good your predictions are compared to actual data.

- **High cost** → bad model (far from data points)
- **Low cost** → good model (close to data points)

### Mean Squared Error (MSE) — Used in Regression

Three terms: **Mean**, **Squared**, **Error**

| Term | Meaning |
|------|---------|
| Error | Difference between prediction and actual: $\hat{y}_i - y_i$ |
| Squared | Square the error (handles +/- differences) |
| Mean | Average across all data points |

**Formula:**

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

Where:
- $\hat{y}_i = mx_i + c$ (predicted value)
- $y_i$ = actual value
- $n$ = number of data points

> Minimizing MSE gives us the best-fit regression line.

### Other Cost Functions (briefly)

| Cost Function | Use Case |
|---------------|----------|
| Cross-Entropy Loss | Classification problems |
| Log Loss | Logistic Regression |

---

## Gradient Descent

The backbone algorithm for optimizing ML models. Two key terms:
- **Gradient**: The slope of the loss function at a given point
- **Descent**: Moving downhill to find the minimum

### The Landscape Analogy

- Loss function creates a "landscape" where height = loss value
- Parameters (m, c) define position on this landscape
- Goal: Find the lowest point (global minimum)
- Like a ball rolling downhill, gradient descent follows the steepest path down

### The Algorithm

**Update Rule:**

$$\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta J(\theta)$$

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Model parameters (e.g., m and c) |
| $J(\theta)$ | Loss function (e.g., MSE) |
| $\eta$ | Learning rate |
| $\nabla_\theta J(\theta)$ | Gradient of loss w.r.t. parameters |
| $-$ | Minus sign because we're descending |

### Steps:
1. Start with random parameter values
2. Calculate the gradient: $\nabla_\theta J(\theta)$
3. Update parameters: $\theta = \theta - \eta \cdot \nabla$
4. Repeat steps 2-3 until convergence

---

## Hands-On Example: Linear Regression

### Dataset
| X | Y |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |

This follows $y = x + 1$, so ideal parameters: **m = 1, c = 1**

### Expanding MSE

$$MSE = \frac{1}{4}\left[m^2\sum x_i^2 + 4c^2 + 2mc\sum x_i + \sum y_i^2 - 2m\sum x_i y_i - 2c\sum y_i\right]$$

### Precomputed Values from Data
- $\sum x_i = 10$
- $\sum x_i^2 = 30$
- $\sum y_i = 14$
- $\sum y_i^2 = 54$
- $\sum x_i y_i = 40$

### Gradients

$$\frac{\partial MSE}{\partial m} = \frac{1}{4}(-80 + 60m + 20c)$$

$$\frac{\partial MSE}{\partial c} = \frac{1}{4}(-28 + 8c + 20m)$$

### Update Equations

$$m_{new} = m_{old} - \eta \cdot \frac{\partial MSE}{\partial m}$$

$$c_{new} = c_{old} - \eta \cdot \frac{\partial MSE}{\partial c}$$

### Excel/Sheets Demonstration Results

| Iterations | Learning Rate | Final m | Final c | Final MSE |
|------------|---------------|---------|---------|-----------|
| 150 | 0.1 | ≈1.02 | ≈0.99 | ≈0.00006 |
| 1000 | 0.01 | ≈0.97 | ≈0.98 | ≈0.0014 |

---

## Learning Rate Issues

### Too High (e.g., 0.2 or 0.3)
- Loss **diverges** (increases to huge values like $10^{115}$)
- Overshoots the minimum, bouncing back and forth
- Gradient sign alternates (+/−/+/−) 

### Too Low (e.g., 0.001)
- Convergence is extremely slow
- Needs ~10x more iterations
- May not reach optimal values in reasonable time

### Just Right (e.g., 0.01 or 0.1)
- Smooth convergence to minimum
- Loss decreases steadily

> Finding the right learning rate often requires experimentation.

---

## Python Implementation from Scratch

### Key Code Components

```python
# Gradient calculations
gradient_m = -2/n * np.sum(x * (y - y_predicted))
gradient_c = -2/n * np.sum(y - y_predicted)

# Parameter updates
m = m - learning_rate * gradient_m
c = c - learning_rate * gradient_c

# Loss calculation
loss = np.mean((y - y_predicted) ** 2)
```

### Derivation of Gradients

From $MSE = \frac{1}{n}\sum(mx + c - y)^2$:

$$\frac{\partial MSE}{\partial m} = \frac{2}{n}\sum x(mx + c - y) = \frac{2}{n}\sum x(\hat{y} - y)$$

$$\frac{\partial MSE}{\partial c} = \frac{2}{n}\sum(mx + c - y) = \frac{2}{n}\sum(\hat{y} - y)$$

> Note: The minus sign in code comes from reordering $(y - \hat{y})$ vs $(\hat{y} - y)$

### Results
- With noise: Cannot reach MSE = 0 (line can't pass through all random points)
- Without noise: MSE → 0, m → 2, c → 1 (perfect fit)

---

## Key Takeaways

1. **Optimization** = finding parameter values that minimize the loss function
2. **MSE** measures average squared difference between predictions and actual values
3. **Gradient Descent** iteratively updates parameters by moving opposite to the gradient
4. **Learning rate** controls step size — too high diverges, too low is slow
5. This vanilla gradient descent is also called **batch gradient descent** (uses all data points)

> Next lecture: Stochastic Gradient Descent