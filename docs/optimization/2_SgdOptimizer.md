# Stochastic Gradient Descent (SGD)

> Part 1 of 4 in the optimization series: SGD → Momentum → RMSProp → Adam

---

## Recap: Vanilla Gradient Descent

**Gradient Descent** is an algorithm for minimizing the loss function (cost function), which measures how far predictions are from actual data.

### Loss Function Example (Mean Squared Error)
For linear regression with a line `y = mx + c`:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $\hat{y}_i$ = predicted value
- $y_i$ = actual value
- $n$ = number of data points

### Gradient Descent Algorithm

1. **Initialize** parameters θ (slope `m` and intercept `c`) with random values
2. **Calculate gradient** of loss with respect to θ
3. **Update parameters**: 
   $$\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta J(\theta)$$
   - $\eta$ = learning rate
   - We subtract (descend) because we want to move opposite to the gradient direction
4. **Repeat** steps 2-3 until convergence or predefined iterations

---

## Problems with Vanilla Gradient Descent

| Issue | Description |
|-------|-------------|
| **Computationally Expensive** | Must calculate predictions for ALL data points, compute MSE, calculate gradients, and repeat for many iterations |
| **Difficult Convergence** | Noisy loss surfaces make finding proper minima challenging |
| **Local Minima Trap** | Can get stuck in local minima instead of finding global minimum |
| **Saddle Points** | Gets stuck where gradient = 0 but it's neither maximum nor minimum |

> **Note**: Vanilla GD = Batch GD (uses entire batch of data for each update)

---

## Four Methods to Improve Gradient Descent

1. **Stochastic Gradient Descent** (this lecture)
2. **Momentum** 
3. **RMSProp** (Root Mean Square Propagation)
4. **Adam Optimizer** (most famous)

---

## Stochastic Gradient Descent (SGD)

### Key Difference from Vanilla GD

| Vanilla GD | Stochastic GD |
|------------|---------------|
| Uses ALL data points to compute gradient | Uses ONE random data point per iteration |
| Full batch calculation | Single point calculation |

### Mathematical Formulation

**Vanilla GD**: Gradient computed over all points  
**SGD**: Gradient computed for single random index `i`

For random data point $i$:
- $x_i$ = input feature
- $y_i$ = actual output  
- $\hat{y}_i = mx_i + c$ = prediction

Gradient for parameter update:
$$\frac{\partial}{\partial m}(y_i - \hat{y}_i)^2 = 2 \cdot (y_i - mx_i - c) \cdot (-x_i)$$

---

## Why SGD Works

### Handling Clustered/Redundant Data

Consider data with clusters:
```
    •••           •••           •••
  (cluster 1)   (cluster 2)   (cluster 3)
```

- **Vanilla GD**: Uses every single point from all clusters (redundant computation)
- **SGD**: Randomly selects representative points from clusters (avoids redundancy)

### Scaling Example

| Scenario | Vanilla GD | SGD |
|----------|-----------|-----|
| 1000 parameters, 1M data points | Calculates gradient using ALL 1M points per update | Calculates gradient using 1 random point per update |
| Computation reduction | - | ~1/1,000,000th of calculations |

---

## Advantages of SGD

### 1. Faster Updates
- Significantly reduced computation per iteration
- Most beneficial for large datasets

### 2. Escape Local Minima
- Random point selection creates "noise" in update direction
- Can break out of local minima and explore other regions

### 3. Escape Saddle Points
- At saddle points, overall gradient ≈ 0
- But gradient for a random point ≠ 0
- Allows continued movement even at saddle points

---

## Trade-offs

| Aspect | Behavior |
|--------|----------|
| **Convergence** | Noisy/oscillating (not smooth descent) |
| **Path** | May go uphill temporarily before descending |
| **Best for** | Large datasets with redundant information |
| **Not ideal for** | Small datasets (need all points to capture patterns) |

---

## Variants of Gradient Descent

| Type | Points Used | Use Case |
|------|-------------|----------|
| **Batch GD** (Vanilla) | All points | Small datasets, smooth convergence needed |
| **Stochastic GD** | 1 random point | Very large datasets |
| **Mini-Batch GD** | Subset of points | Balance between speed and stability |

---

## Implementation Notes

### Matrix Formulation for Linear Regression

To compute predictions as matrix multiplication:

```
X_augmented = [X | 1s]    # Append column of 1s
θ = [m, c]ᵀ               # Parameters as column vector
ŷ = X_augmented · θ       # All predictions in one operation
```

### Gradient Computation

```python
# Vanilla GD gradient
gradient = (2/m) * X.T @ (predictions - y)

# SGD gradient (single random point)
random_idx = np.random.randint(0, m)
x_i, y_i = X[random_idx], y[random_idx]
gradient = 2 * x_i.T * (prediction_i - y_i)
```

### Parameter Update
```python
theta -= learning_rate * gradient
```

---

## Experimental Results

### Performance Comparison (100M data points)

| Method | Time | Final Loss |
|--------|------|------------|
| Batch GD | 78 seconds | ~12 |
| SGD | 0.87 seconds | Similar range |

**Key Finding**: SGD was ~100x faster for large datasets!

---

## Visualization: SGD vs Batch GD

**Batch GD**: Smooth, consistent descent toward minimum  
**SGD**: Oscillating path with occasional uphill movement, but reaches similar final destination much faster

```
Loss
 ↑
 │  ╭──╮        BGD: smooth curve down
 │ ╭╯  ╰──╮     SGD: zigzag pattern
 │╭╯      ╰─────────→ both approach minimum
 └─────────────────→ Iterations
```

---

## Key Takeaways

1. **SGD trades accuracy per step for speed** - each update is noisier but much faster
2. **Best for large datasets** - computational savings are proportional to dataset size
3. **Enables escape from local minima** - randomness helps explore loss landscape
4. **Not a replacement** - for small datasets, vanilla GD may still be preferred
5. **Foundation for advanced optimizers** - Momentum, RMSProp, and Adam build on SGD concepts

---

## Next: Momentum

The next lecture covers **Momentum**, which adds physics-inspired concepts to help smooth out SGD's noisy updates while maintaining its speed advantages.
