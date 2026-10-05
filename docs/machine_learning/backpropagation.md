# Backpropagation Explained

Backpropagation is the core algorithm that allows neural networks to learn by adjusting weights based on the error between predicted and actual values.

## What is Backpropagation?

Backpropagation is a method used to compute the gradient of the loss function with respect to the weights of the network using the chain rule of calculus.

Two main phases:

1. Forward Pass – Compute predictions and loss.
2. Backward Pass – Compute gradients and update weights.

## Flow of Backpropagation

1. Input data is passed through the network.
2. Output is calculated (forward pass).
3. Loss is calculated using the loss function.
4. Gradients of the loss with respect to weights are computed (backward pass).
5. Optimizer updates weights using gradients.

## Chain Rule Overview

If you have:

```
y = f(g(h(x)))
```

Then:

```
dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

In neural networks:

```
L → z → a → W
```

```
∂L/∂W = ∂L/∂z * ∂z/∂a * ∂a/∂W
```

## Simple Example (1-Layer NN)

Setup:

- Input: x = 1.0
- Weight: W = 0.5
- Bias: b = 0
- Target: y = 0
- Activation: identity (a = z)

### Forward Pass:

```
z = W * x + b = 0.5
Loss = (z - y)^2 = (0.5 - 0)^2 = 0.25
```

### Backward Pass:

```
dL/dz = 2 * (z - y) = 1.0
∂z/∂W = x = 1.0
∂L/∂W = dL/dz * ∂z/∂W = 1.0 * 1.0 = 1.0
```

### Weight Update (Gradient Descent):

```
W = W - lr * ∂L/∂W = 0.5 - 0.1 * 1.0 = 0.4
```

## Summary Diagram

```
[Input x] --> [Layer: W, b] --> [Activation] --> [Prediction]
                                      ↓
                                [Loss Function]
                                      ↓
                                [Backpropagate: Compute Gradients]
                                      ↓
                                [Update Weights]
```

## Key Takeaways

- Backpropagation flows error backwards using gradients.
- The chain rule is used to calculate how each weight contributed to the final error.
- Optimizers use these gradients to update model parameters.
- Deep learning frameworks automate this with `.backward()`.
