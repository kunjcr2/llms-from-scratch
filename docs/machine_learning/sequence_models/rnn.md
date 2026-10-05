# Recurrent Neural Networks (RNN)

## Overview

Recurrent Neural Networks are a class of neural networks designed to handle sequential data. Unlike feedforward networks, RNNs have connections that loop back on themselves, allowing them to maintain a form of memory across time steps.

## Architecture

An RNN processes sequences one element at a time, maintaining a hidden state that gets updated at each step:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

Where:
- `h_t` is the hidden state at time t
- `x_t` is the input at time t
- `W_hh`, `W_xh`, `W_hy` are weight matrices
- `b_h`, `b_y` are bias vectors

## Key Characteristics

1. **Parameter Sharing**: The same weights are used across all time steps, reducing the number of parameters.

2. **Sequential Processing**: Inputs are processed in order, making RNNs suitable for time-series data, text, and speech.

3. **Variable Length Input**: RNNs can handle sequences of different lengths.

## Limitations

### Vanishing Gradient Problem
During backpropagation through time (BPTT), gradients can become extremely small as they propagate through many time steps. This makes it difficult for the network to learn long-range dependencies.

### Exploding Gradient Problem
Conversely, gradients can also grow exponentially, causing unstable training. Gradient clipping is commonly used to mitigate this.

### Limited Memory
Standard RNNs struggle to retain information over long sequences due to the vanishing gradient problem.

## Applications

- Language modeling
- Speech recognition
- Time series prediction
- Sequence-to-sequence tasks

## Training

RNNs are trained using **Backpropagation Through Time (BPTT)**, which unrolls the network across time steps and applies standard backpropagation to compute gradients.

## Variants

- **Bidirectional RNN**: Processes sequences in both forward and backward directions
- **Deep RNN**: Stacks multiple RNN layers
- **LSTM**: Addresses vanishing gradient with gating mechanisms
- **GRU**: Simplified version of LSTM with fewer parameters
