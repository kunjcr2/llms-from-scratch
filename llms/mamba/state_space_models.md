# State Space Models (SSMs)

State Space Models are a class of architectures designed to combine the strengths of Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs). First introduced in the context of deep learning by Albert Gu and Tri Dao, SSMs offer the linear scaling of RNNs during inference and the parallelizable training of CNNs.

## The State Space Analogy: A Race Car

To understand the core components of an SSM, consider the analogy of maintaining a race car over time.

- **Input ($x_t$)**: The maintenance actions performed on a given day (e.g., topping up fluids, replacing tires).
- **Hidden State ($h_t$)**: The overall health of the vehicle. This includes the gas and oil levels, tire condition, and motor wear. The state is updated daily based on both the previous state and the maintenance performed.
- **Output ($y_t$)**: The performance or speed of the car, which is measured as a direct result of its current health.

### The System Matrices

The dynamics of this system are governed by three primary matrices:

1.  **Matrix $A$ (State Transition)**: Represents the internal dynamics of the system, such as "wear and tear." It defines how the hidden state evolves from one day to the next (e.g., gas levels dropping, parts aging).
2.  **Matrix $B$ (Control/Input)**: Defines how the input (maintenance) influences the hidden state (e.g., how adding oil improves vehicle health).
3.  **Matrix $C$ (Observation)**: Maps the internal hidden state to the observable output (e.g., how the car's current health translates into its top speed).

In most language models, a fourth matrix, **$D$ (Direct Action)**, is omitted as we assume the input affects the output only indirectly through the hidden state.

---

## Mathematical Formulation

The behavior of a discrete-time State Space Model is described by two fundamental equations:

1.  **State Equation**:
    $$h_t = A h_{t-1} + B x_t$$
    The current state $h_t$ is a combination of the previous state $h_{t-1}$ (modified by $A$) and the current input $x_t$ (modified by $B$).

2.  **Output Equation**:
    $$y_t = C h_t$$
    The output $y_t$ is derived directly from the current state $h_t$ through matrix $C$.

---

## Application to Language Generation

In the context of Large Language Models, the variables map as follows:

- **State ($h_t$)**: The **Context**. This is a high-dimensional vector that stores the abstract history of the conversation, including the topic, tense, and tone.
- **Input ($x_t$)**: The **Last Token**. The most recent word and its representation are fed into the system to update the context.
- **Output ($y_t$)**: The **Next Token Probabilities**. The model outputs a distribution over the vocabulary to sample the next word.

As words are processed, the matrix $A$ helps the model decide which parts of the context to remember and which to forget, while $B$ integrates the newest information into the current state.

---

## Computational Efficiency: The Convolutional Trick

While the recursive form ($h_t = A h_{t-1} + B x_t$) is efficient for step-by-step inference, it is difficult to parallelize on GPUs during training. However, for a Linear Time-Invariant (LTI) system where $A, B, C$ are fixed, we can expand the recurrence:

- $h_0 = B x_0$
- $y_0 = C B x_0$
- $h_1 = A h_0 + B x_1 = A B x_0 + B x_1$
- $y_1 = C A B x_0 + C B x_1$
- $y_k = C A^k B x_0 + C A^{k-1} B x_1 + \dots + C B x_k$

This expansion shows that the entire sequence of outputs $y$ can be computed as a **1D Convolution** between the input sequence $x$ and a precomputed kernel $K = (C A^k B, C A^{k-1} B, \dots, C B)$.

### Why This Matters

1.  **Parallelization**: Convolutions can be computed extremely fast on GPUs, allowing the model to be trained on entire sequences at once.
2.  **Inference Speed**: Once trained, the model can revert to the recursive RNN-like form, providing $O(1)$ time complexity per step and $O(1)$ memory usage relative to sequence length.