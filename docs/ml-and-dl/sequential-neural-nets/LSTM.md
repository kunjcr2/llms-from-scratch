# Long Short-Term Memory (LSTM)

## The Problem LSTM Solves

Standard RNNs suffer from the **vanishing gradient problem**: when training on long sequences, gradients become extremely small as they propagate backward through time. This makes it nearly impossible for the network to learn dependencies between events that are far apart in a sequence.

LSTM, introduced by Hochreiter and Schmidhuber in 1997, solves this by introducing a **cell state** that can carry information unchanged across many time steps, along with **gates** that control what information to add or remove.

---

## Symbol Legend

| Symbol | Meaning |
|--------|---------|
| `x_t` | Input vector at time step t |
| `h_t` | Hidden state (output) at time step t |
| `h_{t-1}` | Hidden state from previous time step |
| `c_t` | Cell state at time step t |
| `c_{t-1}` | Cell state from previous time step |
| `W` | Weight matrix |
| `b` | Bias vector |
| `*` | Element-wise multiplication |
| `+` | Element-wise addition |
| `[a, b]` | Concatenation of vectors a and b |
| `sigma` | Sigmoid activation (outputs 0 to 1) |
| `tanh` | Hyperbolic tangent (outputs -1 to 1) |

---

## LSTM Architecture

An LSTM cell has two states that flow through time:
1. **Cell State (c_t)**: The long-term memory. It flows through the network with only minor linear interactions, allowing gradients to flow unchanged.
2. **Hidden State (h_t)**: The short-term memory and output. This is what gets passed to the next layer or used for predictions.

The cell has **three gates** that control information flow:
1. **Forget Gate**: What to remove from cell state
2. **Input Gate**: What new information to add
3. **Output Gate**: What to output from cell state

---

## Step-by-Step Computation

### Step 1: Forget Gate

**Purpose**: Decide what information to throw away from the cell state.

```
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
```

**What this means**:
- Take the previous hidden state `h_{t-1}` and current input `x_t`
- Concatenate them together
- Multiply by weight matrix `W_f` and add bias `b_f`
- Apply sigmoid to squash output between 0 and 1

**Interpretation**:
- Output of 0 means "completely forget this"
- Output of 1 means "completely keep this"
- Values in between mean "partially keep"

---

### Step 2: Input Gate

**Purpose**: Decide what new information to store in the cell state.

This happens in two parts:

**Part A - What to update**:
```
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
```
This gate decides which values we will update (0 to 1 scale).

**Part B - Candidate values**:
```
c_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)
```
This creates a vector of new candidate values that could be added to the cell state. The tanh squashes values between -1 and 1.

**Interpretation**:
- `i_t` acts as a filter: which components should we update?
- `c_tilde` contains the actual new information to potentially add

---

### Step 3: Update Cell State

**Purpose**: Update the old cell state to the new cell state.

```
c_t = f_t * c_{t-1} + i_t * c_tilde
```

**What this means**:
1. `f_t * c_{t-1}`: Multiply old cell state by forget gate. Values we decided to forget get multiplied by values close to 0.
2. `i_t * c_tilde`: Multiply candidate values by input gate. Only add information we decided is worth adding.
3. Add these together to get the new cell state.

**This is the key innovation**: The cell state update is additive, not multiplicative. Gradients can flow backward through addition without vanishing.

---

### Step 4: Output Gate

**Purpose**: Decide what to output based on the cell state.

```
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
```

```
h_t = o_t * tanh(c_t)
```

**What this means**:
1. Compute output gate `o_t` (0 to 1 scale)
2. Apply tanh to cell state to get values between -1 and 1
3. Multiply by output gate to filter what we actually output

**Interpretation**:
- The cell state contains all our memory
- The output gate filters this to produce the relevant output for this time step
- `h_t` is both the output and the hidden state passed to the next time step

---

## Complete LSTM Equations Summary

```
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)       # Forget gate
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)       # Input gate  
c_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)    # Candidate values
c_t = f_t * c_{t-1} + i_t * c_tilde           # Cell state update
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)       # Output gate
h_t = o_t * tanh(c_t)                          # Hidden state (output)
```

---

## Why LSTM Works

1. **Additive Cell State Update**: The cell state is updated via addition (`c_t = ... + ...`), not multiplication. This allows gradients to flow backward without vanishing.

2. **Gate Control**: Gates learn when to let information through and when to block it. The network can learn to:
   - Remember information for many time steps (forget gate stays near 1)
   - Forget irrelevant information (forget gate goes to 0)
   - Selectively add new information (input gate filters)

3. **Separate Memory Streams**: The cell state (long-term) and hidden state (short-term) serve different purposes, giving the network more flexibility.

---

## Applications

- Machine translation
- Text generation
- Speech recognition
- Sentiment analysis
- Time series forecasting
- Video analysis
