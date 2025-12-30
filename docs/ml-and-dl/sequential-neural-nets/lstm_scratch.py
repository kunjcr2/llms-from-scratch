"""
Long Short-Term Memory (LSTM) Implementation

A minimal implementation of LSTM using PyTorch.
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    Single LSTM cell that processes one time step.
    
    Gates:
        f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  # Forget gate
        i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  # Input gate
        o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)  # Output gate
    
    Cell state update:
        c_tilde = tanh(W_c * [h_{t-1}, x_t] + b_c)
        c_t = f_t * c_{t-1} + i_t * c_tilde
        h_t = o_t * tanh(c_t)
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combined weights for all gates (more efficient)
        # Order: input, forget, cell, output
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input at current time step (batch_size, input_size)
            state: Tuple of (h, c) where:
                h: Hidden state (batch_size, hidden_size) - short term memory
                c: Cell state (batch_size, hidden_size) - long term memory
        
        Returns:
            h_new: New hidden state (batch_size, hidden_size)
            c_new: New cell state (batch_size, hidden_size)
        """
        h, c = state
        
        # Compute all gates at once
        gates = self.W_ih(x) + self.W_hh(h)
        
        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_new = f * c + i * g
        
        # Compute new hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class LSTM(nn.Module):
    """
    Full LSTM module that processes sequences.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Stack of LSTM cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(cell_input_size, hidden_size))
        
        # Dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None
    
    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input sequence (batch, seq_len, input_size) if batch_first
               else (seq_len, batch, input_size)
            state: Tuple of (h_0, c_0) where each has shape
                   (num_layers, batch, hidden_size)
        
        Returns:
            output: All hidden states (batch, seq_len, hidden_size)
            (h_n, c_n): Final states (num_layers, batch, hidden_size) each
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, input_size)
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize states
        if state is None:
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )
            c_0 = torch.zeros_like(h_0)
        else:
            h_0, c_0 = state
        
        # Convert to lists for easier manipulation
        h = list(h_0)
        c = list(c_0)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(x_t, (h[layer_idx], c[layer_idx]))
                x_t = h[layer_idx]
                
                # Apply dropout between layers (not after last layer)
                if self.dropout and layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)
            
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=0)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, (h_n, c_n)


class LSTMLanguageModel(nn.Module):
    """
    LSTM for language modeling (next token prediction).
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Optionally tie input and output embeddings
        if tie_weights and embed_dim == hidden_size:
            self.fc.weight = self.embedding.weight
    
    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input token indices (batch, seq_len)
            state: Optional previous state for generation
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            state: Updated (h_n, c_n)
        """
        embedded = self.embedding(x)
        output, state = self.lstm(embedded, state)
        logits = self.fc(output)
        return logits, state
    
    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            start_tokens: Initial tokens (batch, start_len)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated tokens (batch, max_length)
        """
        self.eval()
        tokens = start_tokens
        state = None
        
        for _ in range(max_length - start_tokens.size(1)):
            logits, state = self(tokens[:, -1:], state)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens


if __name__ == "__main__":
    # Demo
    batch_size = 4
    seq_len = 10
    input_size = 8
    hidden_size = 16
    num_layers = 2
    
    # Create LSTM
    lstm = LSTM(input_size, hidden_size, num_layers)
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output, (h_n, c_n) = lstm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Final hidden state shape: {h_n.shape}")
    print(f"Final cell state shape: {c_n.shape}")
    
    # Language model demo
    vocab_size = 1000
    
    lm = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_size=64,
        num_layers=2
    )
    
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, _ = lm(tokens)
    
    print(f"\nLanguage model input shape: {tokens.shape}")
    print(f"Language model output shape: {logits.shape}")
    
    # Generation demo
    start = torch.randint(0, vocab_size, (1, 3))
    generated = lm.generate(start, max_length=20)
    print(f"\nGenerated sequence shape: {generated.shape}")
