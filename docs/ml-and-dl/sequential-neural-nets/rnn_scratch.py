"""
Recurrent Neural Network (RNN) Implementation

A minimal implementation of a vanilla RNN using PyTorch.
"""

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """
    Single RNN cell that processes one time step.
    
    h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input to hidden weights
        self.W_ih = nn.Linear(input_size, hidden_size)
        # Hidden to hidden weights
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input at current time step (batch_size, input_size)
            h: Hidden state from previous step (batch_size, hidden_size)
        
        Returns:
            New hidden state (batch_size, hidden_size)
        """
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


class RNN(nn.Module):
    """
    Full RNN module that processes sequences.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Stack of RNN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(RNNCell(cell_input_size, hidden_size))
    
    def forward(
        self,
        x: torch.Tensor,
        h_0: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch, seq_len, input_size) if batch_first
               else (seq_len, batch, input_size)
            h_0: Initial hidden state (num_layers, batch, hidden_size)
        
        Returns:
            output: All hidden states (batch, seq_len, hidden_size)
            h_n: Final hidden state (num_layers, batch, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, input_size)
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden states
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )
        
        # Process sequence
        h = list(h_0)  # List of hidden states per layer
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(x_t, h[layer_idx])
                x_t = h[layer_idx]  # Input to next layer
            outputs.append(h[-1])  # Output from last layer
        
        output = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden)
        h_n = torch.stack(h, dim=0)  # (num_layers, batch, hidden)
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, h_n


class RNNClassifier(nn.Module):
    """
    RNN for sequence classification tasks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = RNN(embed_dim, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices (batch, seq_len)
        
        Returns:
            Logits (batch, num_classes)
        """
        embedded = self.embedding(x)
        _, h_n = self.rnn(embedded)
        # Use final hidden state from last layer
        return self.fc(h_n[-1])


if __name__ == "__main__":
    # Demo
    batch_size = 4
    seq_len = 10
    input_size = 8
    hidden_size = 16
    num_layers = 2
    
    # Create RNN
    rnn = RNN(input_size, hidden_size, num_layers)
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output, h_n = rnn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Final hidden state shape: {h_n.shape}")
    
    # Classifier demo
    vocab_size = 1000
    num_classes = 5
    
    classifier = RNNClassifier(vocab_size, embed_dim=32, hidden_size=64, num_classes=num_classes)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = classifier(tokens)
    
    print(f"\nClassifier input shape: {tokens.shape}")
    print(f"Classifier output shape: {logits.shape}")
