import torch
import torch.nn as nn
import torch.nn.functional as F

class Rope(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(Rope, self).__init__()
        
        # Store the embedding dimension (must be even for proper 2D rotations)
        assert d_model % 2 == 0, "d_model must be even for proper RoPE implementation"
        self.d_model = d_model
        
        # Store the maximum sequence length this RoPE can handle
        self.max_len = max_len
        
        # Create position indices [0, 1, 2, ..., max_len-1] and add dimension for broadcasting
            # Shape: (max_len, 1) - each position gets its own row
        self.register_buffer('position_ids', torch.arange(max_len).unsqueeze(1))
        
        # Create frequency terms for rotation - these determine how fast each dimension pair rotates
            # Uses exponential decay: smaller indices rotate faster, larger indices rotate slower
            # torch.arange(0, d_model, 2) creates [0, 2, 4, 6, ...] (even indices only)
            # The formula creates frequencies: [1/10000^(0/d), 1/10000^(2/d), 1/10000^(4/d), ...]
            # This is equal to (10000 ** (-(2 * i) / d_model) for i in range(d_model // 2))
        self.register_buffer('div_term', torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)))

    def forward(self, x):
        # Input shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Get position indices for current sequence length (trim to actual sequence length)
            # If input has 100 tokens, this gets positions [0, 1, 2, ..., 99]
        position_ids = self.position_ids[:seq_len]  # Shape: (seq_len, 1)
        
        # Calculate rotation angles for each position and frequency based on 2017 paper
            # Multiply each position by each frequency term to get rotation angles
            # Shape: (seq_len, d_model//2)
            # This is basically: pos/(10000^(2i/d_model))
        angles = position_ids * self.div_term
        
        # Calculate sine and cosine values for rotation
            # Shape: (seq_len, d_model//2)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # Reshape input to separate even and odd dimensions for 2D rotation
            # Split x into pairs: (x_0, x_1), (x_2, x_3), (x_4, x_5), ...
            # Shape: (batch_size, seq_len, d_model//2, 2)
        x_pairs = x.view(batch_size, seq_len, d_model // 2, 2)
        
        # Extract even and odd components
            # x_even contains x_0, x_2, x_4, ... (first element of each pair)
            # x_odd contains x_1, x_3, x_5, ... (second element of each pair)
        x_even = x_pairs[..., 0]  # Shape: (batch_size, seq_len, d_model//2)
        x_odd = x_pairs[..., 1]   # Shape: (batch_size, seq_len, d_model//2)
        
        # Apply 2D rotation to each pair of dimensions
            # Rotation matrix: [[cos, -sin], [sin, cos]]
            # For each pair (x_i, x_{i+1}), compute:
            # x_i' = x_i * cos - x_{i+1} * sin
            # x_{i+1}' = x_i * sin + x_{i+1} * cos
        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals
        
        # Combine rotated even and odd components back into pairs
            # Shape: (batch_size, seq_len, d_model//2, 2)
        rotated_pairs = torch.stack([rotated_even, rotated_odd], dim=-1)
        
        # Reshape back to original format
            # Shape: (batch_size, seq_len, d_model)
        rotated_x = rotated_pairs.view(batch_size, seq_len, d_model)
        
        return rotated_x