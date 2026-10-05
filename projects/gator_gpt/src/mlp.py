import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    SwiGLU MLP for a decoder-only Transformer block.

    - d_model: 384
    - d_ff: ~768 (≈2.5 × d_model)
    - Linear layers are bias-free
    - RMSNorm is applied outside this module
    - Input/Output shape: (batch, seq_len, d_model)
    - BF16-friendly: uses ops that preserve input dtype
    """
    def __init__(self, d_model: int = 384, d_ff: int = 768):
        super().__init__()
        # Fused "up" + "gate" projection to reduce matmuls: d_model -> 2*d_ff
        self.w1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        # Down projection: d_ff -> d_model
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        up, gate = self.w1(x).chunk(2, dim=-1)  # (B, T, d_ff) each

        # We split in two because SwiGLU works like that and it takes -
            # First half which is content
            # Second half which is how much of content in the context
        x = up * F.silu(gate)                   # SwiGLU: up ⊗ swish(gate)
        x = self.w2(x)                          # (B, T, d_model)
        return x