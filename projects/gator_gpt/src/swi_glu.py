import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        # project input into [u | g]
        self.in_proj = nn.Linear(d_model, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def silu(self, x):
        # manual SiLU = x * sigmoid(x)
        return x * torch.sigmoid(x)

    def forward(self, x):
        u, g = self.in_proj(x).chunk(2, dim=-1)  # split last dim
        h = self.silu(u) * g                     # SwiGLU activation
        h = self.dropout(h)
        return self.out_proj(h)

# --- example ---
if __name__ == "__main__":
    x = torch.randn(4, 10, 32)   # (batch, seq, d_model)
    ffn = SwiGLU(d_model=32, hidden_dim=96, dropout=0.1)
    y = ffn(x)
    print(y.shape)  # -> torch.Size([4, 10, 32])
