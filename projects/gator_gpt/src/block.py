import torch
import torch.nn as nn
import torch.nn.functional as F
from gqa import GQA
from mlp import MLP

class Block(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        gqa_groups: int = 2,
        max_len: int = 1024,
        d_ff: int = 768,
        eps: float = 1e-5,
        dropout_p: float = 0.0,  # keep 0.0 for pretrain
    ):
        super().__init__()
        self.rms1 = nn.modules.normalization.RMSNorm(d_model, eps)
        self.rms2 = nn.modules.normalization.RMSNorm(d_model, eps)

        self.attn = GQA(d_model, n_heads, gqa_groups, max_len)  # should include proj_out
        self.mlp = MLP(d_model, d_ff)  # your SwiGLU MLP (bias-free)

        self.drop_attn = nn.Dropout(dropout_p)
        self.drop_mlp = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-rms
        x = x + self.drop_attn(self.attn(self.rms1(x)))
        x = x + self.drop_mlp(self.mlp(self.rms2(x)))
        return x
