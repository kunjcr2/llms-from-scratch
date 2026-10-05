import torch  # import torch
import torch.nn as nn  # import neural network module
import torch.nn.functional as F  # import functional API (SDPA)
from rope import Rope  # import provided RoPE implementation

class GQA(nn.Module):
    def __init__(self,
                 d_model: int = 384,
                 n_heads: int = 8,
                 gqa_groups: int = 2,
                 max_len: int = 1024,
                ):
        super().__init__()  # initialize base Module
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # validate head split
        assert n_heads % gqa_groups == 0, "n_heads must be divisible by gqa_groups"  # validate grouping

        self.d_model = d_model  # store model dimension
        self.n_heads = n_heads  # store number of query heads
        self.gqa_groups = gqa_groups  # store number of groups for GQA
        self.head_dim = d_model // n_heads  # compute per-head dimension
        self.n_kv_heads = n_heads // gqa_groups  # compute number of K/V heads for GQA
        self.max_len = max_len  # store max sequence length

        # Define bias-free linear projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)  # Q projection: d_model -> H*D
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)  # K projection: d_model -> H_kv*D
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)  # V projection: d_model -> H_kv*D
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)  # Output projection: H*D -> d_model

        # Instantiate two RoPE modules with the exact composite dims requested
        self.rope_q = Rope(d_model=n_heads * self.head_dim, max_len=max_len)  # RoPE for Q (expects (B,T,H*D))
        self.rope_k = Rope(d_model=self.n_kv_heads * self.head_dim, max_len=max_len)  # RoPE for K (expects (B,T,H_kv*D))

    def forward(self,
                x: torch.Tensor,  # (B, T, d_model)
                attention_mask: torch.Tensor | None = None  # (B, T) 1=real,0=pad (ignored: no attention bias)
                ) -> torch.Tensor:  # returns (B, T, d_model)
        B, T, C = x.shape  # unpack input shape

        # Linear projections for Q, K, V
        q = self.q_proj(x)  # (B, T, H*D)
        k = self.k_proj(x)  # (B, T, H_kv*D)
        v = self.v_proj(x)  # (B, T, H_kv*D)

        # Apply RoPE to Q over the flattened head dimension
        q = self.rope_q(q)  # (B, T, H*D) with rotary positional encoding applied
        q = q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # (B, H, T, D)

        # Apply RoPE to K over the flattened head dimension
        k = self.rope_k(k)  # (B, T, H_kv*D) with rotary positional encoding applied
        k = k.view(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # (B, H_kv, T, D)

        # Reshape V to heads (no RoPE for V)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # (B, H_kv, T, D)

        # Expand K and V from n_kv_heads to n_heads via repeat_interleave on the head axis
        expand_factor = self.n_heads // self.n_kv_heads  # compute replication factor
        k = k.repeat_interleave(expand_factor, dim=1)  # (B, H, T, D)
        v = v.repeat_interleave(expand_factor, dim=1)  # (B, H, T, D)
        # Above thing converts [1,2,3,4] -> [1,1,1,2,2,2,3,3,3,4,4,4] when expand_factor is 3 and dim=0

        # Compute SDPA with purely causal masking (no external attention bias)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)  # (B, H, T, D)

        # Merge heads back to (B, T, H*D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.n_heads * self.head_dim)  # (B, T, d_model)

        # Project to output dimension
        out = self.o_proj(out)  # (B, T, d_model)

        return out  # return attended representations
