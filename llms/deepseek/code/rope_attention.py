import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import Rope

class RopeAttention(nn.Module):

    """
    RopeAttention module that implements a variant of the attention mechanism
    with Rotary Position Embeddings (RoPE) and down-projection for queries, keys and values.

    Args:
        d_model (int): Dimension of the input embeddings.
        n_heads (int): Number of attention heads.
        kv_latent_dim (int): Latent dimension for keys and values after down-projection.
        vocab_size (int): Size of the vocabulary for output projection.

    Returns:
        out (torch.Tensor): Output tensor after applying attention and projection.
        c_kv (torch.Tensor): Key-value cache after down-projection.
        k_r (torch.Tensor): RoPE cache for keys.
    """

    def __init__(self, d_model, n_heads, kv_latent_dim, vocab_size):
        super().__init__()
        self.d_model = d_model # Dimensiweon of embeddings
        self.n_heads = n_heads # Number of heads
        self.dh = d_model // n_heads # dimensions of heads

        self.rope = Rope(d_model, 20000) # RoPE instance

        self.W_dq = nn.Linear(d_model, kv_latent_dim, bias = False) # Query down projection
        self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False) # Down projection

        self.W_uk = nn.Linear(kv_latent_dim, d_model, bias = False) # Up projection to Keys
        self.W_uv = nn.Linear(kv_latent_dim, d_model, bias = False) # Up projection to values
        self.W_uq = nn.Linear(kv_latent_dim, d_model, bias = False) # Up projection to queries

        self.W_qr = nn.Linear(kv_latent_dim, d_model, bias = False) # Query projection for RoPE
        self.W_kr = nn.Linear(d_model, self.dh, bias = False) # Key projection for RoPE

        self.W_o = nn.Linear(d_model, vocab_size, bias = False) # Output projection
        self.ln_kv = nn.LayerNorm(kv_latent_dim) # Layer norm for kv
        self.ln_kr = nn.LayerNorm(self.dh) # Layer norm for kr

    def forward(self, x, kv_cache=None, kr_cache=None, past_length=0):
        B, S, D = x.size() # Batch size, sequence length, and embedding dimension

        # Query down projection and attention scores
        c_q = self.ln_kv(self.W_dq(x)) # (B, S, kv_latent_dim) - down projection

        #### WITHOUT ROPE ####

        # queries up projection, first
        q_c = self.W_uq(c_q).view(B, S, self.n_heads, self.dh) # (B, S, num_heads, head_dim)

        # Keys and values down projection
        new_c_kv = self.ln_kv(self.W_dkv(x)) # (B, S, kv_latent_dim) - down projection

        # update cache
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache, new_c_kv], dim=1)

        S_full = c_kv.size(1) # number of tokens in total cache

        # keys and values up projection
        k_c = self.W_uk(c_kv).view(B, S_full, self.n_heads, self.dh) # (B, S_full, num_heads, head_dim)
        v_c = self.W_uv(c_kv).view(B, S_full, self.n_heads, self.dh) # (B, S_full, num_heads, head_dim)

        #### WITH ROPE ####

        # queries up projection
        q_r = self.rope(self.W_qr(c_q)).view(B, S, self.n_heads, self.dh) # (B, S, num_heads, head_dim)

        # Keys up projection
        new_kr_cache = self.ln_kr(self.W_kr(x)) # (B, S, dh) - down projection
        if kr_cache is None:
            k_r = new_kr_cache
        else:
            k_r = torch.cat([kr_cache, new_kr_cache], dim=1)

        # Multiple heads for keys
        k_r = torch.stack([k_r] * self.n_heads, dim=2) # (B, S_full, num_heads, head_dim)

        #### JOINING ####
        # Concatenate queries and keys
        q = torch.cat([q_c, q_r], dim=3) # (B, S, num_heads, head_dim * 2)
        k = torch.cat([k_c, k_r], dim=3) # (B, S_full, num_heads, head_dim * 2)

        # Compute attention scores
        # Permute to (B, num_heads, S, head_dim*2) and (B, num_heads, S_full, head_dim*2)
        q_permuted = q.permute(0, 2, 1, 3) # (B, num_heads, S, head_dim*2)
        k_permuted = k.permute(0, 2, 3, 1) # (B, num_heads, head_dim*2, S_full) - Transpose last two for matmul

        attn_scores = torch.matmul(q_permuted, k_permuted) # (B, num_heads, S, S_full)

        # Mask, softmax, and dropout
        attn_scores = attn_scores / ((self.dh * 2) ** 0.5) # Scale by combined head dim
        mask = torch.tril(torch.ones((S, S_full), device=x.device), diagonal=past_length)
        attn_scores = attn_scores.masked_fill(mask.view(1, 1, S, S_full) == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, num_heads, S, S_full)

        # Context vector using attention weights and values
        # v_c is (B, S_full, num_heads, head_dim), permute to (B, num_heads, S_full, head_dim)
        v_c_permuted = v_c.permute(0, 2, 1, 3)
        out_heads = torch.matmul(attn_weights, v_c_permuted) # (B, num_heads, S, head_dim)

        out_heads = out_heads.permute(0, 2, 1, 3).contiguous().view(B, S, self.d_model) # Permute back and combine heads

        out = self.W_o(out_heads) # (B, S, d_model) - output projection
        
        return out, c_kv, k_r  # Return output, key-value cache, and key RoPE cache
    
#### Few changes were made like using .permute instead of .transpose for clarity,
    # and using .contiguous() to ensure the output tensor is contiguous in memory.
    # Also, the comments were updated for better understanding.
    # And at the end, attention score caluclation was simplified by removing the unnecessary transpose.
