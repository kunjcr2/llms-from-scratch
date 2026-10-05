import torch
import torch.nn as nn
import torch.nn.functional as F

class RopelessAttention(nn.Module):
  def __init__(self, d_model, n_heads, kv_latent_dim):
    super().__init__()
    self.d_model = d_model # Dimension of embeddings
    self.n_heads = n_heads # Number of heads
    self.dh = d_model // n_heads # dimensions of heads

    self.W_q = nn.Linear(d_model, d_model, qkv_bias = False) # Wuery projection
    self.W_dkv = nn.Linear(d_model, kv_latent_dim, qkv_bias=False) # Down projection
    self.W_uk = nn.Linear(kv_latent_dim, d_model, qkv_bias = False) # Up projection to Keys
    self.W_uv = nn.Linear(kv_latent_dim, d_model, qkv_bias = False) # Up projection to values
    self.W_o = nn.Linear(d_model, d_model, qkv_bias = False) # Output projection

    self.ln = nn.LayerNorm(kv_latent_dim) # Layer norm
    self.register_buffer('absorbed_k', None) # Holds W_q @ W_uk

  def forward(self, x, kv_cache=None, past_length=0):
    B, S, D = x.size()

    # Computing absorbed query once: W_q @ W_uk.T, Shape: (D, kv_latent_dim)
      # Absorbed query matrix
    if self.absorbed_k is None:
      # Matmul directly transposes the second weight matrix
      absorbed = torch.matmul(self.W_q.weight, self.W_uk.weight) # dim: (D, kv)
      self.absorbed_k = absorbed.view(self.n_heads, self.dh, -1) # (num_heads, head_dim, latent_dim)

    # Calculating kv_cache for new token
      # If we dont have kv_cache, we assign new_kv_cache to variable c_kv
    new_c_kv = self.ln(self.W_dkv(x)) # (B, S, kv_latent_dim)
    if kv_cache is None:
      c_kv = new_c_kv
    else: # If we have alod cache, we join them
      c_kv = torch.cat([kv_cache, new_c_kv], dim=1) # (B, s_full, kv_latent_dim)

    S_full = c_kv.size(1)

    # Working on values matrix
    v_full = self.W_uv(c_kv) # (B, S_full, D)
    v = v_full.view(B, S_full, self.n_heads, self.dh) # (B, S_full, num_heads, head_dim)

    # Breaking input x since W_q is absorbed
    q = x.view(B, S, self.n_heads, self.dh) # (B, S, num_heads, head_dim)

    # Computing attention scores for the last token ONLY
    attn_scores = torch.zeroes(B, self.n_heads, S, S_full, devoce=x.device)
    # We first multiply first head of input with first head of absorbed query
      # Then we multiply the product with transpose of c_kv to get the attention scores
    for h in range(self.n_heads):
      tmp = torch.matmul(q[:, :, h, :], self.absorbed_k[h, :, :]) # (B, S, kv_latent_dim)
      attn_scores[:, h, :, :] = torch.bmm(tmp, c_kv.transpose(1,2)) # (B, S, kv_latent_dim)@(B, kv_latent_dim, s_full)=(B, S, S_full)

    attn_scores = attn_scores / (self.dh**0.5) # variance near 1
    mask = torch.tril(torch.ones((S, S_full), device=x.device), diagonal=past_length) # (S, S_full)
    attn_scores = attn_scores.masked_fill(mask.view(1, 1, S, S_full) == 0, float('-inf'))

    # Softmax on scores to get weights
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Applying weights to each head o V sepratey
    out_heads = []
    for h in range(self.n_heads):
      context_h = torch.matmul(attn_weights[:, h, :, :], v[:, :, h, :])
      out_heads.append(context_h)

    # concating all the out put heads together
    out = torch.cat(out_heads, dim=-1)

    # Returning after output projection
    return self.W_o(out), c_kv