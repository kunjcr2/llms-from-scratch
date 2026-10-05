# pip install torch if you don't have it
import torch
import torch.nn as nn
import math

class BlockSparseSelfAttention(nn.Module):
    """
    JUST the attention step (Q,K,V -> output) with a simple block-sparse pattern:
      - Dense inside each block
      - Each block's first token (block head) attends to all block heads
      - Every token can attend to its own block head

    Shapes:
      Q, K, V: (batch, heads, seq_len, head_dim)
      return:  (batch, heads, seq_len, head_dim)
    """
    def __init__(self, block_size: int):
        super().__init__()
        assert block_size > 0
        self.block_size = block_size

    @torch.no_grad()
    def _make_mask(self, seq_len: int, device):
        """
        Build a boolean mask M of shape (seq_len, seq_len)
        where M[i, j] = True means token i is allowed to attend to token j.
        Pattern:
          - Dense within each block
          - Block heads attend to all block heads
          - Everyone can attend to its own block head
        """
        bs = self.block_size
        M = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Dense inside each block
        for start in range(0, seq_len, bs):
            end = min(start + bs, seq_len)
            M[start:end, start:end] = True

        # Indices of block heads (0, bs, 2*bs, ...)
        heads = torch.arange(0, seq_len, bs, device=device)

        # Block heads attend to all block heads
        M[heads[:, None], heads[None, :]] = True

        # Every token can attend to its own block head
        # Compute each token's head index: (i // bs) * bs
        token_idx = torch.arange(seq_len, device=device)
        own_head = (token_idx // bs) * bs
        M[token_idx[:, None], own_head[None, :]] = True  # broadcast over j==own_head

        return M  # (S, S)

    def forward(self, Q, K, V):
        """
        Q, K, V: (B, H, S, D)
        """
        B, H, S, D = Q.shape
        device = Q.device

        # Build (and cache if you like) the sparse mask
        M = self._make_mask(S, device)  # (S, S)
        # Expand to (B, H, S, S) by broadcasting
        M4 = M.view(1, 1, S, S)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B, H, S, S)

        # Apply mask: disallow where M==False
        scores = scores.masked_fill(M4, float('-inf'))

        # Softmax on allowed positions only
        attn = torch.softmax(scores, dim=-1)  # (B, H, S, S)

        # Weighted sum
        out = torch.matmul(attn, V)  # (B, H, S, D)
        return out, attn, M


# ---- tiny demo ----
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 24, 32      # batch, heads, seq_len, head_dim
    block_size = 6

    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    attn_layer = BlockSparseSelfAttention(block_size)
    out, attn_weights, mask = attn_layer(Q, K, V)

    print("out shape:", out.shape)               # (1, 2, 24, 32)
    print("attention mask (True=allowed):", mask.shape)
    # For example, check how many keys each query can see:
    print("avg allowed per row:", mask.float().mean(dim=-1).mean().item())
