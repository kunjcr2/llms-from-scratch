"""
DeepSeek-Style Compressed Sparse Attention (CSA)
-------------------------------------------------
Educational implementation of all 6 components from DeepSeek V4's CSA:

  1. Shared KV projection          (D → C, MQA-style)
  2. Per-dimension weighted compression  (data-dependent, not averaging)
  3. Overlapping windows           (no hard group boundaries)
  4. DSA sparse selection          (top-k relevant summaries only)
  5. Low-rank query projection     (LoRA-style W_Q decomposition)
  6. Normalization + attention sink logits  (training stability)

NOT production code. Written to be read and understood.
Real DeepSeek V4 also uses RoPE positional embeddings — omitted here for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# Component 6 helper: RMS Normalization
# Used to normalize queries and compressed KV entries before attention
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Simpler than LayerNorm — no mean subtraction, just scale by RMS.
    DeepSeek uses this on query heads and compressed KV entries.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # learned per-dim scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


# ─────────────────────────────────────────────────────────────────────────────
# Main CSA Module
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekCSA(nn.Module):
    def __init__(
        self,
        d_model: int = 128,       # D: hidden state dimension (7168 in real DeepSeek V4)
        n_heads: int = 4,          # H: number of query heads
        kv_dim: int = 32,          # C: shared KV dimension (512 in real DeepSeek V4)
        local_window: int = 8,     # w: recent tokens that get exact attention
        compress_ratio: int = 4,   # r: how many tokens get compressed into 1 summary
        overlap: int = 2,          # how many tokens overlap between adjacent compression windows
        top_k: int = 4,            # DSA: how many summaries to select per query
        lora_rank: int = 16,       # rank for low-rank query projection
        n_sink_logits: int = 2,    # number of learnable attention sink logits
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_dim = kv_dim
        self.d_head = kv_dim       # each head attends to kv_dim-dim entries
        self.local_window = local_window
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        self.top_k = top_k
        self.n_sink_logits = n_sink_logits

        # ── Component 1: Shared KV projection ────────────────────────────────
        # Projects D-dim hidden state → C-dim KV entry
        # All heads share one KV entry per token (MQA-style)
        self.W_kv = nn.Linear(d_model, kv_dim, bias=False)

        # ── Component 2: Per-dimension compression weights ────────────────────
        # For each token in a group, produce a C-dim score vector
        # (one score per KV dimension, not one scalar for the whole vector)
        self.W_z = nn.Linear(d_model, kv_dim, bias=False)

        # ── Component 5: Low-rank query projection ────────────────────────────
        # Instead of one big W_Q of shape [D, n_heads * kv_dim],
        # decompose into down-projection + up-projection (like LoRA)
        # Reduces parameters from D*H*C to D*r + r*H*C
        self.W_q_down = nn.Linear(d_model, lora_rank, bias=False)
        self.W_q_up   = nn.Linear(lora_rank, n_heads * kv_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(n_heads * kv_dim, d_model, bias=False)

        # ── Component 6a: Normalization ───────────────────────────────────────
        # Normalize queries (per head) and compressed KV entries
        # Prevents attention logits from exploding at long sequences
        self.q_norm  = RMSNorm(kv_dim)    # applied per head
        self.kv_norm = RMSNorm(kv_dim)    # applied to each compressed summary

        # ── Component 6b: Attention sink logits ──────────────────────────────
        # Learnable scalar logits added to attention scores
        # They absorb excess attention mass so the model doesn't over-focus
        # on a few tokens — improves training stability
        # Shape: [n_heads, n_sink_logits] — one set of sinks per head
        self.sink_logits = nn.Parameter(torch.zeros(n_heads, n_sink_logits))

    # ─────────────────────────────────────────────────────────────────────────
    # Component 2 + 3: Per-dimension weighted compression with overlapping windows
    # ─────────────────────────────────────────────────────────────────────────

    def compress_with_overlap(
        self,
        h_old: torch.Tensor,     # [B, T_old, D]  hidden states of old tokens
        kv_old: torch.Tensor,    # [B, T_old, C]  KV entries of old tokens
    ) -> torch.Tensor:           # [B, n_summaries, C]
        """
        Compress old tokens into summary tokens using:
          - Per-dimension importance weighting (not scalar, not averaging)
          - Overlapping windows (each token contributes to multiple summaries)

        How overlapping works:
          compress_ratio=4, overlap=2 means each window has 4 tokens
          and shifts by (4-2)=2 tokens instead of 4.
          So windows are: [0,1,2,3], [2,3,4,5], [4,5,6,7], ...
          tok_2 and tok_3 appear in both window 0 and window 1.
        """
        B, T_old, D = h_old.shape
        r    = self.compress_ratio
        step = r - self.overlap      # how many tokens to advance each window

        if T_old < r:
            # Not enough tokens to form even one compression window
            return None

        summaries = []

        # Slide a window of size r across the old tokens, stepping by `step`
        for start in range(0, T_old - r + 1, step):
            end = start + r

            h_window  = h_old[:, start:end, :]    # [B, r, D]
            kv_window = kv_old[:, start:end, :]   # [B, r, C]

            # Per-dimension scores: each token gets a C-dim score vector
            # W_z maps each token's hidden state to C scores
            scores = self.W_z(h_window)            # [B, r, C]

            # Softmax ACROSS the r tokens, per dimension
            # dim=1 means: for each of the C dimensions,
            # the r tokens compete for how much they contribute
            weights = F.softmax(scores, dim=1)     # [B, r, C]

            # Weighted sum: different dims can come from different tokens
            summary = (weights * kv_window).sum(dim=1)  # [B, C]
            summaries.append(summary)

        if not summaries:
            return None

        # Stack all summaries: list of [B, C] → [B, n_summaries, C]
        return torch.stack(summaries, dim=1)

    # ─────────────────────────────────────────────────────────────────────────
    # Component 4: DSA — DeepSeek Sparse Attention (top-k selection)
    # ─────────────────────────────────────────────────────────────────────────

    def dsa_select_topk(
        self,
        q_t: torch.Tensor,         # [B, H, 1, d_head]  query for current token
        summaries: torch.Tensor,   # [B, n_summaries, C]  all compressed summaries
    ) -> torch.Tensor:             # [B, top_k, C]  selected summaries only
        """
        Lightning indexer: quickly score all summaries and keep only top-k.

        Scoring: cheap dot product between the query and each summary.
        We average across heads for a single relevance score per summary.

        In real DeepSeek this uses a more optimized indexing kernel.
        """
        B, n_summaries, C = summaries.shape
        k = min(self.top_k, n_summaries)  # can't select more than we have

        # Expand summaries for all heads: [B, n_summaries, C] → [B, H, n_summaries, C]
        summaries_h = summaries.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # Cheap dot product scores: [B, H, 1, d_head] x [B, H, C, n_summaries]
        # → [B, H, 1, n_summaries]
        scores = torch.matmul(q_t, summaries_h.transpose(-2, -1))  # [B, H, 1, n_summaries]

        # Average across heads to get one score per summary: [B, n_summaries]
        scores_avg = scores.squeeze(2).mean(dim=1)

        # Top-k indices by score
        topk_indices = scores_avg.topk(k, dim=-1).indices  # [B, k]

        # Gather the selected summaries
        # topk_indices: [B, k] → expand to [B, k, C] for gathering
        idx = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        selected = summaries.gather(dim=1, index=idx)      # [B, k, C]

        return selected

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]

        For each token t:
          1. Compute shared KV entries for all tokens
          2. Compute low-rank query
          3. Get local window KV (exact)
          4. Compress old tokens with per-dim weighting + overlapping windows
          5. DSA: select top-k summaries
          6. Normalize queries and compressed KV
          7. Attend to: sink logits + top-k summaries + local window
          8. Output projection
        """
        B, T, D = x.shape

        # ── Component 1: Shared KV projection for all tokens ─────────────────
        kv = self.W_kv(x)    # [B, T, C]

        # ── Component 5: Low-rank query projection ────────────────────────────
        q = self.W_q_up(self.W_q_down(x))             # [B, T, H*C]
        q = q.reshape(B, T, self.n_heads, self.d_head) # [B, T, H, d_head]
        q = q.permute(0, 2, 1, 3)                      # [B, H, T, d_head]

        # ── Component 6a: Normalize queries per head ──────────────────────────
        q = self.q_norm(q)    # RMSNorm applied to last dim (d_head = kv_dim)

        outputs = []

        for t in range(T):
            q_t = q[:, :, t:t+1, :]    # [B, H, 1, d_head]  query for token t

            # ── Local window: exact attention ─────────────────────────────────
            local_start = max(0, t - self.local_window + 1)
            kv_local = kv[:, local_start:t+1, :]    # [B, local_len, C]

            # ── Old tokens: compress + DSA select ────────────────────────────
            kv_selected = None
            if local_start > 0:
                h_old  = x[:, :local_start, :]      # hidden states of old tokens
                kv_old = kv[:, :local_start, :]     # KV entries of old tokens

                # Component 2+3: compress with per-dim weighting + overlapping
                summaries = self.compress_with_overlap(h_old, kv_old)

                if summaries is not None:
                    # Component 6a: normalize compressed KV entries
                    summaries = self.kv_norm(summaries)   # [B, n_summaries, C]

                    # Component 4: DSA top-k selection
                    kv_selected = self.dsa_select_topk(q_t, summaries)  # [B, k, C]

            # ── Build full KV set: selected summaries + local window ──────────
            kv_parts = []
            if kv_selected is not None:
                kv_parts.append(kv_selected)
            kv_parts.append(kv_local)
            kv_all = torch.cat(kv_parts, dim=1)     # [B, n_kv, C]

            # Expand to all heads (MQA: shared KV across heads)
            # [B, n_kv, C] → [B, H, n_kv, d_head]
            kv_h = kv_all.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

            # ── Attention scores ──────────────────────────────────────────────
            scale = math.sqrt(self.d_head)
            attn_scores = torch.matmul(q_t, kv_h.transpose(-2, -1)) / scale
            # attn_scores: [B, H, 1, n_kv]

            # ── Component 6b: Add attention sink logits ───────────────────────
            # sink_logits: [H, n_sink_logits] → [1, H, 1, n_sink_logits]
            sinks = self.sink_logits.unsqueeze(0).unsqueeze(2)  # [1, H, 1, n_sinks]
            sinks = sinks.expand(B, -1, -1, -1)                 # [B, H, 1, n_sinks]

            # Concatenate sink logits before softmax
            # Sinks absorb excess attention mass so real tokens get stable weights
            attn_scores_with_sinks = torch.cat([sinks, attn_scores], dim=-1)
            # [B, H, 1, n_sinks + n_kv]

            attn_w = F.softmax(attn_scores_with_sinks, dim=-1)
            # [B, H, 1, n_sinks + n_kv]

            # Drop the sink weights — they absorbed mass but produce no output
            attn_w = attn_w[:, :, :, self.n_sink_logits:]
            # [B, H, 1, n_kv]

            # ── Weighted sum of values ────────────────────────────────────────
            out_h = torch.matmul(attn_w, kv_h)               # [B, H, 1, d_head]
            out   = out_h.permute(0, 2, 1, 3).reshape(B, 1, self.n_heads * self.d_head)
            outputs.append(out)

        # Stack all token outputs: [B, T, H*d_head]
        out = torch.cat(outputs, dim=1)

        # ── Output projection ─────────────────────────────────────────────────
        return self.W_o(out)    # [B, T, D]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, D = 2, 32, 128

    model = DeepSeekCSA(
        d_model=128,
        n_heads=4,
        kv_dim=32,           # C: shared KV dimension
        local_window=8,      # exact attention on last 8 tokens
        compress_ratio=4,    # group every 4 old tokens into 1 summary
        overlap=2,           # adjacent windows share 2 tokens
        top_k=4,             # DSA: pick top 4 summaries
        lora_rank=16,        # low-rank query projection rank
        n_sink_logits=2,     # 2 attention sink logits per head
    )

    x = torch.randn(B, T, D)
    out = model(x)

    print("=" * 60)
    print("DeepSeek-Style CSA Demo")
    print("=" * 60)
    print(f"Input shape:   {x.shape}   [B, T, D]")
    print(f"Output shape:  {out.shape}  [B, T, D]")
    print()

    # Parameter breakdown
    total = sum(p.numel() for p in model.parameters())
    print(f"{'Component':<35} {'Parameters':>12}")
    print("-" * 50)
    print(f"{'W_kv (shared KV projection)':<35} {sum(p.numel() for p in model.W_kv.parameters()):>12,}")
    print(f"{'W_z (per-dim compression weights)':<35} {sum(p.numel() for p in model.W_z.parameters()):>12,}")
    print(f"{'W_q_down (low-rank query down)':<35} {sum(p.numel() for p in model.W_q_down.parameters()):>12,}")
    print(f"{'W_q_up (low-rank query up)':<35} {sum(p.numel() for p in model.W_q_up.parameters()):>12,}")
    print(f"{'W_o (output projection)':<35} {sum(p.numel() for p in model.W_o.parameters()):>12,}")
    print(f"{'q_norm + kv_norm (RMSNorm)':<35} {sum(p.numel() for p in list(model.q_norm.parameters()) + list(model.kv_norm.parameters())):>12,}")
    print(f"{'sink_logits':<35} {model.sink_logits.numel():>12,}")
    print("-" * 50)
    print(f"{'TOTAL':<35} {total:>12,}")
    print()

    # Show effective KV size per token vs standard attention
    print(f"{'Token':<8} {'Local (exact)':<16} {'Max summaries':<16} {'DSA top-k':<12} {'Total KV':<12} {'Standard attn':<14}")
    print("-" * 80)
    r    = model.compress_ratio
    ovlp = model.overlap
    step = r - ovlp
    w    = model.local_window
    k    = model.top_k

    for t in [0, 4, 8, 16, 24, 31]:
        local_start = max(0, t - w + 1)
        local_len   = t - local_start + 1
        old_len     = local_start
        n_summaries = max(0, (old_len - r) // step + 1) if old_len >= r else 0
        selected    = min(k, n_summaries)
        total_kv    = local_len + selected
        print(f"{t:<8} {local_len:<16} {n_summaries:<16} {selected:<12} {total_kv:<12} {t+1:<14}")

    print()
    print(f"At T=31: CSA attends to {local_len + selected} tokens vs {T} in standard attention.")
    print(f"Savings grow massively at 1M tokens.")