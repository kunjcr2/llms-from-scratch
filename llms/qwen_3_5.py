"""
QWEN3.5 ARCHITECTURE
Source: NVIDIA blog (Mar 2026), HuggingFace model card
No technical report yet.
Early Fusion: No separate vision encoder. Hybrid GDN + GatedAttention + MoE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Interleaved MRoPE: Unified position encoding for text + image + video.
# Assigns t/h/v axes in round-robin across embedding dim pairs so every
# axis gets equal share of low and high frequency rotation bands.
# ─────────────────────────────────────────────────────────────────────────────


class InterleavedMRoPE(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        assert head_dim % 2 == 0
        num_pairs = head_dim // 2
        self.axis_assignment = [["t", "h", "v"][i % 3] for i in range(num_pairs)]
        freqs = 1.0 / (theta ** (torch.arange(0, num_pairs).float() / num_pairs))
        self.register_buffer("base_freqs", freqs)

    def compute_freqs(self, position_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, N = position_ids["t"].shape
        num_pairs = len(self.axis_assignment)
        angles = torch.zeros(B, N, num_pairs, device=position_ids["t"].device)
        for i, axis in enumerate(self.axis_assignment):
            angles[:, :, i] = position_ids[axis].float() * self.base_freqs[i]
        return torch.polar(torch.ones_like(angles), angles)

    def apply(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        x_c = torch.view_as_complex(x.float().reshape(B, H, N, -1, 2))
        rotated = x_c * freqs.unsqueeze(1)
        return torch.view_as_real(rotated).reshape(B, H, N, D).type_as(x)


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm: Divides by root-mean-square only (no mean subtraction).
# Faster than LayerNorm, same performance. Used before every sub-layer.
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU MLP: Gated FFN. Two parallel paths (gate + up) multiplied together,
# then projected down. Swish activation on gate path gives smooth gating.
# Used inside each MoE expert.
# ─────────────────────────────────────────────────────────────────────────────


class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: Optional[int] = None):
        super().__init__()
        self.intermediate_dim = intermediate_dim or int(hidden_dim * 8 / 3)
        self.W_gate = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)
        self.W_up = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)
        self.W_down = nn.Linear(self.intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


# ─────────────────────────────────────────────────────────────────────────────
# PatchEmbedding: Replaces the entire SigLIP-2 + MLP merger pipeline.
# In early fusion, image patches are just linearly projected into the same
# hidden space as text tokens. No separate encoder needed.
# ─────────────────────────────────────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    def __init__(
        self, patch_size: int = 16, in_channels: int = 3, hidden_dim: int = 7168
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.proj(image)  # (B, hidden_dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Gated DeltaNet: Linear attention O(N) — maintains a fixed-size state matrix
# instead of attending over all previous tokens. Gate controls how much of
# the old state to keep vs replace per step. Cheap but lossy for long-range.
# ─────────────────────────────────────────────────────────────────────────────


class GatedDeltaNet(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.W_gate = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)
        self.W_o = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self, x: torch.Tensor, position_ids: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_q(x).view(B, N, H, D)
        K = F.normalize(self.W_k(x).view(B, N, H, D), dim=-1)
        V = self.W_v(x).view(B, N, H, D)
        G = torch.sigmoid(self.W_gate(x).view(B, N, H, D))

        # MRoPE on Q and K (position awareness even in linear attention)
        mrope = InterleavedMRoPE(D).to(x.device)
        freqs = mrope.compute_freqs(position_ids)
        Q = mrope.apply(Q.transpose(1, 2), freqs).transpose(1, 2)
        K = mrope.apply(K.transpose(1, 2), freqs).transpose(1, 2)

        # Recurrent state per head: (B, H, D, D)
        state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        outputs = []

        # Q, K, V, G, S -> (B, N, H, D)
        for t in range(N):
            q_t = Q[:, t]  # (B, H, D)
            k_t = K[:, t]  # (B, H, D)
            v_t = V[:, t]  # (B, H, D)
            g_t = G[:, t].unsqueeze(-1)  # (B, H, D, 1)

            pred = torch.einsum(
                "bhk,bhkv->bhv", k_t, state
            )  # delta rule: what state predicts
            delta = torch.einsum("bhk,bhv->bhkv", k_t, v_t - pred)
            state = g_t * state + delta  # gated update

            out_t = torch.einsum("bhk,bhkv->bhv", q_t, state)  # read from state
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1).reshape(B, N, H * D)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────────────────
# Gated Full Attention: Standard O(N²) attention + learned output gate.
# The gate acts as a "dimmer switch" — even after attention decides what to
# focus on, the gate controls how much flows forward. Used every ~5th layer
# for precise long-range recall that GDN's state compression misses.
# ─────────────────────────────────────────────────────────────────────────────


class GatedFullAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_gate = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.mrope = InterleavedMRoPE(self.head_dim)

    def forward(
        self, x: torch.Tensor, position_ids: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        B, N, C = x.shape
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # QK-Norm
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # Interleaved MRoPE
        freqs = self.mrope.compute_freqs(position_ids)
        Q = self.mrope.apply(Q, freqs)
        K = self.mrope.apply(K, freqs)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, C)
        out = self.W_o(out)

        gate = torch.sigmoid(self.W_gate(x))
        return gate * out


# ─────────────────────────────────────────────────────────────────────────────
# MoE Layer: 512 experts, top-10 routed + 1 shared = 11 active per token.
# Higher sparsity than Qwen3-VL (4.28% vs 9.4% activation).
# Every layer uses MoE (not just some layers).
# ─────────────────────────────────────────────────────────────────────────────


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 512,
        top_k: int = 10,
        num_shared_experts: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU_MLP(hidden_dim) for _ in range(num_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU_MLP(hidden_dim) for _ in range(num_shared_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_flat = x.view(-1, C)
        scores = F.softmax(self.router(x_flat), dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = topk_indices[:, k] == e
                if mask.any():
                    routed_out[mask] += topk_weights[mask, k : k + 1] * self.experts[e](
                        x_flat[mask]
                    )

        shared_out = sum(s(x_flat) for s in self.shared)
        return (routed_out + shared_out).view(B, N, C)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Transformer Block: Alternates between GDN (cheap O(N) linear attn)
# and GatedFullAttention (precise O(N²)). Every ~5th layer is full attention.
# All layers use MoE FFN.
# ─────────────────────────────────────────────────────────────────────────────


class HybridTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        layer_idx: int,
        full_attn_every_n: int = 5,
        num_experts: int = 512,
        top_k: int = 10,
        num_shared_experts: int = 1,
    ):
        super().__init__()
        self.norm_1 = RMSNorm(hidden_dim)
        self.norm_2 = RMSNorm(hidden_dim)
        head_dim = hidden_dim // num_heads
        use_full_attn = (layer_idx + 1) % full_attn_every_n == 0
        self.attn = (
            GatedFullAttention(hidden_dim, num_heads)
            if use_full_attn
            else GatedDeltaNet(hidden_dim, num_heads, head_dim)
        )
        self.ffn = MoELayer(hidden_dim, num_experts, top_k, num_shared_experts)

    def forward(
        self, x: torch.Tensor, position_ids: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), position_ids)
        x = x + self.ffn(self.norm_2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Token Prediction Head: Predicts N+1, N+2, N+3, N+4 simultaneously.
# Richer training signal (must understand deeper structure to predict 4 ahead).
# Enables speculative decoding at inference for faster generation.
# ─────────────────────────────────────────────────────────────────────────────


class MultiTokenPredictionHead(nn.Module):
    def __init__(
        self, hidden_dim: int, vocab_size: int = 248320, num_future_tokens: int = 4
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Linear(hidden_dim, vocab_size, bias=False)
                for _ in range(num_future_tokens)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        return [head(hidden_states) for head in self.heads]


# ─────────────────────────────────────────────────────────────────────────────
# Full Qwen3.5 Model. Early fusion — no separate vision encoder.
# Image patches projected directly into token embedding space.
# Hybrid GDN + GatedAttention layers, every layer MoE, multi-token prediction.
# ─────────────────────────────────────────────────────────────────────────────


class Qwen35(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 7168,
        vocab_size: int = 248320,
        num_layers: int = 60,
        num_heads: int = 56,
        num_experts: int = 512,
        top_k: int = 10,
        num_shared_experts: int = 1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=16, hidden_dim=hidden_dim)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [
                HybridTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    layer_idx=i,
                    full_attn_every_n=5,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_shared_experts=num_shared_experts,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = MultiTokenPredictionHead(
            hidden_dim, vocab_size, num_future_tokens=4
        )

    def forward(
        self, text_tokens: torch.Tensor, images: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        B = text_tokens.shape[0]
        x = self.token_embed(text_tokens)

        if images is not None:
            visual_tokens = self.patch_embed(images)
            x = torch.cat([visual_tokens, x], dim=1)

        N = x.shape[1]
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        position_ids = {"t": pos, "h": pos, "v": pos}

        for layer in self.layers:
            x = layer(x, position_ids)

        x = self.norm(x)
        return self.lm_head(x)


MODEL_SPECS = {
    "total_params": "397B",
    "active_params": "17B (4.28%)",
    "num_experts": 512,
    "active_experts": "10 routed + 1 shared = 11",
    "num_layers": 60,
    "hidden_dim": 7168,
    "vocab_size": 248320,
    "context": "256K (extensible to 1M)",
    "languages": 201,
    "attention": "Hybrid GatedDeltaNet + GatedFullAttention",
    "training": "Early fusion from scratch on interleaved multimodal tokens",
    "precision": "FP8",
    "multi_token_pred": 4,
}
