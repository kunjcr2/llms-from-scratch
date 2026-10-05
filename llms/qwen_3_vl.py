"""
QWEN3-VL ARCHITECTURE
Source: arxiv 2511.21631
Late Fusion: SigLIP-2 Vision Encoder + MLP Merger + Qwen3 LLM
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# RoPE: Rotary Position Embedding (standard 1D for text within vision encoder)
# Encodes position by rotating Q and K vectors. Tokens further apart
# have larger angular difference → attention naturally decays with distance.
# ─────────────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex exponential frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))   # (dim/2,)
    t = torch.arange(max_seq_len).float()                                # (max_seq_len,)
    freqs = torch.outer(t, freqs)                                        # (max_seq_len, dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)                    # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a tensor. x: (B, H, N, head_dim)"""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x_complex.shape[-2]].unsqueeze(0).unsqueeze(0)        # broadcast (1, 1, N, dim/2)
    rotated = x_complex * freqs
    return torch.view_as_real(rotated).reshape(*x.shape).type_as(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2D-RoPE: Used inside the SigLIP-2 Vision Encoder.
# Images are 2D grids, so we split embedding dims in half:
#   first half rotated by row position, second half by col position.
# ─────────────────────────────────────────────────────────────────────────────

def precompute_2d_rope_freqs(dim: int, max_h: int, max_w: int, theta: float = 10000.0):
    """Precompute 2D rotary frequencies for image patches."""
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))
    rows = torch.arange(max_h).float()
    cols = torch.arange(max_w).float()
    freqs_h = torch.polar(torch.ones_like(torch.outer(rows, freqs)), torch.outer(rows, freqs))
    freqs_w = torch.polar(torch.ones_like(torch.outer(cols, freqs)), torch.outer(cols, freqs))
    return freqs_h, freqs_w


def apply_2d_rope(x: torch.Tensor, freqs_h: torch.Tensor, freqs_w: torch.Tensor,
                  h: int, w: int) -> torch.Tensor:
    """Apply 2D rotary embeddings. x: (B, H, N, head_dim) where N = h*w patches."""
    B, H, N, D = x.shape
    half = D // 2
    x_h = x[..., :half]
    x_w = x[..., half:]

    x_h_c = torch.view_as_complex(x_h.float().reshape(B, H, h, w, -1, 2))
    x_w_c = torch.view_as_complex(x_w.float().reshape(B, H, h, w, -1, 2))

    fh = freqs_h[:h].unsqueeze(1).unsqueeze(0).unsqueeze(0)    # (1, 1, h, 1, dim/4)
    fw = freqs_w[:w].unsqueeze(0).unsqueeze(0).unsqueeze(0)    # (1, 1, 1, w, dim/4)

    x_h_rot = torch.view_as_real(x_h_c * fh).reshape(B, H, N, half)
    x_w_rot = torch.view_as_real(x_w_c * fw).reshape(B, H, N, half)

    return torch.cat([x_h_rot, x_w_rot], dim=-1).type_as(x)


# ─────────────────────────────────────────────────────────────────────────────
# Interleaved MRoPE: Unified position encoding for text + image + video.
# Instead of chunking dims into t/h/v blocks (which starves some axes),
# we interleave: dim0=t, dim1=h, dim2=v, dim3=t, dim4=h, dim5=v, ...
# Every axis gets equal share of low and high frequency bands.
# ─────────────────────────────────────────────────────────────────────────────

class InterleavedMRoPE(nn.Module):
    """
    Multimodal Rotary Position Embedding with interleaved t/h/v assignment.
    Text:  t=h=v=seq_pos  → behaves like standard 1D RoPE
    Image: t=0, h=row, v=col
    Video: t=frame, h=row, v=col
    """

    def __init__(self, head_dim: int, max_positions: int = 262144, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        assert head_dim % 2 == 0, "head_dim must be even"
        num_pairs = head_dim // 2  # each pair of dims shares one frequency

        # Assign each dim-pair to t, h, or v in round-robin
        # pair 0 → t, pair 1 → h, pair 2 → v, pair 3 → t, ...
        self.axis_assignment = [['t', 'h', 'v'][i % 3] for i in range(num_pairs)]

        # Precompute base frequencies per pair
        freqs = 1.0 / (theta ** (torch.arange(0, num_pairs).float() / num_pairs))
        self.register_buffer('base_freqs', freqs)   # (num_pairs,)

    def compute_freqs(self, position_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        position_ids: {"t": (B, N), "h": (B, N), "v": (B, N)}
        Returns: complex freqs (B, N, num_pairs) for rotating Q and K.
        """
        B, N = position_ids['t'].shape
        num_pairs = len(self.axis_assignment)
        angles = torch.zeros(B, N, num_pairs, device=position_ids['t'].device)

        for i, axis in enumerate(self.axis_assignment):
            # angle = position_on_this_axis * frequency_for_this_pair
            angles[:, :, i] = position_ids[axis].float() * self.base_freqs[i]

        return torch.polar(torch.ones_like(angles), angles)   # (B, N, num_pairs)

    def apply(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_heads, N, head_dim)
        freqs: (B, N, num_pairs)  complex
        Returns: rotated x, same shape.
        """
        B, H, N, D = x.shape
        x_complex = torch.view_as_complex(x.float().reshape(B, H, N, -1, 2))  # (B, H, N, num_pairs)
        freqs = freqs.unsqueeze(1)                                               # (B, 1, N, num_pairs)
        rotated = x_complex * freqs
        return torch.view_as_real(rotated).reshape(B, H, N, D).type_as(x)


def build_position_ids(token_types: List[str], text_positions: List[int],
                       patch_grid_info: List[dict], device='cpu') -> Dict[str, torch.Tensor]:
    """
    Build (t, h, v) position IDs for an interleaved multimodal sequence.

    token_types:     "text" | "image" | "video" per token
    text_positions:  running text position counter per token
    patch_grid_info: dict with {row, col, frame_idx} per visual token

    Returns: {"t": (1, N), "h": (1, N), "v": (1, N)}
    """
    t_ids, h_ids, v_ids = [], [], []
    text_pos = 0

    for i, tt in enumerate(token_types):
        if tt == "text":
            t_ids.append(text_pos)
            h_ids.append(text_pos)
            v_ids.append(text_pos)
            text_pos += 1
        elif tt == "image":
            info = patch_grid_info[i]
            t_ids.append(0)
            h_ids.append(info['row'])
            v_ids.append(info['col'])
        elif tt == "video":
            info = patch_grid_info[i]
            t_ids.append(info['frame_idx'])
            h_ids.append(info['row'])
            v_ids.append(info['col'])

    return {
        "t": torch.tensor([t_ids], device=device),
        "h": torch.tensor([h_ids], device=device),
        "v": torch.tensor([v_ids], device=device),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm: Cheaper alternative to LayerNorm.
# Only divides by root-mean-square (no mean subtraction). Same performance.
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU MLP: Gated FFN used in every transformer layer.
# Two parallel projections: W_gate (with Swish activation) and W_up.
# Elementwise multiply gates information flow. Then W_down projects back.
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: Optional[int] = None):
        super().__init__()
        self.intermediate_dim = intermediate_dim or int(hidden_dim * 8 / 3)
        self.W_gate = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)
        self.W_up   = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)
        self.W_down = nn.Linear(self.intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Attention with QKV Bias, QK-Norm, and Interleaved MRoPE.
# QKV Bias: learnable offset on Q, K, V projections (helps stability).
# QK-Norm: normalize Q and K to unit vectors before dot product
#          (prevents attention score explosion at scale).
# MRoPE:  applied to Q and K after normalization for position awareness.
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mrope = InterleavedMRoPE(self.head_dim)

    def forward(self, x: torch.Tensor, position_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        attn   = F.softmax(scores, dim=-1)
        out    = (attn @ V).transpose(1, 2).reshape(B, N, C)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────────────────
# MoE Layer: Mixture of Experts FFN.
# Router scores all experts per token, top-k activate.
# Shared expert always fires for universal knowledge (grammar, basics).
# ─────────────────────────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 64,
                 top_k: int = 2, num_shared_experts: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router      = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts     = nn.ModuleList([SwiGLU_MLP(hidden_dim) for _ in range(num_experts)])
        self.shared      = nn.ModuleList([SwiGLU_MLP(hidden_dim) for _ in range(num_shared_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_flat = x.view(-1, C)
        scores = F.softmax(self.router(x_flat), dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (topk_indices[:, k] == e)
                if mask.any():
                    routed_out[mask] += topk_weights[mask, k:k+1] * self.experts[e](x_flat[mask])

        shared_out = sum(s(x_flat) for s in self.shared)
        return (routed_out + shared_out).view(B, N, C)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block: Pre-norm residual block.
# RMSNorm → Attention (with MRoPE) → residual → RMSNorm → FFN → residual
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int,
                 use_moe: bool = False, num_experts: int = 64, top_k: int = 2):
        super().__init__()
        self.norm_1 = RMSNorm(hidden_dim)
        self.norm_2 = RMSNorm(hidden_dim)
        self.attn   = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn    = MoELayer(hidden_dim, num_experts, top_k) if use_moe else SwiGLU_MLP(hidden_dim)

    def forward(self, x: torch.Tensor, position_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), position_ids)
        x = x + self.ffn(self.norm_2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Vision Encoder: SigLIP-2 ViT with 2D-RoPE for dynamic resolution.
# Patches image into 16x16 tiles, projects to hidden_dim, applies 2D-RoPE
# in each self-attention layer. Returns all layer outputs for DeepStack.
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoderAttention(nn.Module):
    """Self-attention for the vision encoder using 2D-RoPE."""

    def __init__(self, hidden_dim: int, num_heads: int, max_grid: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        freqs_h, freqs_w = precompute_2d_rope_freqs(self.head_dim, max_grid, max_grid)
        self.register_buffer('freqs_h', freqs_h)
        self.register_buffer('freqs_w', freqs_w)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, C = x.shape
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        Q = apply_2d_rope(Q, self.freqs_h, self.freqs_w, h, w)
        K = apply_2d_rope(K, self.freqs_h, self.freqs_w, h, w)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn   = F.softmax(scores, dim=-1)
        out    = (attn @ V).transpose(1, 2).reshape(B, N, C)
        return self.W_o(out)


class VisionEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm_1 = RMSNorm(hidden_dim)
        self.norm_2 = RMSNorm(hidden_dim)
        self.attn   = VisionEncoderAttention(hidden_dim, num_heads)
        self.ffn    = SwiGLU_MLP(hidden_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), h, w)
        x = x + self.ffn(self.norm_2(x))
        return x


class VisionEncoder(nn.Module):
    """
    SigLIP-2 Vision Transformer with dynamic resolution + 2D-RoPE.
    Returns final output AND all intermediate layer outputs for DeepStack.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3,
                 hidden_dim: int = 1152, num_layers: int = 27, num_heads: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList([VisionEncoderBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_dim)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        x = self.proj(image)                          # (B, D, H/P, W/P)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)              # (B, num_patches, D)

        layer_outputs = {}
        for i, layer in enumerate(self.layers):
            x = layer(x, h, w)
            layer_outputs[i] = x

        return self.norm(x), layer_outputs


# ─────────────────────────────────────────────────────────────────────────────
# DeepStack: Combines multi-level ViT features via residual connections.
# Early layers capture edges/textures, late layers capture semantics.
# Combining gives the LLM both fine-grained detail AND high-level meaning.
# ─────────────────────────────────────────────────────────────────────────────

class DeepStack(nn.Module):
    def __init__(self, hidden_dim: int = 1152, extract_layers: Tuple = (5, 10, 15, 20, 26)):
        super().__init__()
        self.extract_layers = extract_layers
        self.projections = nn.ModuleDict({
            str(l): nn.Linear(hidden_dim, hidden_dim, bias=False)
            for l in extract_layers[:-1]
        })

    def forward(self, layer_outputs: Dict[int, torch.Tensor],
                final_output: torch.Tensor) -> torch.Tensor:
        combined = final_output
        for l in reversed(self.extract_layers[:-1]):
            combined = combined + self.projections[str(l)](layer_outputs[l])
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# MLP Merger: Projects vision encoder output space → LLM input space.
# Trained in isolation first (Stage 0) while encoder and LLM are frozen,
# to bridge the modality gap before joint training begins.
# ─────────────────────────────────────────────────────────────────────────────

class MLPMerger(nn.Module):
    def __init__(self, vision_dim: int = 1152, llm_dim: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Video Timestamp Alignment: Injects "<Xs>" text tokens before each frame
# in the sequence. Evolved from T-RoPE (implicit positional time encoding)
# to explicit textual timestamps the LLM can directly read and reason about.
# MRoPE temporal dim still exists — timestamps are an additional signal.
# ─────────────────────────────────────────────────────────────────────────────

def build_video_token_sequence(frames: List[Tuple[float, List]]) -> List:
    token_sequence = []
    for timestamp_sec, patch_tokens in frames:
        token_sequence.append(f"<{timestamp_sec:.1f}s>")
        token_sequence.extend(patch_tokens)
    return token_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Full Qwen3-VL Model.
# Late fusion: image → VisionEncoder → DeepStack → MLPMerger → visual tokens
# Visual tokens prepended to text tokens → LLM with Interleaved MRoPE.
# ─────────────────────────────────────────────────────────────────────────────

class Qwen3VL(nn.Module):
    def __init__(self, hidden_dim: int = 4096, vocab_size: int = 152064,
                 num_layers: int = 64, num_heads: int = 32,
                 num_experts: int = 64, top_k: int = 2):
        super().__init__()
        self.vision_encoder = VisionEncoder(hidden_dim=1152, num_layers=27, num_heads=16)
        self.deepstack      = DeepStack(hidden_dim=1152)
        self.merger         = MLPMerger(vision_dim=1152, llm_dim=hidden_dim)
        self.token_embed    = nn.Embedding(vocab_size, hidden_dim)
        self.layers         = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, use_moe=True,
                             num_experts=num_experts, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.norm    = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        final_out, layer_outputs = self.vision_encoder(image)
        multi_level = self.deepstack(layer_outputs, final_out)
        return self.merger(multi_level)

    def forward(self, text_tokens: torch.Tensor,
                images: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = text_tokens.shape[0]
        x = self.token_embed(text_tokens)

        if images is not None:
            visual_tokens = self.encode_image(images)
            x = torch.cat([visual_tokens, x], dim=1)

        N = x.shape[1]
        # Default: treat all as text positions for simplicity
        # In practice, position_ids built from token_types via build_position_ids()
        pos = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        position_ids = {"t": pos, "h": pos, "v": pos}

        for layer in self.layers:
            x = layer(x, position_ids)

        x = self.norm(x)
        return self.lm_head(x)


PRETRAINING_STAGES = {
    "S0": {"trains": ["MLP Merger only"],    "freezes": ["Vision Encoder", "LLM"], "tokens": "67B",  "seq_len": 8192},
    "S1": {"trains": ["Everything"],         "freezes": [],                         "tokens": "~1T",  "seq_len": 8192},
    "S2": {"trains": ["Everything"],         "freezes": [],                         "tokens": "~1T",  "seq_len": 32768},
    "S3": {"trains": ["Everything"],         "freezes": [],                         "tokens": "100B", "seq_len": 262144},
}

POSTTRAINING = {
    "A_SFT":         {"data": "1.2M samples", "context": ["32K epoch 1", "256K epoch 2"]},
    "B_CoT":         {"data": "Hard reasoning problems, 1:1 VL/text ratio"},
    "C_ReasoningRL": {"algorithm": "SAPO", "queries": "~30K", "sampling": "16 per query"},
    "D_GeneralRL":   {"reward": ["rule-based", "Qwen2.5-VL-72B judge"]},
}