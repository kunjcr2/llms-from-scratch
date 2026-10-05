"""
Segment Anything Model (SAM): From-Scratch Implementation
==========================================================
An educational implementation of SAM for promptable image segmentation,
built from scratch to match the architecture described in the lecture notes.

Key Components:
- Image Encoder: ViT-based encoder (simplified) that produces 256×64×64 embeddings
- Prompt Encoder: Handles sparse prompts (points, boxes) and dense prompts (masks)
  with no transformer — just embeddings, projections, and convolutions
- Mask Decoder: 2× bidirectional cross-attention blocks producing 3 ambiguity-level
  masks + IoU confidence scores

Architecture Flow:
    Image  ──► Image Encoder  ──► (256, 64, 64) image embedding
    Prompt ──► Prompt Encoder ──► sparse tokens (N, 256) + dense (256, 64, 64)
    Both   ──► Mask Decoder   ──► 3 masks (1, 1024, 1024) each + 3 IoU scores

Simplifications vs Original Paper:
- Uses a smaller ViT (configurable depth) instead of ViT-H (632M params)
- Positional encoding uses learnable embeddings (not sinusoidal 2D)
- No CLIP text prompt support (would require a separate CLIP model)

Reference: Kirillov et al., "Segment Anything" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ==============================================================================
# Image Encoder (ViT-based)
# ==============================================================================
class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project to embedding dim.

    Uses Conv2D with kernel_size = stride = patch_size (same trick as ViT).

    Dimension flow:
        (B, 3, 1024, 1024) -> Conv2D -> (B, embed_dim, 64, 64) -> flatten -> (B, 4096, embed_dim)
    """

    def __init__(self, image_size=1024, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 4096 for 1024/16

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 1024, 1024)
        x = self.proj(x)              # (B, embed_dim, 64, 64)
        x = x.flatten(2)              # (B, embed_dim, 4096)
        x = x.transpose(1, 2)         # (B, 4096, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """Standard transformer encoder block: self-attention + MLP + LayerNorm.

    Used in the image encoder's ViT backbone.
    """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        # Self-attention with pre-norm
        h = self.norm1(x)
        h = self.attn(h, h, h)[0]
        x = x + h

        # MLP with pre-norm
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class ImageEncoder(nn.Module):
    """MAE-pretrained ViT image encoder for SAM.

    Dimension flow (from notes):
        (3, 1024, 1024) -> patch 16x16 -> (4096, 768)
        -> project -> (4096, embed_dim) -> ViT blocks -> (4096, embed_dim)
        -> reshape -> (embed_dim, 64, 64)
        -> 1x1 conv project -> (256, 64, 64)

    The output 256 is the decoder's expected embedding dimension.
    Spatial info (64×64) is preserved — unlike classification ViTs that
    use only a CLS token, SAM uses ALL patch tokens.

    Args:
        image_size: Input image resolution (1024)
        patch_size: Patch size (16 → 64×64 grid)
        in_channels: Image channels (3 for RGB)
        embed_dim: ViT internal embedding dimension (768 for base, 1280 for huge)
        depth: Number of transformer blocks
        num_heads: Attention heads per block
        out_dim: Output channel dimension for the decoder (256)
    """

    def __init__(
        self,
        image_size=1024,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_dim=256,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = (image_size // patch_size) ** 2  # 4096

        # Learnable position embeddings for all patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Project from ViT embed_dim to decoder's expected 256 channels
        # Using 1x1 convolution as described in the notes
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, bias=False),
            nn.LayerNorm([out_dim, image_size // patch_size, image_size // patch_size]),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([out_dim, image_size // patch_size, image_size // patch_size]),
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: (B, 3, 1024, 1024) input image

        Returns:
            (B, 256, 64, 64) image embedding
        """
        # Patch embedding: (B, 3, 1024, 1024) -> (B, 4096, embed_dim)
        x = self.patch_embed(x)

        # Add position embeddings
        x = x + self.pos_embed

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape to 2D spatial grid: (B, 4096, embed_dim) -> (B, embed_dim, 64, 64)
        grid_size = self.image_size // self.patch_size  # 64
        x = x.transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, grid_size, grid_size
        )

        # Project to decoder dimension: (B, embed_dim, 64, 64) -> (B, 256, 64, 64)
        x = self.neck(x)

        return x


# ==============================================================================
# Prompt Encoder (No Transformer — just embeddings and convolutions)
# ==============================================================================
class PromptEncoder(nn.Module):
    """Encodes sparse prompts (points, boxes) and dense prompts (masks).

    No transformer inside — purely projections and convolutions.

    Sparse Branch:
        - Point: (x, y) normalized coords + positional encoding + type embedding -> 256-dim
        - Box: two corners, each treated like a point with distinct type embeddings
        - Output: (num_sparse_tokens, 256)

    Dense Branch:
        - Mask: (1, 1024, 1024) -> Conv2D downsampling -> (256, 64, 64)
        - Added element-wise to image embedding

    Output Tokens (mask queries, analogous to DETR's object queries):
        - 3 learnable tokens for 3 ambiguity-level masks
        - 1 learnable token for IoU score prediction
        - These are prepended to the sparse token sequence

    Args:
        embed_dim: Embedding dimension (256)
        image_size: Original image size (1024)
        mask_in_channels: Input mask channels (1)
        num_mask_tokens: Number of output mask predictions (3)
    """

    def __init__(
        self,
        embed_dim=256,
        image_size=1024,
        mask_in_channels=1,
        num_mask_tokens=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.input_image_size = (image_size, image_size)

        # ── Sparse Prompt: Positional Encoding ──────────────────────────
        # Project (x, y) coordinates to embed_dim via a learned linear layer
        self.point_proj = nn.Linear(2, embed_dim)

        # ── Type Embeddings (3 types: foreground point, background point, box corner) ──
        # point_fg = foreground point, point_bg = background point
        # box_corner1 = top-left corner, box_corner2 = bottom-right corner
        self.point_fg_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.point_bg_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.box_corner1_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.box_corner2_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # ── Dense Prompt: Mask Downsampling ─────────────────────────────
        # (1, 1024, 1024) -> Conv layers -> (256, 64, 64)
        # Matches image encoder output spatial dimensions
        self.mask_downscale = nn.Sequential(
            nn.Conv2d(mask_in_channels, embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim // 4, image_size // 2, image_size // 2]),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim, image_size // 4, image_size // 4]),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=image_size // 4 // 16, stride=image_size // 4 // 16),
            # Final: (256, 64, 64)
        )

        # Embedding for when no mask prompt is given (zeros + learned "no mask" token)
        self.no_mask_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)

        # ── Output Tokens (Mask Queries) ────────────────────────────────
        # 3 mask tokens (one per ambiguity level) + 1 IoU prediction token
        # Analogous to object queries in DETR
        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Parameter(
            torch.randn(num_mask_tokens, embed_dim) * 0.02
        )
        self.iou_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def _encode_points(self, points, labels):
        """Encode point prompts.

        Args:
            points: (B, N, 2) normalized (x, y) coordinates in [0, 1]
            labels: (B, N) where 1 = foreground, 0 = background

        Returns:
            (B, N, 256) point embeddings
        """
        # Project coordinates to embed_dim
        point_embeds = self.point_proj(points)  # (B, N, 256)

        # Add type embeddings based on foreground/background label
        fg_mask = (labels == 1).unsqueeze(-1).float()  # (B, N, 1)
        bg_mask = (labels == 0).unsqueeze(-1).float()

        point_embeds = point_embeds + fg_mask * self.point_fg_embed + bg_mask * self.point_bg_embed

        return point_embeds

    def _encode_boxes(self, boxes):
        """Encode box prompts.

        Each box is 2 corner points with distinct type embeddings.

        Args:
            boxes: (B, M, 4) where each box is (x1, y1, x2, y2) normalized

        Returns:
            (B, 2*M, 256) box corner embeddings
        """
        B, M, _ = boxes.shape

        # Split into two corners
        corner1 = boxes[:, :, :2]  # (B, M, 2) — top-left
        corner2 = boxes[:, :, 2:]  # (B, M, 2) — bottom-right

        # Project and add type embeddings
        c1_embed = self.point_proj(corner1) + self.box_corner1_embed  # (B, M, 256)
        c2_embed = self.point_proj(corner2) + self.box_corner2_embed  # (B, M, 256)

        # Interleave: [corner1_box1, corner2_box1, corner1_box2, corner2_box2, ...]
        box_embeds = torch.stack([c1_embed, c2_embed], dim=2)  # (B, M, 2, 256)
        box_embeds = box_embeds.reshape(B, M * 2, self.embed_dim)  # (B, 2M, 256)

        return box_embeds

    def _encode_mask(self, mask):
        """Encode dense mask prompt.

        Args:
            mask: (B, 1, 1024, 1024) binary mask or None

        Returns:
            (B, 256, 64, 64) dense embedding to add to image embedding
        """
        if mask is not None:
            return self.mask_downscale(mask)
        else:
            # No mask provided — return learned "no mask" embedding
            # Broadcast to (B, 256, 64, 64)
            return self.no_mask_embed.expand(-1, -1, 64, 64)

    def forward(self, points=None, labels=None, boxes=None, mask=None, batch_size=1):
        """Encode all prompts into sparse tokens and dense embeddings.

        Args:
            points: (B, N, 2) optional point coordinates
            labels: (B, N) optional point labels (1=fg, 0=bg)
            boxes: (B, M, 4) optional box coordinates
            mask: (B, 1, H, W) optional mask prompt
            batch_size: batch size (used when no prompts given)

        Returns:
            sparse_embeddings: (B, num_tokens, 256) — output tokens + point/box tokens
            dense_embeddings: (B, 256, 64, 64) — mask embedding to add to image
        """
        B = batch_size
        device = self.mask_tokens.device

        sparse_parts = []

        # Always prepend the output tokens (3 mask + 1 IoU)
        output_tokens = torch.cat([
            self.iou_token.unsqueeze(0),   # (1, 1, 256)
            self.mask_tokens.unsqueeze(0),  # (1, 3, 256)
        ], dim=1).expand(B, -1, -1)         # (B, 4, 256)
        sparse_parts.append(output_tokens)

        # Encode point prompts if provided
        if points is not None and labels is not None:
            point_embeds = self._encode_points(points, labels)
            sparse_parts.append(point_embeds)

        # Encode box prompts if provided
        if boxes is not None:
            box_embeds = self._encode_boxes(boxes)
            sparse_parts.append(box_embeds)

        # Concatenate all sparse tokens: (B, 4 + N + 2M, 256)
        sparse_embeddings = torch.cat(sparse_parts, dim=1)

        # Encode dense mask prompt
        dense_embeddings = self._encode_mask(mask)
        if dense_embeddings.shape[0] == 1 and B > 1:
            dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)

        return sparse_embeddings, dense_embeddings


# ==============================================================================
# Mask Decoder (Lightweight Transformer with Bidirectional Cross-Attention)
# ==============================================================================
class TwoWayAttentionBlock(nn.Module):
    """Bidirectional cross-attention block used in SAM's mask decoder.

    From the notes (Part 3 architecture):
        Token Sequence -> Self-Attention -> Token-to-Image Cross-Attention -> MLP
        Image Sequence -> Image-to-Token Cross-Attention (using updated tokens)

    This is the key difference from standard decoders — attention flows BOTH ways:
    tokens attend to image features AND image features attend to tokens.

    Args:
        embed_dim: Embedding dimension (256)
        num_heads: Number of attention heads
    """

    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()

        # Step 1: Self-attention among tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Step 2: Token-to-Image cross-attention
        # Q = tokens, K/V = image features
        self.cross_attn_token_to_image = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Step 3: MLP on tokens
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Step 4: Image-to-Token cross-attention
        # Q = image features, K/V = tokens (bidirectional!)
        self.cross_attn_image_to_token = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, tokens, image_embeddings):
        """
        Args:
            tokens: (B, num_tokens, 256) — sparse prompt tokens + output tokens
            image_embeddings: (B, 4096, 256) — flattened image features

        Returns:
            tokens: (B, num_tokens, 256) — updated token sequence
            image_embeddings: (B, 4096, 256) — updated image features
        """
        # Step 1: Self-attention among tokens
        h = self.norm1(tokens)
        h = self.self_attn(h, h, h)[0]
        tokens = tokens + h

        # Step 2: Token-to-Image cross-attention (tokens query image)
        h = self.norm2(tokens)
        h = self.cross_attn_token_to_image(h, image_embeddings, image_embeddings)[0]
        tokens = tokens + h

        # Step 3: MLP
        h = self.norm3(tokens)
        h = self.mlp(h)
        tokens = tokens + h

        # Step 4: Image-to-Token cross-attention (image queries tokens)
        h = self.norm4(image_embeddings)
        h = self.cross_attn_image_to_token(h, tokens, tokens)[0]
        image_embeddings = image_embeddings + h

        return tokens, image_embeddings


class MaskDecoder(nn.Module):
    """SAM's mask decoder: produces 3 masks + IoU scores from image + prompt embeddings.

    Architecture (from notes):
        1. Add dense (mask) embedding to image embedding element-wise
        2. Flatten image embedding to sequence: (256, 64, 64) -> (4096, 256)
        3. Run 2× TwoWayAttentionBlock (bidirectional cross-attention)
        4. One more token-to-image cross-attention
        5. Upscale image features: (256, 64, 64) -> (256, 256, 256) via transposed conv
        6. Each of 3 mask tokens -> MLP -> 256-dim vector
        7. Dot product with upscaled image -> (1, 256, 256) per mask
        8. Bilinear interpolation -> (1, 1024, 1024) per mask
        9. IoU token -> MLP -> 3 scalar confidence scores

    Args:
        embed_dim: Embedding dimension (256)
        num_heads: Attention heads in decoder blocks
        num_mask_tokens: Number of mask predictions (3)
        image_size: Original image size for final upsampling (1024)
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_mask_tokens=3,
        image_size=1024,
    ):
        super().__init__()
        self.num_mask_tokens = num_mask_tokens
        self.image_size = image_size

        # 2× bidirectional cross-attention blocks
        self.decoder_blocks = nn.ModuleList([
            TwoWayAttentionBlock(embed_dim, num_heads)
            for _ in range(2)
        ])

        # Final token-to-image cross-attention (applied once more after the 2 blocks)
        self.final_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # ── Image Upscaling ────────────────────────────────────────────
        # (256, 64, 64) -> 2× transposed conv -> (256, 256, 256)
        # as described in notes: "2× transposed convolution upscaling"
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim, 128, 128]),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm([embed_dim, 256, 256]),
            nn.GELU(),
        )

        # ── Per-Mask MLPs ──────────────────────────────────────────────
        # Each mask token -> MLP -> 256-dim vector (for dot product with upscaled image)
        self.mask_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_mask_tokens)
        ])

        # ── IoU Prediction Head ────────────────────────────────────────
        # IoU token -> MLP -> num_mask_tokens scores
        self.iou_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_mask_tokens),
        )

    def forward(self, image_embeddings, sparse_embeddings, dense_embeddings):
        """
        Args:
            image_embeddings: (B, 256, 64, 64) from image encoder
            sparse_embeddings: (B, num_tokens, 256) from prompt encoder
            dense_embeddings: (B, 256, 64, 64) mask/no-mask embedding

        Returns:
            masks: (B, 3, 1024, 1024) — three predicted masks
            iou_scores: (B, 3) — IoU confidence per mask
        """
        B = image_embeddings.shape[0]

        # Add dense embedding to image embedding (element-wise)
        image_with_mask = image_embeddings + dense_embeddings  # (B, 256, 64, 64)

        # Flatten image to sequence for attention: (B, 256, 64, 64) -> (B, 4096, 256)
        image_seq = image_with_mask.flatten(2).transpose(1, 2)  # (B, 4096, 256)

        tokens = sparse_embeddings  # (B, num_tokens, 256)

        # ── 2× Bidirectional Cross-Attention Blocks ────────────────────
        for block in self.decoder_blocks:
            tokens, image_seq = block(tokens, image_seq)

        # ── Final Token-to-Image Cross-Attention ───────────────────────
        h = self.final_norm(tokens)
        h = self.final_attn(h, image_seq, image_seq)[0]
        tokens = tokens + h

        # ── Extract Output Tokens ──────────────────────────────────────
        # Token layout: [IoU_token, mask_token_0, mask_token_1, mask_token_2, ...]
        iou_token_out = tokens[:, 0, :]                                    # (B, 256)
        mask_tokens_out = tokens[:, 1:1 + self.num_mask_tokens, :]         # (B, 3, 256)

        # ── Upscale Image Features ─────────────────────────────────────
        # Reshape image sequence back to spatial: (B, 4096, 256) -> (B, 256, 64, 64)
        image_spatial = image_seq.transpose(1, 2).reshape(B, -1, 64, 64)
        # Upscale: (B, 256, 64, 64) -> (B, 256, 256, 256)
        upscaled = self.upscale(image_spatial)

        # ── Generate Masks via Dot Product ─────────────────────────────
        # Each mask token (256-dim) dot-producted with upscaled image (256 channels)
        # collapses channels -> (1, 256, 256) per mask
        masks = []
        for i in range(self.num_mask_tokens):
            # Mask token -> MLP -> (B, 256)
            mask_vec = self.mask_mlps[i](mask_tokens_out[:, i, :])
            # Dot product: (B, 256, 1, 1) * (B, 256, 256, 256) -> sum over channels
            mask_vec = mask_vec.unsqueeze(-1).unsqueeze(-1)   # (B, 256, 1, 1)
            mask = (upscaled * mask_vec).sum(dim=1, keepdim=True)  # (B, 1, 256, 256)
            # Bilinear interpolation to original resolution
            mask = F.interpolate(
                mask, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )  # (B, 1, 1024, 1024)
            masks.append(mask)

        # Stack masks: (B, 3, 1024, 1024)
        masks = torch.cat(masks, dim=1)

        # ── IoU Score Prediction ───────────────────────────────────────
        iou_scores = self.iou_mlp(iou_token_out)  # (B, 3)

        return masks, iou_scores


# ==============================================================================
# Full SAM Model
# ==============================================================================
class SAM(nn.Module):
    """Segment Anything Model — combines all three components.

    The key design insight: image encoder runs ONCE per image (expensive),
    then prompt encoder + mask decoder can run many times per click (cheap).

    Args:
        image_size: Input image resolution (1024)
        patch_size: ViT patch size (16)
        encoder_embed_dim: ViT embedding dimension (768 base, 1280 huge)
        encoder_depth: Number of ViT transformer blocks
        encoder_heads: ViT attention heads
        decoder_embed_dim: Decoder embedding dimension (256)
        decoder_heads: Decoder attention heads
        num_mask_tokens: Number of output masks (3 for ambiguity)
    """

    def __init__(
        self,
        image_size=1024,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_embed_dim=256,
        decoder_heads=8,
        num_mask_tokens=3,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            out_dim=decoder_embed_dim,
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim=decoder_embed_dim,
            image_size=image_size,
            num_mask_tokens=num_mask_tokens,
        )

        self.mask_decoder = MaskDecoder(
            embed_dim=decoder_embed_dim,
            num_heads=decoder_heads,
            num_mask_tokens=num_mask_tokens,
            image_size=image_size,
        )

    @torch.no_grad()
    def encode_image(self, image):
        """Run image encoder once (cache result for multiple prompts).

        Args:
            image: (B, 3, 1024, 1024) preprocessed image tensor

        Returns:
            (B, 256, 64, 64) image embedding
        """
        return self.image_encoder(image)

    def predict_masks(
        self,
        image_embeddings,
        points=None,
        labels=None,
        boxes=None,
        mask_input=None,
    ):
        """Run prompt encoder + mask decoder (lightweight, runs per-click).

        Args:
            image_embeddings: (B, 256, 64, 64) — cached from encode_image
            points: (B, N, 2) optional point prompts (normalized coords)
            labels: (B, N) point labels (1=foreground, 0=background)
            boxes: (B, M, 4) optional box prompts (normalized coords)
            mask_input: (B, 1, 1024, 1024) optional mask prompt

        Returns:
            masks: (B, 3, 1024, 1024) — 3 predicted masks at different granularity
            iou_scores: (B, 3) — IoU confidence score per mask
        """
        B = image_embeddings.shape[0]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            labels=labels,
            boxes=boxes,
            mask=mask_input,
            batch_size=B,
        )

        masks, iou_scores = self.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
        )

        return masks, iou_scores

    def forward(
        self,
        images,
        points=None,
        labels=None,
        boxes=None,
        mask_input=None,
    ):
        """Full forward pass: image + prompt -> masks + IoU scores.

        Args:
            images: (B, 3, 1024, 1024) input images
            points: (B, N, 2) optional point prompts
            labels: (B, N) optional point labels
            boxes: (B, M, 4) optional box prompts
            mask_input: (B, 1, 1024, 1024) optional mask prompt

        Returns:
            masks: (B, 3, 1024, 1024)
            iou_scores: (B, 3)
        """
        image_embeddings = self.image_encoder(images)
        return self.predict_masks(
            image_embeddings, points, labels, boxes, mask_input
        )


# ==============================================================================
# Loss Functions (from notes: focal + dice + IoU MSE)
# ==============================================================================
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for mask prediction — down-weights easy pixels.

    From notes: L_focal = -alpha * (1 - p_t)^gamma * log(p_t)
    In a 1024x1024 mask, >95% pixels may be background. Without focal loss,
    the model could predict "all background" and still get low loss.

    Args:
        pred: (B, 1, H, W) predicted mask logits
        target: (B, 1, H, W) ground truth binary mask

    Returns:
        Scalar focal loss
    """
    pred_sigmoid = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
    focal_weight = alpha * (1 - p_t) ** gamma

    return (focal_weight * bce).mean()


def dice_loss(pred, target, smooth=1.0):
    """Dice loss — directly optimizes overlap between predicted and GT mask.

    From notes: L_dice = 1 - 2|P ∩ G| / (|P| + |G|)
    Complements focal loss which operates per-pixel independently.

    Args:
        pred: (B, 1, H, W) predicted mask logits
        target: (B, 1, H, W) ground truth binary mask

    Returns:
        Scalar dice loss
    """
    pred_sigmoid = torch.sigmoid(pred).flatten(1)
    target_flat = target.flatten(1)

    intersection = (pred_sigmoid * target_flat).sum(dim=1)
    union = pred_sigmoid.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def sam_loss(pred_masks, pred_iou, gt_mask, focal_weight=20.0, dice_weight=1.0, iou_weight=1.0):
    """Combined SAM loss: select best mask, then focal + dice + IoU MSE.

    From notes:
        1. Out of 3 predicted masks, pick the one with lowest loss
        2. Mask loss = focal_weight * L_focal + dice_weight * L_dice
        3. IoU loss = MSE(predicted_IoU, actual_IoU)
        4. Total = L_mask + iou_weight * L_iou

    Args:
        pred_masks: (B, 3, H, W) three predicted mask logits
        pred_iou: (B, 3) predicted IoU scores
        gt_mask: (B, 1, H, W) ground truth binary mask
        focal_weight: Weight for focal loss (default 20.0 per paper)
        dice_weight: Weight for dice loss (default 1.0)
        iou_weight: Weight for IoU MSE loss (default 1.0)

    Returns:
        total_loss: Scalar combined loss
    """
    B = pred_masks.shape[0]
    num_masks = pred_masks.shape[1]

    # ── Step 1: Find best mask per sample ──────────────────────────────
    # Compute loss for each of the 3 masks, select the one with lowest loss
    per_mask_losses = []
    for i in range(num_masks):
        pred_i = pred_masks[:, i:i+1, :, :]  # (B, 1, H, W)
        fl = focal_loss(pred_i, gt_mask)
        dl = dice_loss(pred_i, gt_mask)
        per_mask_losses.append(focal_weight * fl + dice_weight * dl)

    per_mask_losses = torch.stack(per_mask_losses, dim=0)  # (3,)
    best_mask_idx = per_mask_losses.argmin()

    # ── Step 2: Compute mask loss on best mask ─────────────────────────
    mask_loss = per_mask_losses[best_mask_idx]

    # ── Step 3: IoU prediction loss ────────────────────────────────────
    # Compute actual IoU between best predicted mask and ground truth
    best_pred = torch.sigmoid(pred_masks[:, best_mask_idx:best_mask_idx+1, :, :])
    best_pred_binary = (best_pred > 0.5).float()
    gt_float = gt_mask.float()

    intersection = (best_pred_binary * gt_float).sum(dim=(1, 2, 3))
    union = best_pred_binary.sum(dim=(1, 2, 3)) + gt_float.sum(dim=(1, 2, 3)) - intersection
    actual_iou = intersection / (union + 1e-6)  # (B,)

    predicted_iou = pred_iou[:, best_mask_idx]   # (B,)
    iou_loss = F.mse_loss(predicted_iou, actual_iou)

    # ── Total Loss ─────────────────────────────────────────────────────
    total_loss = mask_loss + iou_weight * iou_loss

    return total_loss


# ==============================================================================
# Visualization Utilities
# ==============================================================================
def show_mask(mask, ax, color=None, alpha=0.5):
    """Overlay a binary mask on a matplotlib axis.

    Args:
        mask: (H, W) binary mask
        ax: matplotlib axis
        color: RGB tuple (default: blue)
        alpha: transparency
    """
    if color is None:
        color = np.array([30 / 255, 144 / 255, 255 / 255])

    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=alpha * mask.astype(float))


def show_points(coords, labels, ax, marker_size=50):
    """Plot point prompts on a matplotlib axis.

    Args:
        coords: (N, 2) point coordinates
        labels: (N,) labels (1=foreground green star, 0=background red star)
        ax: matplotlib axis
    """
    fg = coords[labels == 1]
    bg = coords[labels == 0]
    ax.scatter(fg[:, 0], fg[:, 1], color='green', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25, zorder=5)
    ax.scatter(bg[:, 0], bg[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25, zorder=5)


def visualize_predictions(image, masks, iou_scores, points=None, labels=None):
    """Display image with all 3 predicted masks and their IoU scores.

    Args:
        image: PIL Image or numpy array
        masks: (3, H, W) three binary masks
        iou_scores: (3,) IoU confidence scores
        points: optional (N, 2) point prompts to plot
        labels: optional (N,) point labels
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    if points is not None and labels is not None:
        show_points(points, labels, axes[0])
    axes[0].set_title("Input + Prompt")
    axes[0].axis('off')

    # Three masks at different granularity levels
    mask_names = ["Mask 0 (Global)", "Mask 1 (Sub-global)", "Mask 2 (Local)"]
    colors = [
        np.array([1.0, 0.2, 0.2]),   # Red
        np.array([0.2, 1.0, 0.2]),   # Green
        np.array([0.2, 0.2, 1.0]),   # Blue
    ]

    for i in range(3):
        axes[i + 1].imshow(image)
        if masks[i].sum() > 0:
            show_mask(masks[i], axes[i + 1], color=colors[i])
        if points is not None and labels is not None:
            show_points(points, labels, axes[i + 1])
        axes[i + 1].set_title(f"{mask_names[i]}\nIoU: {iou_scores[i]:.3f}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Main: Build model and run a quick shape verification
# ==============================================================================
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Building SAM model (ViT-Base config)...")
    model = SAM(
        image_size=1024,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_embed_dim=256,
        decoder_heads=8,
        num_mask_tokens=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"  Image encoder:  {sum(p.numel() for p in model.image_encoder.parameters()):,}")
    print(f"  Prompt encoder: {sum(p.numel() for p in model.prompt_encoder.parameters()):,}")
    print(f"  Mask decoder:   {sum(p.numel() for p in model.mask_decoder.parameters()):,}")
    print()

    # ── Shape Verification ─────────────────────────────────────────────
    print("Running shape verification...")
    B = 1

    # Dummy image
    dummy_image = torch.randn(B, 3, 1024, 1024).to(device)

    # Dummy point prompt: 1 foreground point
    dummy_points = torch.tensor([[[0.5, 0.5]]]).to(device)   # (1, 1, 2)
    dummy_labels = torch.tensor([[1]]).to(device)             # (1, 1)

    # Forward pass
    with torch.no_grad():
        # Step 1: Image encoder (run once, cache)
        image_emb = model.encode_image(dummy_image)
        print(f"Image embedding shape: {image_emb.shape}")  # (1, 256, 64, 64)

        # Step 2: Prompt encoder + Mask decoder (run per click)
        masks, iou_scores = model.predict_masks(
            image_emb, points=dummy_points, labels=dummy_labels
        )
        print(f"Masks shape:      {masks.shape}")       # (1, 3, 1024, 1024)
        print(f"IoU scores shape: {iou_scores.shape}")   # (1, 3)

    print("\nAll shapes verified! Model is ready.")
    print("\nNote: This is a from-scratch implementation for educational purposes.")
    print("For production use, load Meta's pretrained weights via:")
    print("  from segment_anything import sam_model_registry, SamPredictor")
