"""
Swin Transformer: Minimal PyTorch Implementation
=================================================
A clean, educational implementation of the Swin Transformer architecture.

Key Components:
- Window Partition / Reverse: Split feature maps into local windows
- Window Attention: Multi-head self-attention within windows
- Shifted Window Attention: Cyclic shift + masking for cross-window connections
- Patch Merging: Hierarchical downsampling (2x spatial reduction, 2x channel increase)
- Swin Transformer Block: W-MSA/SW-MSA + MLP with residual connections

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.
    
    Args:
        x: (B, H, W, C) feature map
        window_size: Size of each window (M)
    
    Returns:
        windows: (num_windows * B, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition back to feature map.
    
    Args:
        windows: (num_windows * B, M, M, C)
        window_size: Size of each window (M)
        H, W: Original feature map dimensions
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self Attention with Relative Position Bias.
    
    Computes attention only within local windows (M x M patches).
    Uses learnable relative position bias instead of absolute position embeddings.
    """
    
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias table: (2M-1) x (2M-1) possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index for each token pair in window
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))  # (2, M, M)
        coords_flat = coords.flatten(1)  # (2, M*M)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, M*M, M*M)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (M*M, M*M, 2)
        relative_coords[:, :, 0] += window_size - 1  # Shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1  # Flatten 2D index to 1D
        relative_position_index = relative_coords.sum(-1)  # (M*M, M*M)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows * B, M*M, C) - flattened window tokens
            mask: (num_windows, M*M, M*M) or None - attention mask for SW-MSA
        
        Returns:
            (num_windows * B, M*M, C)
        """
        B_, N, C = x.shape
        
        # QKV projection and reshape for multi-head attention
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (M*M, M*M, num_heads)
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        
        # Apply mask for shifted windows (blocks attention across window boundaries)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # Broadcast mask
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Standard MLP block with GELU activation."""
    
    def __init__(self, dim: int, hidden_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block: W-MSA or SW-MSA + MLP.
    
    Alternates between regular window attention (shift_size=0) and 
    shifted window attention (shift_size=window_size//2).
    """
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: tuple, 
        num_heads: int, 
        window_size: int = 7, 
        shift_size: int = 0,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Ensure shift doesn't exceed half window size
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        
        # Create attention mask for shifted windows
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._create_mask(input_resolution))
        else:
            self.register_buffer("attn_mask", None)
    
    def _create_mask(self, input_resolution: tuple) -> torch.Tensor:
        """Create attention mask for SW-MSA using cyclic shift regions."""
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        
        # Assign different region IDs based on shift boundaries
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        # Convert mask to window format
        mask_windows = window_partition(img_mask, self.window_size)  # (num_windows, M, M, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        
        # Create pairwise mask: 0 if same region, -100 if different
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C) - flattened feature map
        
        Returns:
            (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: {L} vs {H * W}"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows * B, M, M, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # Residual connections
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer for hierarchical downsampling.
    
    Concatenates 2x2 neighboring patches and projects channels: 4C -> 2C.
    Reduces spatial resolution by 2x.
    """
    
    def __init__(self, input_resolution: tuple, dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
        
        Returns:
            (B, H/2 * W/2, 2C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        
        x = x.view(B, H, W, C)
        
        # Sample 2x2 patches and concatenate
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2 * W/2, 2C)
        
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding: Split image into patches and project to embedding dimension."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image
        
        Returns:
            (B, H/P * W/P, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer: Hierarchical Vision Transformer.
    
    Architecture:
    - Stage 1: Patch Embed (H/4 x W/4) + Swin Blocks
    - Stage 2: Patch Merge (H/8 x W/8) + Swin Blocks  
    - Stage 3: Patch Merge (H/16 x W/16) + Swin Blocks
    - Stage 4: Patch Merge (H/32 x W/32) + Swin Blocks
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        
        # Build stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * (2 ** i_layer))
            resolution = (
                patches_resolution[0] // (2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer)
            )
            
            # Stack of Swin Transformer Blocks
            blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=resolution,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio
                )
                for j in range(depths[i_layer])
            ])
            
            # Patch merging (downsample) except for last stage
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(resolution, dim)
            else:
                downsample = None
            
            self.layers.append(nn.ModuleDict({
                'blocks': blocks,
                'downsample': downsample
            }))
        
        # Final norm and classifier head
        self.norm = nn.LayerNorm(int(embed_dim * (2 ** (self.num_layers - 1))))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * (2 ** (self.num_layers - 1))), num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        x = self.patch_embed(x)
        
        for layer in self.layers:
            for block in layer['blocks']:
                x = block(x)
            if layer['downsample'] is not None:
                x = layer['downsample'](x)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with classification head."""
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ========================
# Demo / Test
# ========================
if __name__ == "__main__":
    # Create Swin-Tiny configuration
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    print("=" * 50)
    print("Swin Transformer - Minimal Implementation")
    print("=" * 50)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Show hierarchical feature maps at each stage
    print("Hierarchical Feature Maps:")
    print("-" * 40)
    x_feat = model.patch_embed(x)
    print(f"After Patch Embed: {tuple(x_feat.shape)} -> (B, H/4 * W/4, C)")
    
    for i, layer in enumerate(model.layers):
        for block in layer['blocks']:
            x_feat = block(x_feat)
        print(f"After Stage {i+1}:     {tuple(x_feat.shape)}")
        if layer['downsample'] is not None:
            x_feat = layer['downsample'](x_feat)
            print(f"After Merge {i+1}:     {tuple(x_feat.shape)}")
