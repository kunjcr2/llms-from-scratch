import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Path
from torch.utils.data import DataLoader
from typing import Any, Dict

# --- Tiny Transformer block using built-in MHA ---
class TinyTransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        # Self-attn (pre-norm)
        x_res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask,
                                need_weights=False)
        x = x_res + self.drop1(attn_out)
        # MLP (pre-norm)
        x = x + self.mlp(self.norm2(x))
        return x

# --- Patchify with Unfold + linear projection to D ---
class PatchEmbed(nn.Module):
    """
    Inputs: (B, C, H, W), patch_size=2 -> (B, N, D) where N = (H/2)*(W/2)
    """
    def __init__(self, in_chans=3, embed_dim=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x)                 # (B, C*ps*ps, N)
        patches = patches.transpose(1, 2)        # (B, N, C*ps*ps)
        x = self.proj(patches)                   # (B, N, D)
        return x

# --- Tiny ViT (PoC) ---
class TinyViT(nn.Module):
    def __init__(self, image_size=10, patch_size=2, in_chans=3,
                 embed_dim=64, depth=4, num_heads=8, num_classes=10,
                 mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        num_patches = (image_size // patch_size) ** 2  # 25 for 10x10, ps=2

        # class token + positional embeddings (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # encoder stack
        self.blocks = nn.Sequential(
            *[TinyTransformerEncoderBlock(embed_dim, num_heads, mlp_ratio,
                                          attn_dropout, dropout)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_probs: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x)              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)       # (B, N+1, D) — cls first
        x = x + self.pos_embed               # add positional encodings

        x = self.blocks(x)                   # (B, N+1, D)
        x = self.norm(x)
        cls_out = x[:, 0, :]                 # take [CLS]
        logits = self.head(cls_out)          # (B, C)

        if return_probs:
            return F.softmax(logits, dim=-1) # for inference only
        return logits                         # for training with CrossEntropyLoss

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: Dict[str, Any],
    save_dir: Path,
):
    pass

def main():
    pass

if __name__ == '__main__':
    main()