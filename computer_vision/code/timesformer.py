"""
TimeSformer (Time-Space Transformer): From-Scratch Implementation
=================================================================
An educational implementation of TimeSformer for video classification
using divided space-time attention, with pretrained ViT-Base weights.

Key Components:
- Patch Embedding: nn.Conv2d to split frames into 16x16 patches
- Divided Space-Time Attention: Separate temporal and spatial attention
  in each transformer block (T+S approach from the paper)
- Pretrained ViT-Base/16: Loaded via timm for transfer learning
- CLS Token: Single learnable token for video-level classification

Architecture per Transformer Block:
    Input -> Temporal Attention -> LayerNorm + Residual
          -> Spatial Attention  -> LayerNorm + Residual
          -> MLP               -> LayerNorm + Residual -> Output

Dataset: Kinetics-400 subset (bench press vs. deadlift)
Reference: Bertasius et al., "Is Space-Time Attention All You Need
           for Video Understanding?" (2021)
"""

import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange # reshaping and permuting the tensors
from tqdm import tqdm # bruh, you know it, its just a progress bar
import timm # Torch IMage Models, it has bunch of pretrained models in there
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Video Frame Dataset
# ==============================================================================
class VideoFrameDataset(Dataset):
    """Dataset that loads pre-extracted video frames from a folder structure.

    Expected directory structure:
        root_dir/
            class_1/
                video_1/
                    frame_001.jpg
                    frame_002.jpg
                    ...
                video_2/
                    ...
            class_2/
                ...

    Each video folder contains pre-extracted frames as images.
    A fixed number of frames are sampled uniformly from each video.

    Args:
        root_dir: Path to the root directory containing class folders
        num_frames: Number of frames to sample per video (default: 8)
        transform: Optional torchvision transform to apply to each frame
    """

    def __init__(self, root_dir, num_frames=8, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []  # List of (video_folder_path, class_index)

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for video_folder in sorted(os.listdir(cls_dir)):
                video_path = os.path.join(cls_dir, video_folder)
                if os.path.isdir(video_path):
                    self.samples.append((video_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # Get sorted list of frame files
        frame_files = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Sample num_frames frames uniformly from the video
        total_frames = len(frame_files)
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        else:
            # If fewer frames than needed, repeat last frame
            indices = list(range(total_frames))
            indices += [total_frames - 1] * (self.num_frames - total_frames)
            indices = torch.tensor(indices)

        frames = []
        for i in indices:
            frame_path = os.path.join(video_path, frame_files[i])
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        # Stack frames: (num_frames, C, H, W)
        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, label


# ==============================================================================
# TimeSformer Transformer Block (Divided Space-Time Attention)
# ==============================================================================
class TimeSformerBlock(nn.Module):
    """Single transformer block with divided space-time attention.

    Instead of joint attention over all patches in all frames (O((NF)^2)),
    this block applies:
        1. Temporal attention — each patch attends across frames (O(NF * F))
        2. Spatial attention  — each patch attends within its frame (O(NF * N))

    This reduces per-token complexity from O(NF) to O(N + F).

    Args:
        dim: Embedding dimension (768 for ViT-Base)
        heads: Number of attention heads (12 for ViT-Base)
    """

    def __init__(self, dim, heads):
        super().__init__()

        # ── Temporal & Spatial Multi-Head Attention ──────────────────────
        # Separate attention modules → separate learnable W_Q, W_K, W_V
        # This is a key advantage over joint attention (shared parameters)
        self.temporal_attention = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )
        self.spatial_attention = nn.MultiheadAttention(
            dim, heads, batch_first=True
        )

        # ── Layer Norms (one per sub-layer) ─────────────────────────────
        self.norm1 = nn.LayerNorm(dim)  # After temporal attention
        self.norm2 = nn.LayerNorm(dim)  # After spatial attention
        self.norm3 = nn.LayerNorm(dim)  # After MLP

        # ── MLP: project to 4x dim then back ───────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        """Forward pass through one divided space-time transformer block.

        Args:
            x: (B, T, N, D) — batch, frames, patches, embedding dim

        Returns:
            x: (B, T, N, D) — same shape, with attended context
        """
        B, T, N, D = x.shape

        # ── Step 1: Temporal Attention ──────────────────────────────────
        # For each spatial patch, attend across all frames.
        # Rearrange: merge (B, N) into one dim → (B*N, T, D)
        # Now each "sequence" is T frames of the same spatial patch.
        xt = rearrange(x, 'b t n d -> (b n) t d')
        xt = self.temporal_attention(xt, xt, xt)[0]  # [0] = context vectors
        xt = rearrange(xt, '(b n) t d -> b t n d', b=B, n=N)
        x = x + self.norm1(xt)  # Residual + LayerNorm

        # ── Step 2: Spatial Attention ───────────────────────────────────
        # For each frame, attend across all spatial patches.
        # Rearrange: merge (B, T) into one dim → (B*T, N, D)
        # Now each "sequence" is N patches within one frame.
        xs = rearrange(x, 'b t n d -> (b t) n d')
        xs = self.spatial_attention(xs, xs, xs)[0]
        xs = rearrange(xs, '(b t) n d -> b t n d', b=B, t=T)
        x = x + self.norm2(xs)  # Residual + LayerNorm

        # ── Step 3: MLP ────────────────────────────────────────────────
        x = x + self.norm3(self.mlp(x))  # Residual + LayerNorm

        return x


# ==============================================================================
# TimeSformer Model
# ==============================================================================
class TimeSformer(nn.Module):
    """TimeSformer: Time-Space Transformer for Video Classification.

    Extends Vision Transformer to videos using divided space-time attention.
    Each frame is split into patches, projected to embeddings, and processed
    through multiple transformer blocks with separate temporal and spatial
    attention.

    Args:
        num_classes: Number of action classes to predict
        num_frames: Number of frames sampled per video (default: 8)
        image_size: Spatial resolution of each frame (default: 224)
        patch_size: Size of each image patch (default: 16)
        embed_dim: Embedding dimension (default: 768 for ViT-Base)
        depth: Number of transformer blocks (default: 12)
        heads: Number of attention heads per block (default: 12)
    """

    def __init__(
        self,
        num_classes=2,
        num_frames=8,
        image_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        heads=12,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        # For 224px image with 16px patches: (224/16)^2 = 14^2 = 196 patches

        # ── Patch Embedding via Conv2D ──────────────────────────────────
        # kernel_size = stride = patch_size → non-overlapping patches
        # Input: (B*T, 3, H, W) → Output: (B*T, embed_dim, H/P, W/P)
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # ── CLS Token ──────────────────────────────────────────────────
        # Learnable classification token prepended to the sequence.
        # Shape: (1, 1, 1, embed_dim) — will be broadcast across batch & patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # ── Position Embeddings (separate for space and time) ───────────
        # Temporal PE: (1, F+1, 1, D) — +1 slot for CLS token at index 0
        # Spatial PE: (1, 1, N, D) — no extra slot for CLS (already in temporal)
        self.time_embed = nn.Parameter(
            torch.randn(1, num_frames + 1, 1, embed_dim)
        )
        self.space_embed = nn.Parameter(
            torch.randn(1, 1, self.num_patches, embed_dim)
        )

        # ── Transformer Blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, heads) for _ in range(depth)
        ])

        # ── Classification Head ────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Forward pass: video → class logits.

        Args:
            x: (B, T, C, H, W) — batch of videos
                B = batch size
                T = number of frames (e.g., 8)
                C = channels (3 for RGB)
                H, W = spatial dimensions (e.g., 224)

        Returns:
            logits: (B, num_classes) — classification scores
        """
        B, T, C, H, W = x.shape

        # ── Patch Embedding ─────────────────────────────────────────────
        # Merge batch and time dims for Conv2D: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Apply patch embedding convolution
        x = self.patch_embed(x)
        # Shape: (B*T, embed_dim, H/P, W/P) e.g., (B*T, 768, 14, 14)

        # Flatten spatial dims and transpose to get tokens
        x = x.flatten(2)       # (B*T, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B*T, num_patches, embed_dim)

        # Restore temporal dimension
        x = x.view(B, T, -1, self.embed_dim)
        # Shape: (B, T, num_patches, embed_dim) e.g., (B, 8, 196, 768)

        # ── Add Position Embeddings (excluding CLS) ─────────────────────
        # time_embed[:, 1:T+1] skips the CLS position (index 0)
        # .unsqueeze adds missing dims for broadcasting
        x = x + self.time_embed[:, 1:T+1, :, :] + self.space_embed

        # ── Prepend CLS Token ──────────────────────────────────────────
        cls = self.cls_token.expand(B, 1, self.num_patches, -1)
        # Add CLS position embedding (index 0 from time_embed)
        cls = cls + self.time_embed[:, :1, :, :]
        x = torch.cat([cls, x], dim=1)
        # Shape: (B, T+1, num_patches, embed_dim) — CLS is "frame 0"

        # ── Pass Through Transformer Blocks ────────────────────────────
        for block in self.blocks:
            x = block(x)

        # ── Extract CLS Token & Classify ───────────────────────────────
        # CLS token is at temporal position 0, spatial position 0
        cls_output = self.norm(x[:, 0, 0])  # (B, embed_dim)
        return self.head(cls_output)         # (B, num_classes)


# ==============================================================================
# Load Pretrained ViT Weights into TimeSformer
# ==============================================================================
def load_pretrained_vit_weights(model, vit):
    """Copy weights from a pretrained ViT-Base into TimeSformer.

    Maps ViT's spatial-only attention weights to TimeSformer's
    divided space-time attention blocks. Temporal attention weights
    are initialized from the same spatial weights.

    Args:
        model: TimeSformer model to load weights into
        vit: Pretrained ViT model from timm
    """
    # Copy patch embedding weights
    model.patch_embed.weight.data.copy_(vit.patch_embed.proj.weight.data)
    model.patch_embed.bias.data.copy_(vit.patch_embed.proj.bias.data)

    # Copy transformer block weights
    for i, block in enumerate(model.blocks):
        vit_block = vit.blocks[i]

        # Temporal attention ← initialized from ViT's spatial attention
        block.temporal_attention.in_proj_weight.data.copy_(
            vit_block.attn.qkv.weight.data
        )
        block.temporal_attention.in_proj_bias.data.copy_(
            vit_block.attn.qkv.bias.data
        )
        block.temporal_attention.out_proj.weight.data.copy_(
            vit_block.attn.proj.weight.data
        )
        block.temporal_attention.out_proj.bias.data.copy_(
            vit_block.attn.proj.bias.data
        )

        # Spatial attention ← from ViT's spatial attention
        block.spatial_attention.in_proj_weight.data.copy_(
            vit_block.attn.qkv.weight.data
        )
        block.spatial_attention.in_proj_bias.data.copy_(
            vit_block.attn.qkv.bias.data
        )
        block.spatial_attention.out_proj.weight.data.copy_(
            vit_block.attn.proj.weight.data
        )
        block.spatial_attention.out_proj.bias.data.copy_(
            vit_block.attn.proj.bias.data
        )

        # Layer norms
        block.norm1.weight.data.copy_(vit_block.norm1.weight.data)
        block.norm1.bias.data.copy_(vit_block.norm1.bias.data)
        block.norm2.weight.data.copy_(vit_block.norm1.weight.data)
        block.norm2.bias.data.copy_(vit_block.norm1.bias.data)
        block.norm3.weight.data.copy_(vit_block.norm2.weight.data)
        block.norm3.bias.data.copy_(vit_block.norm2.bias.data)

        # MLP weights
        block.mlp[0].weight.data.copy_(vit_block.mlp.fc1.weight.data)
        block.mlp[0].bias.data.copy_(vit_block.mlp.fc1.bias.data)
        block.mlp[2].weight.data.copy_(vit_block.mlp.fc2.weight.data)
        block.mlp[2].bias.data.copy_(vit_block.mlp.fc2.bias.data)

    # Copy final layer norm
    model.norm.weight.data.copy_(vit.norm.weight.data)
    model.norm.bias.data.copy_(vit.norm.bias.data)

    print("Pretrained ViT weights loaded into TimeSformer successfully.")


# ==============================================================================
# Training Loop
# ==============================================================================
def train(model, train_loader, optimizer, criterion, num_epochs=10):
    """Train the TimeSformer model.

    Args:
        model: TimeSformer model
        train_loader: DataLoader yielding (video_tensor, label) batches
        optimizer: Optimizer (e.g., AdamW)
        criterion: Loss function (e.g., CrossEntropyLoss)
        num_epochs: Number of training epochs
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for videos, labels in pbar:
            videos = videos.to(device)     # (B, T, C, H, W)
            labels = labels.to(device)     # (B,)

            optimizer.zero_grad()
            outputs = model(videos)        # (B, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.1f}%'
            })

        epoch_acc = 100. * correct / total
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.1f}%")


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    # ── Load Pretrained ViT ────────────────────────────────────────────
    print("Loading pretrained ViT-Base/16...")
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit.eval()
    print("ViT loaded.\n")

    # ── Create Dataset ─────────────────────────────────────────────────
    # Update this path to your local dataset directory
    DATA_ROOT = "./exercise_data/output"  # Contains class folders (bench_press, deadlift)

    train_dataset = VideoFrameDataset(
        root_dir=DATA_ROOT,
        num_frames=8,
    )
    print(f"Dataset: {len(train_dataset)} videos, "
          f"{len(train_dataset.classes)} classes: {train_dataset.classes}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )

    # ── Initialize TimeSformer ─────────────────────────────────────────
    model = TimeSformer(
        num_classes=len(train_dataset.classes),
        num_frames=8,
        embed_dim=768,
        depth=12,
        heads=12,
    ).to(device)

    # Load pretrained ViT weights
    load_pretrained_vit_weights(model, vit)

    # ── Training Setup ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ── Train ──────────────────────────────────────────────────────────
    print("\nStarting training...")
    train(model, train_loader, optimizer, criterion, num_epochs=10)
    # Expected: ~80% training accuracy at 10 epochs
    # Note: Very small dataset (~80 videos), no validation split
