import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# 🧠 Custom MHA block using scaled_dot_product_attention
class MyAttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        H = self.n_heads
        D = self.head_dim

        qkv = self.qkv(x)  # [B, N, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, N, C] -> [B, H, N, D]
        q = q.view(B, N, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        #     scaled_dot_product_attention(...)

        # 🧠 FlashAttention + causal + compile-friendly
        with sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False, backends=SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # [B, H, N, D] -> [B, N, C]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out(out)

# ⚡ Simple MLP + attention model
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MyAttentionBlock()
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x

# 🎯 Training loop on fake data
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer().to(device)
    model = torch.compile(model)  # 🚀 compile for speed

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for step in range(10):
        # 🔁 Fake data: batch size 8, sequence 64, model dim 128
        x = torch.randn(8, 64, 128, device=device)
        y = torch.randn(8, 64, 128, device=device)

        # 🧙 bfloat16 autocasting (FlashAttention requires float16 or bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}: loss = {loss.item():.4f}")

if __name__ == "__main__":
    train()
