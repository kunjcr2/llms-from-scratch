import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
T = 1500
image_size = 28
batch_size = 128
epochs = 5 # 20 in full training
lr = 1e-3
base_channels = 64

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Forward Diffusion Parameters
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

class TimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, t):
        half_dim = self.dimension // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / (half_dim - 1))
        args = t[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_projection = nn.Linear(time_dim, out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = torch.relu(h)
        h = self.conv1(h)
        
        time_emb = self.time_projection(t)[:, :, None, None]
        h = h + time_emb
        
        h = self.norm2(h)
        h = torch.relu(h)
        h = self.conv2(h)
        
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        q = self.q(h).reshape(B, C, H * W)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W)
        
        attention = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(C), dim=-1)
        out = torch.bmm(v, attention.transpose(1, 2)).reshape(B, C, H, W)
        
        return x + self.projection(out)

class DDPMUNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        time_dim = base_channels * 4
        
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        
        # Down
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.pool = nn.AvgPool2d(2)
        
        # Middle
        self.mid1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.attention = AttentionBlock(base_channels * 2)
        self.mid2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        
        # Up
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # According to transcript, concatenated with d1 (base_channels).
        # h from up has base_channels * 2. So total = base_channels * 3
        self.up1 = ResBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        self.up2 = ResBlock(base_channels, base_channels, time_dim)
        
        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x = self.conv_in(x)
        
        # Down
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        
        # Middle
        h = self.pool(d2)
        h = self.mid1(h, t_emb)
        h = self.attention(h)
        h = self.mid2(h, t_emb)
        
        # Up
        h = self.up(h)
        h = torch.cat([h, d1], dim=1) # Note: transcript logic matches concatenation with d1
        h = self.up1(h, t_emb)
        h = self.up2(h, t_emb)
        
        return self.conv_out(h)

def forward_diffusion(x0, t):
    sqrt_alpha_bars = torch.sqrt(alpha_bars[t])[:, None, None, None].to(x0.device)
    sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars[t])[:, None, None, None].to(x0.device)
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bars * x0 + sqrt_one_minus_alpha_bars * noise
    return xt, noise

if __name__ == "__main__":
    model = DDPMUNet(base_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on {device}...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            t = torch.randint(0, T, (x.shape[0],)).to(device)
            
            xt, noise = forward_diffusion(x, t)
            
            noise_predicted = model(xt, t)
            loss = ((noise_predicted - noise) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
