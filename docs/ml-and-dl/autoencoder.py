"""
Autoencoder Implementations in PyTorch

This file contains implementations of:
1. Vanilla Autoencoder (fully connected)
2. Convolutional Autoencoder (for images)
3. Variational Autoencoder (VAE)

Each implementation includes the model class and a simple training loop example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# 1. VANILLA AUTOENCODER
# =============================================================================

class VanillaAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder.
    Good for tabular data or flattened images.
    """
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        
        # Encoder: compresses input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        
        # Decoder: reconstructs input from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),  # Output in [0, 1] for normalized data
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


# =============================================================================
# 2. CONVOLUTIONAL AUTOENCODER
# =============================================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for image data.
    Uses Conv2d for encoding and ConvTranspose2d for decoding.
    Designed for 28x28 grayscale images (e.g., MNIST).
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        # Encoder: 1x28x28 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
        )
        
        # Decoder: latent_dim -> 1x28x28
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1x28x28
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decode(z)
        return x_recon
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 7, 7)
        return self.decoder_conv(x)


# =============================================================================
# 3. VARIATIONAL AUTOENCODER (VAE)
# =============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder.
    Learns a probabilistic latent space with mean (mu) and variance (logvar).
    Uses reparameterization trick for backpropagation through sampling.
    """
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        
        # Encoder: outputs mu and logvar for latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Separate heads for mu and logvar
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        This makes sampling differentiable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x, x_recon, mu, logvar, beta: float = 1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        x: Original input
        x_recon: Reconstructed input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term (beta-VAE uses beta > 1)
    
    Returns:
        Total loss, reconstruction loss, KL loss
    """
    # Reconstruction loss (BCE for binary/normalized data)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, 1)
    # Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# =============================================================================
# TRAINING EXAMPLES
# =============================================================================

def train_vanilla_autoencoder():
    """Example training loop for vanilla autoencoder on MNIST."""
    # Hyperparameters
    input_dim = 28 * 28
    latent_dim = 32
    batch_size = 128
    epochs = 10
    lr = 1e-3
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = VanillaAutoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim).to(device)
            
            optimizer.zero_grad()
            recon = model(data)
            loss = F.mse_loss(recon, data, reduction='sum')
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model


def train_vae():
    """Example training loop for VAE on MNIST."""
    # Hyperparameters
    input_dim = 28 * 28
    latent_dim = 20
    batch_size = 128
    epochs = 10
    lr = 1e-3
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = VAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, input_dim).to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(data, recon, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model


def generate_samples(vae_model, num_samples: int = 16, latent_dim: int = 20):
    """Generate new samples from a trained VAE by sampling from latent space."""
    device = next(vae_model.parameters()).device
    vae_model.eval()
    
    with torch.no_grad():
        # Sample from standard normal (prior)
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = vae_model.decode(z)
    
    return samples.view(num_samples, 1, 28, 28)


if __name__ == '__main__':
    print("Training Vanilla Autoencoder...")
    # Uncomment to train:
    # model = train_vanilla_autoencoder()
    
    print("\nTraining VAE...")
    # Uncomment to train:
    # vae = train_vae()
    # samples = generate_samples(vae)
    
    print("\nAutoencoder implementations ready!")
    print("Uncomment the training calls in __main__ to run.")
