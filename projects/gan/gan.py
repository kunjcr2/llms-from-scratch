"""
Generative Adversarial Network (GAN) for MNIST Digit Generation
================================================================
A straightforward DCGAN-style implementation that generates 28x28
grayscale images of handwritten digits using the MNIST dataset.

Usage:
    python gan.py              # Train the GAN
    python gan.py --inference  # Generate digits from a saved model
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# ─── Hyperparameters ────────────────────────────────────────────────────────────

LATENT_DIM = 100        # Size of the noise vector z
CHANNELS = 1            # Grayscale images
IMG_SIZE = 28           # MNIST is 28x28
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
EPOCHS = 50
SAMPLE_INTERVAL = 500   # Save generated images every N batches
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Generator ──────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Maps a latent noise vector z (100-dim) to a 1x28x28 image
    using transposed convolutions (learnable upsampling).

    Architecture:
        z (100,1,1) → ConvT → (256,7,7) → (128,14,14) → (64,28,28) → (1,28,28)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.model = nn.Sequential(
            # (B, 100, 1, 1) → (B, 256, 7, 7)
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 256, 7, 7) → (B, 128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 14, 14) → (B, 64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 28, 28) → (B, 1, 28, 28)
            nn.ConvTranspose2d(64, CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),  # Output in [-1, 1] to match normalized images
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Reshape noise vector to (B, latent_dim, 1, 1) for conv input
        z = z.view(z.size(0), -1, 1, 1)
        return self.model(z)


# ─── Discriminator ──────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Takes a 1x28x28 image and outputs a probability (real vs fake)
    using strided convolutions (learnable downsampling).

    Architecture:
        (1,28,28) → Conv → (64,14,14) → (128,7,7) → (256,4,4) → (1,1,1)
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # (B, 1, 28, 28) → (B, 64, 14, 14) — no BatchNorm on first layer (DCGAN convention)
            nn.Conv2d(CHANNELS, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 14, 14) → (B, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 7, 7) → (B, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 256, 4, 4) → (B, 1, 1, 1)
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img).view(img.size(0), 1)  # Flatten to (B, 1)


# ─── Training ───────────────────────────────────────────────────────────────────

def train():
    """Train the GAN on MNIST."""

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ── Data ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ── Models ──
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # ── Optimizers ──
    opt_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # ── Loss ──
    criterion = nn.BCELoss()

    # ── Labels ──
    real_label = 1.0
    fake_label = 0.0

    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    print(f"Generator params:     {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print("-" * 60)

    batches_done = 0

    for epoch in range(EPOCHS):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # Ground truths
            real_targets = torch.full((batch_size, 1), real_label, device=DEVICE)
            fake_targets = torch.full((batch_size, 1), fake_label, device=DEVICE)

            # ────────────────────────────────────────────────────────
            # 1) Train Discriminator
            # ────────────────────────────────────────────────────────
            opt_D.zero_grad()

            # Loss on real images
            real_preds = discriminator(real_imgs)
            d_loss_real = criterion(real_preds, real_targets)

            # Loss on fake images
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z).detach()  # Detach to avoid training G
            fake_preds = discriminator(fake_imgs)
            d_loss_fake = criterion(fake_preds, fake_targets)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_D.step()

            # ────────────────────────────────────────────────────────
            # 2) Train Generator
            # ────────────────────────────────────────────────────────
            opt_G.zero_grad()

            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z)
            gen_preds = discriminator(gen_imgs)
            g_loss = criterion(gen_preds, real_targets)  # Fool D: want D to say "real"

            g_loss.backward()
            opt_G.step()

            # ── Logging ──
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{EPOCHS}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"D_loss: {d_loss.item():.4f}  "
                    f"G_loss: {g_loss.item():.4f}"
                )

            # ── Save sample images ──
            if batches_done % SAMPLE_INTERVAL == 0:
                save_image(
                    gen_imgs.data[:25],
                    f"outputs/{batches_done:06d}.png",
                    nrow=5,
                    normalize=True,
                )
            batches_done += 1

    # ── Save final model ──
    torch.save(generator.state_dict(), "checkpoints/generator.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator.pth")
    print("\nTraining complete. Models saved to checkpoints/")


# ─── Inference ──────────────────────────────────────────────────────────────────

def generate(num_images: int = 25, output_path: str = "outputs/generated.png"):
    """Load a trained Generator and produce synthetic MNIST digits."""

    generator = Generator().to(DEVICE)
    generator.load_state_dict(torch.load("checkpoints/generator.pth", map_location=DEVICE))
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_images, LATENT_DIM, device=DEVICE)
        images = generator(z)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(images, output_path, nrow=5, normalize=True)
    print(f"Generated {num_images} images → {output_path}")


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN for MNIST digit generation")
    parser.add_argument("--inference", action="store_true", help="Generate images from saved model")
    parser.add_argument("--num-images", type=int, default=25, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="outputs/generated.png", help="Output image path")
    args = parser.parse_args()

    if args.inference:
        generate(num_images=args.num_images, output_path=args.output)
    else:
        train()
