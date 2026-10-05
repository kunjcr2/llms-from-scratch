
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Architecture Components ---

class ImageEncoder(nn.Module):
    """
    Minimal image encoder.
    In reality, this would be a ResNet or ViT.
    Here, we use a simple ConvNet for demonstration.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Input: (B, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.fc(x)

class TextEncoder(nn.Module):
    """
    Minimal text encoder.
    In reality, this would be a Transformer (BERT/GPT).
    Here, we use a global embedding pooling + linear projection.
    """
    def __init__(self, vocab_size=1000, embed_dim=256, max_seq_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        # Simple attention-ish pooling or just mean pooling
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        # x: (B, SeqLen)
        embeds = self.embedding(x) # (B, Seq, 64)
        # Mean pooling over sequence dimension to get sentence representation
        text_repr = embeds.mean(dim=1)
        return self.fc(text_repr)

class SimpleDualEncoderVLM(nn.Module):
    def __init__(self, embed_dim=256, vocab_size=1000, temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / temperature))
        
        # In CLIP, there are often separate projection heads.
        # Here we incorporated the projection into the encoder's last fc layer for simplicity.

    def forward(self, images, text):
        # Get Features
        I_e = self.image_encoder(images) # (B, D)
        T_e = self.text_encoder(text)    # (B, D)

        # Normalize Features (Crucial for Contrastive Loss)
        I_e = F.normalize(I_e, p=2, dim=1)
        T_e = F.normalize(T_e, p=2, dim=1)

        return I_e, T_e

# --- 2. Loss Function ---

def symmetric_contrastive_loss(image_embeddings, text_embeddings, logit_scale):
    """
    Computes InfoNCE loss symmetrically.
    """
    # Calculate cosine similarity matrix
    # (B, D) @ (D, B) -> (B, B)
    logits = logit_scale * (image_embeddings @ text_embeddings.T)
    
    # Target is just the identity matrix (image i matches text i)
    # We use range(B) as targets for CrossEntropyLoss
    batch_size = logits.shape[0]
    targets = torch.arange(batch_size, device=logits.device)
    
    # Image-to-Text Loss
    loss_i2t = F.cross_entropy(logits, targets)
    
    # Text-to-Image Loss
    loss_t2i = F.cross_entropy(logits.T, targets)
    
    return (loss_i2t + loss_t2i) / 2

# --- 3. Training Loop Demo ---

def run_training_demo():
    print("Initializing VLM Demo...")
    BATCH_SIZE = 8
    EMBED_DIM = 64
    VOCAB_SIZE = 100
    STEPS = 5

    model = SimpleDualEncoderVLM(embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Model created. Running for {STEPS} steps on dummy data.")

    for step in range(STEPS):
        # 1. Create Dummy Data
        # Images: random noise
        images = torch.randn(BATCH_SIZE, 3, 32, 32)
        # Text: random integer tokens
        text = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 10))

        # 2. Forward Pass
        I_embeds, T_embeds = model(images, text)

        # 3. Compute Loss
        loss = symmetric_contrastive_loss(I_embeds, T_embeds, model.logit_scale.exp())

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}/{STEPS} | Loss: {loss.item():.4f}")

    print("Demo Finished Successfully.")

if __name__ == "__main__":
    run_training_demo()
