
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Architecture Components ---

class SimpleEncoder(nn.Module):
    """
    Generic encoder.
    """
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    """
    Projection head to map representation h -> z.
    Crucial for SimCLR.
    """
    def __init__(self, in_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), # Hidden layer maintains size
            nn.ReLU(),
            nn.Linear(in_dim, out_dim) # Project to embedding space
        )

    def forward(self, x):
        return self.net(x)

class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.projection_head = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=1) # Normalize z

# --- 2. Loss Function (NT-Xent) ---

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Calculates NT-Xent loss for a batch of positive pairs (z_i, z_j).
    Batch size N -> 2N views.
    """
    batch_size = z_i.shape[0]
    
    # Concatenate all views: [z_i; z_j] -> Size (2N, D)
    features = torch.cat([z_i, z_j], dim=0)
    
    # Compute similarity matrix: (2N, 2N)
    similarity_matrix = torch.matmul(features, features.T)
    
    # Create mask to ignore self-similarity
    mask = torch.eye(2 * batch_size, device=features.device).bool()
    
    # Discard global similarity diagonals (self-similarity)
    # We want to predict the partner view for each view.
    # For index k (0 to N-1), partner is k + N
    # For index k (N to 2N-1), partner is k - N
    
    # We can simplify by just using CrossEntropy.
    # The logits are similarity / temperature.
    logits = similarity_matrix / temperature
    
    # Mask out self-similarity by filling diagonal with -inf
    logits.masked_fill_(mask, -9e15)
    
    # Targets:
    # row 0 matches row N
    # row 1 matches row N+1
    # row 2 matches row N+2 ...
    # row N-1 matches row 2N-1
    # row N matches row 0
    labels = torch.cat([
        torch.arange(batch_size, device=features.device) + batch_size,
        torch.arange(batch_size, device=features.device)
    ], dim=0)
    
    loss = F.cross_entropy(logits, labels)
    return loss

# --- 3. Training Loop Demo ---

def run_training_demo():
    print("Initializing Contrastive Learning Demo...")
    BATCH_SIZE = 16
    INPUT_DIM = 128
    STEPS = 5

    model = ContrastiveModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model created. Running for {STEPS} steps on dummy data.")

    for step in range(STEPS):
        # 1. Dummy Data & "Augmentation"
        # In real life, you load an image x, and create x_i = aug(x), x_j = aug(x)
        # Here we just generate random noise for base, and add noise for views.
        base_data = torch.randn(BATCH_SIZE, INPUT_DIM)
        
        view_1 = base_data + 0.1 * torch.randn_like(base_data)
        view_2 = base_data + 0.1 * torch.randn_like(base_data)

        # 2. Forward Pass
        z_1 = model(view_1)
        z_2 = model(view_2)

        # 3. Compute Loss
        loss = nt_xent_loss(z_1, z_2, temperature=0.5)

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}/{STEPS} | Loss: {loss.item():.4f}")

    print("Demo Finished Successfully.")

if __name__ == "__main__":
    run_training_demo()
