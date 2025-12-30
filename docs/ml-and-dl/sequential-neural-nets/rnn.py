"""
RNN Training Pipeline

Demonstrates RNN usage with PyTorch's nn.RNN on a synthetic sequence dataset.
Task: Predict the next value in a sine wave sequence.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


# --- Dataset ---

class SineWaveDataset(Dataset):
    """
    Synthetic dataset: predict next value in a sine wave.
    """
    
    def __init__(self, num_samples: int = 1000, seq_len: int = 50):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.data = self._generate_data()
    
    def _generate_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.num_samples):
            # Random starting point and frequency
            start = np.random.uniform(0, 2 * np.pi)
            freq = np.random.uniform(0.5, 2.0)
            
            # Generate sine wave
            t = np.linspace(start, start + freq * np.pi, self.seq_len + 1)
            wave = np.sin(t).astype(np.float32)
            
            # Input: first seq_len values, Target: next value
            x = torch.tensor(wave[:-1]).unsqueeze(-1)  # (seq_len, 1)
            y = torch.tensor(wave[-1])  # scalar
            data.append((x, y))
        
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


# --- Model ---

class RNNPredictor(nn.Module):
    """
    RNN for sequence regression using PyTorch's nn.RNN.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            Prediction: (batch,)
        """
        # output: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        output, h_n = self.rnn(x)
        
        # Use last hidden state from final layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        return self.fc(last_hidden).squeeze(-1)


# --- Training ---

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: torch.device = None
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    
    return history


# --- Main ---

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 50
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    EPOCHS = 50
    LR = 0.001
    
    # Create datasets
    train_dataset = SineWaveDataset(num_samples=800, seq_len=SEQ_LEN)
    val_dataset = SineWaveDataset(num_samples=200, seq_len=SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    model = RNNPredictor(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 50)
    
    # Train
    history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    
    # Test prediction
    print("\n--- Test Prediction ---")
    model.eval()
    x, y_true = val_dataset[0]
    x = x.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        y_pred = model(x).item()
    
    print(f"True next value: {y_true:.4f}")
    print(f"Predicted value: {y_pred:.4f}")
    print(f"Error: {abs(y_true - y_pred):.4f}")
