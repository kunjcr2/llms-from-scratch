"""
LSTM Training Pipeline

Demonstrates LSTM usage with PyTorch's nn.LSTM on a text classification dataset.
Task: Sentiment classification on synthetic movie review-like data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random


# --- Dataset ---

class SentimentDataset(Dataset):
    """
    Synthetic sentiment dataset with positive/negative sentences.
    """
    
    # Simple vocabulary with sentiment-bearing words
    POSITIVE_WORDS = [
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "best", "happy", "beautiful", "perfect", "brilliant"
    ]
    NEGATIVE_WORDS = [
        "bad", "terrible", "awful", "horrible", "worst", "hate",
        "boring", "poor", "disappointing", "ugly", "annoying", "waste"
    ]
    NEUTRAL_WORDS = [
        "the", "a", "is", "was", "it", "this", "that", "movie", "film",
        "story", "really", "very", "quite", "just", "so", "and", "but"
    ]
    
    def __init__(self, num_samples: int = 1000, max_len: int = 20):
        self.max_len = max_len
        
        # Build vocabulary
        all_words = self.POSITIVE_WORDS + self.NEGATIVE_WORDS + self.NEUTRAL_WORDS
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for word in all_words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        
        self.vocab_size = len(self.word2idx)
        self.data = self._generate_data(num_samples)
    
    def _generate_sentence(self, sentiment: int) -> list[str]:
        """Generate a sentence with given sentiment (0=neg, 1=pos)."""
        length = random.randint(5, self.max_len)
        words = []
        
        # Add sentiment words based on label
        sentiment_words = self.POSITIVE_WORDS if sentiment == 1 else self.NEGATIVE_WORDS
        num_sentiment = random.randint(1, 3)
        
        for _ in range(length):
            if num_sentiment > 0 and random.random() < 0.3:
                words.append(random.choice(sentiment_words))
                num_sentiment -= 1
            else:
                words.append(random.choice(self.NEUTRAL_WORDS))
        
        random.shuffle(words)
        return words
    
    def _generate_data(self, num_samples: int) -> list[tuple[torch.Tensor, int]]:
        data = []
        for _ in range(num_samples):
            sentiment = random.randint(0, 1)
            words = self._generate_sentence(sentiment)
            indices = [self.word2idx.get(w, 1) for w in words]
            data.append((torch.tensor(indices, dtype=torch.long), sentiment))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx]


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    Returns padded sequences, labels, and lengths.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    
    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded, labels, lengths


# --- Model ---

class LSTMClassifier(nn.Module):
    """
    LSTM for text classification using PyTorch's nn.LSTM.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension depends on bidirectional
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Token indices (batch, seq_len)
            lengths: Actual lengths for each sequence
        
        Returns:
            Logits: (batch, num_classes)
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Pack sequences for efficiency (optional but recommended)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (h_n, c_n) = self.lstm(packed)
        else:
            output, (h_n, c_n) = self.lstm(embedded)
        
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # Concatenate final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        
        return self.classifier(hidden)


# --- Training ---

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y, lengths in dataloader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y, lengths in dataloader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            
            logits = model(x, lengths)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return total_loss / len(dataloader), correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.001,
    device: torch.device = None
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
    
    model.load_state_dict(best_state)
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    
    return history


# --- Main ---

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    EMBED_DIM = 64
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    EPOCHS = 30
    LR = 0.001
    
    # Create datasets
    train_dataset = SentimentDataset(num_samples=800)
    val_dataset = SentimentDataset(num_samples=200)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    
    # Create model
    model = LSTMClassifier(
        vocab_size=train_dataset.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=True
    )
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 60)
    
    # Train
    history = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    
    # Test inference
    print("\n--- Test Inference ---")
    model.eval()
    
    # Get a sample
    x, y_true, lengths = next(iter(val_loader))
    
    with torch.no_grad():
        logits = model(x[:5], lengths[:5])
        preds = logits.argmax(dim=1)
    
    labels = ["Negative", "Positive"]
    for i in range(5):
        print(f"True: {labels[y_true[i]]:8s} | Pred: {labels[preds[i]]}")
