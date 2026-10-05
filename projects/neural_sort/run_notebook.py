import torch
from torch.utils.data import Dataset, DataLoader

class SortDataset(Dataset):
    def __init__(self, size, n, value_range):
        self.size = size
        self.n = n
        self.min_val, self.max_val = value_range

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(self.min_val, self.max_val + 1, (self.n,))
        y = torch.argsort(x)
        return x, y

dataset = SortDataset(10, 8, (-100, 100))
x, y = dataset[0]
print(f"Input x: {x.tolist()}")
print(f"Sorted x: {x[y].tolist()}")
print(f"Permutation indices y: {y.tolist()}")

import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 256, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        return self.transformer_encoder(src)

class PointerDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)
        self.lstm_cell = nn.LSTMCell(d_model, d_model)
        
        self.decoder_start_input = nn.Parameter(torch.randn(1, d_model))
        self.decoder_start_state = nn.Parameter(torch.randn(1, d_model))
        self.decoder_start_cell = nn.Parameter(torch.randn(1, d_model))

    def forward(self, encoder_outputs, target_len=None, teacher_forcing_ratio=0.0, targets=None):
        batch_size, n, d_model = encoder_outputs.size()
        h_t = self.decoder_start_state.expand(batch_size, -1)
        c_t = self.decoder_start_cell.expand(batch_size, -1)
        decoder_input = self.decoder_start_input.expand(batch_size, -1)
        
        encoder_features = self.W1(encoder_outputs)
        logits_list, predictions = [], []
        mask = torch.zeros(batch_size, n, dtype=torch.bool, device=encoder_outputs.device)

        t_len = n if target_len is None else target_len
        for t in range(t_len):
            h_t, c_t = self.lstm_cell(decoder_input, (h_t, c_t))
            decoder_features = self.W2(h_t).unsqueeze(1)
            scores = self.v(torch.tanh(encoder_features + decoder_features)).squeeze(2)
            scores = scores.masked_fill(mask, -1e8)
            logits_list.append(scores)
            
            predicted_idx = scores.argmax(dim=1)
            predictions.append(predicted_idx)
            
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_indices = targets[:, t]
            else:
                next_indices = predicted_idx
                
            mask = mask.clone()
            mask.scatter_(1, next_indices.unsqueeze(1), True)
            
            batch_indices = torch.arange(batch_size, device=encoder_outputs.device)
            decoder_input = encoder_outputs[batch_indices, next_indices, :]

        return torch.stack(logits_list, dim=1), torch.stack(predictions, dim=1)

class PointerSortNet(nn.Module):
    def __init__(self, vocab_size=201, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_layers)
        self.decoder = PointerDecoder(d_model)

    def forward(self, src, teacher_forcing_ratio=0.0, targets=None):
        encoder_outputs = self.encoder(src)
        return self.decoder(encoder_outputs, src.size(1), teacher_forcing_ratio, targets)

model = PointerSortNet()
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")

import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

n = 8
batch_size = 256
steps = 2000

dataset = SortDataset(batch_size * 50, n, (-100, 100))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()
step = 0
while step < steps:
    for x, y in dataloader:
        if step >= steps: break
        
        x_shifted = (x - (-100)).to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits, preds = model(x_shifted, teacher_forcing_ratio=1.0, targets=y)
        
        score = criterion(logits.view(-1, n), y.view(-1))
        score.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 200 == 0:
            exact_match = (preds == y).all(dim=1).sum().item() / batch_size * 100
            print(f"Step {step:04d} | Loss: {score.item():.4f} | Exact Match: {exact_match:.2f}%")
            
        step += 1
print("Training complete!")

def num_inversions(arr):
    inv = 0
    size = len(arr)
    for i in range(size):
        for j in range(i + 1, size):
            if arr[i] > arr[j]: inv += 1
    return inv

model.eval()
test_size = 500
test_dataset = SortDataset(test_size, n, (-100, 100))
test_loader = DataLoader(test_dataset, batch_size=64)

exact_matches, element_correct, total_elements, total_inversions = 0, 0, 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x_shifted = (x - (-100)).to(device)
        y = y.to(device)
        _, preds = model(x_shifted)
        
        exact_matches += (preds == y).all(dim=1).sum().item()
        element_correct += (preds == y).sum().item()
        total_elements += x.size(0) * x.size(1)
        
        for b_idx in range(x.size(0)):
            pred_indices = preds[b_idx].cpu()
            sorted_x_pred = x[b_idx][pred_indices]
            total_inversions += num_inversions(sorted_x_pred.tolist())
        
print(f"1. Exact match accuracy: {(exact_matches / test_size * 100):.2f}%")
print(f"2. Element-wise accuracy: {(element_correct / total_elements * 100):.2f}%")
print(f"3. Average inversion count: {(total_inversions / test_size):.2f}")

# No matplotlib required for simple array printing constraints
demo_dataset = SortDataset(5, n, (-100, 100))
model.eval()

with torch.no_grad():
    for i in range(5):
        x, y = demo_dataset[i]
        x_shifted = (x - (-100)).unsqueeze(0).to(device)
        
        _, preds = model(x_shifted)
        pred_indices = preds[0].cpu()
        
        print(f"Sample {i+1}:")
        print(f"  Input:         {x.tolist()}")
        print(f"  Model Output:  {x[pred_indices].tolist()}")
        print(f"  Correct Sort:  {x[y].tolist()}")
        print("-" * 50)
