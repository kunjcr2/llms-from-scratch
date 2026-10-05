## Sarcasm Detection

# !pip install -q opendatasets transformers
import opendatasets as od
od.download("https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection")

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device}')

df = pd.read_json('/content/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines=True)
df.head()

# Sampling similar distribution for both, dropping rticle link
min_size = min(df.groupby('is_sarcastic').size())
df = df.groupby('is_sarcastic').apply(lambda x: x.sample(min_size)).reset_index(drop=True)
df.drop(columns=['article_link'], inplace=True)

print(df.head())
print("="*20)
print(df['is_sarcastic'].value_counts())

# bert model and tokenizer
bert_model = AutoModel.from_pretrained('bert-base-uncased')
tok = AutoTokenizer.from_pretrained('bert-base-uncased')

# train, test, val splits
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
print('Number of headlines in X_train:', len(X_train))
print('Number of headlines in X_test:', len(X_test))
print('Number of headlines in X_val:', len(X_val))

class SarcasmDataset(Dataset):
    """
    Custom Dataset for Sarcasm Detection with X being tokenized headlines and y being labels.
    """
    def __init__(self, headlines, labels, tokenizer):
        self.X = [tokenizer(
            headline,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) for headline in headlines]
        self.y = torch.Tensor(labels.values).float()

    def __len__(self):
        return len(self.X)
      
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SarcasmDataset(X_train, y_train, tok)
test_ds = SarcasmDataset(X_test, y_test, tok)
val_ds = SarcasmDataset(X_val, y_val, tok)

# Hyperparameters
LR=1e-3
BATCH_SIZE=16
EPOCHS=10

# Data loaders
train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

for param in bert_model.parameters():
    param.requires_grad = False # Freezing model weights

class BertModel(nn.Module):
  """
  Bert Model with 2 additional FC layers for binary classification.
  """
  def __init__(self, model):
    super().__init__()
    self.model = model
    self.fc1 = nn.Linear(768, 384) # 768 <- embedding dim
    self.fc2 = nn.Linear(384, 1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)

  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids, attention_mask)[0][:, 0] # first [CLS] token for each
    output = self.relu(self.dropout(self.fc1(output)))
    output = self.fc2(output)
    return output

model = BertModel(bert_model)
model.to(device)
model = torch.compile(model) # compiling beforehand

loss_fn = nn.BCEWithLogitsLoss() # Requires Logits and no Sigmoid before calculating the loss.
optim = Adam(model.parameters(), lr=LR)

train_loss_plot, val_loss_plot, train_acc_plot, val_acc_plot = [], [], [], []

# Training loops
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    # train round
    for data, target in train:
        # squeezing through first dimensions for input_ids and attention_mask
        input_ids = data["input_ids"].squeeze(1).to(device)
        attention_mask = data["attention_mask"].squeeze(1).to(device)
        target = target.float().to(device)

        optim.zero_grad()

        # Squeezing the last dim such that the dimension is (16, ) instead of (16, 1)
        logits = model(input_ids, attention_mask).squeeze(-1)  # (B,)
        loss = loss_fn(logits, target)

        loss.backward()
        optim.step()

        train_loss += loss.item() * target.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).long()
        train_correct += (preds == target.long()).sum().item()
        train_total += target.size(0)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    # val round
    with torch.no_grad():
        for data, target in val:
            input_ids = data["input_ids"].squeeze(1).to(device)
            attention_mask = data["attention_mask"].squeeze(1).to(device)
            target = target.float().to(device)

            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = loss_fn(logits, target)

            val_loss += loss.item() * target.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).long()
            val_correct += (preds == target.long()).sum().item()
            val_total += target.size(0)

    train_loss /= train_total
    val_loss /= val_total
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total

    # plotting
    train_loss_plot.append(train_loss)
    train_acc_plot.append(train_acc)
    val_loss_plot.append(val_loss)
    val_acc_plot.append(val_acc)

    # printing
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    print("=" * 30)

model.eval()
all_preds = []
all_targets = []

# test round
with torch.no_grad():
    for data, target in test:
        input_ids = data["input_ids"].squeeze(1).to(device)
        attention_mask = data["attention_mask"].squeeze(1).to(device)
        target = target.to(device)

        logits = model(input_ids, attention_mask).squeeze(-1)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())

# concatenate batches
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
test_acc = accuracy_score(all_targets, all_preds)

print(f"Test Accuracy: {test_acc:.4f}")

epochs = range(1, len(train_loss_plot) + 1)
plt.figure(figsize=(12, 5))

# Plotting

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_plot, label="Train Loss")
plt.plot(epochs, val_loss_plot, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_plot, label="Train Accuracy")
plt.plot(epochs, val_acc_plot, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()

plt.tight_layout()
plt.show()

