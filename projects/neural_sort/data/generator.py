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

def get_dataloader(size, n, value_range, batch_size):
    dataset = SortDataset(size, n, value_range)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
