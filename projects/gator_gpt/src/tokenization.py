import torch
import tiktoken
from gpt_dataset import GPTDataset

from torch.utils.data import Dataset, DataLoader

###################################### TURNS TEXTS TO TOKENS
def text_to_token_ids(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)
  return encoded_tensor
############################################################################

###################################### TURN TOKENS TO TEXT
def token_ids_to_text(token_ids, tokenizer):
  decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
  return decoded
############################################################################

###################################### CREATES DATA LOADERS
def create_dataloader(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0): # Changed default to 0

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("p50k_base")

    # Create dataset
    dataset = GatorGPTDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers # Pass the potentially modified num_workers
    )

    return dataloader
############################################################################
