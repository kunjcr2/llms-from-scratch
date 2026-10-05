import torch
import torch.nn as nn
import torch.optim as optim
from data.generator import get_dataloader
from model.model import PointerSortNet
from config import config

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    vocab_range = config.value_range[1] - config.value_range[0] + 1
    # Adding 1 for safety, just shift min to 0. 
    # If range is [-100, 100], vocab_size is 201.
    
    model = PointerSortNet(
        vocab_size=vocab_range, 
        d_model=config.d_model, 
        nhead=config.nhead, 
        num_layers=config.num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Curriculum settings from config: start with n=4
    current_n = 4
    step = 0
    
    model.train()
    
    while step < config.curriculum_steps:
        # Step intervals for curriculum are scaled down or fixed. User requested:
        # "Curriculum: start with n=4 for first 20k steps, then n=8 for next 20k, then n=16"
        # Since I set curriculum_steps=60000 here to match the 20k+20k+20k pattern,
        # let's adapt it to those numbers.
        if step == 20000:
            current_n = 8
            print("Curriculum update: n=8")
        elif step == 40000:
            current_n = 16
            print("Curriculum update: n=16")
            
        dataloader = get_dataloader(size=config.batch_size * 50, n=current_n, value_range=config.value_range, batch_size=config.batch_size)
        
        for x, y in dataloader:
            if step >= config.curriculum_steps:  # Safety break
                break
                
            # Shift x values to completely non-negative range [0, 200]
            x_shifted = x - config.value_range[0]
            
            x_shifted, y = x_shifted.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Use strict teacher forcing to preserve valid target mask
            logits, preds = model(x_shifted, teacher_forcing_ratio=1.0, targets=y)
            
            # logits: (batch, n, n), targets: (batch, n)
            loss = criterion(logits.view(-1, current_n), y.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 500 == 0:
                # Calculate exact match
                corrects = (preds == y).all(dim=1).sum().item()
                exact_match = corrects / config.batch_size * 100
                print(f"Step {step:05d} | n={current_n:02d} | Loss: {loss.item():.4f} | Exact Match: {exact_match:6.2f}%")
                
                if exact_match > 99.0:
                    print(f"Reached >99% exact match at step {step}. Saving checkpoint...")
                    torch.save(model.state_dict(), f"checkpoint_n{current_n}.pt")
                    
            step += 1
            if step >= config.curriculum_steps:
                break
            
if __name__ == '__main__':
    # Override config curriculum steps to 60k
    config.curriculum_steps = 60000
    train()
