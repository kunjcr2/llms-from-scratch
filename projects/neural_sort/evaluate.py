import torch
from data.generator import get_dataloader
from model.model import PointerSortNet
from config import config

def num_inversions(arr):
    # arr: 1D torch or numpy array
    inv = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    vocab_range = config.value_range[1] - config.value_range[0] + 1
    
    model = PointerSortNet(
        vocab_size=vocab_range, 
        d_model=config.d_model, 
        nhead=config.nhead, 
        num_layers=config.num_layers
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("checkpoint_n16.pt"))
        print("Loaded checkpoint_n16.pt")
    except:
        print("No checkpoint found. Evaluating untrained model.")
        
    model.eval()
    
    test_size = 10000
    dataloader = get_dataloader(size=test_size, n=config.n, value_range=config.value_range, batch_size=config.batch_size)
    
    exact_matches = 0
    element_correct = 0
    total_elements = 0
    total_inversions = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x_shifted = x - config.value_range[0]
            x_shifted, y = x_shifted.to(device), y.to(device)
            
            # Predict permutation
            logits, preds = model(x_shifted, teacher_forcing_ratio=0.0)
            
            # Metrics computation
            # 1. Exact match
            exact_matches += (preds == y).all(dim=1).sum().item()
            
            # 2. Element-wise match
            element_correct += (preds == y).sum().item()
            total_elements += x.size(0) * x.size(1)
            
            # 3. Inversions
            # preds gives indices. We need to apply these to x to get sorted x
            # then count inversions. Or just calculate inversions on x[preds]
            for b_idx in range(x.size(0)):
                pred_indices = preds[b_idx]
                sorted_x_pred = x[b_idx][pred_indices]
                total_inversions += num_inversions(sorted_x_pred.tolist())
            
    print("\n--- Evaluation Results ---")
    print(f"1. Exact match accuracy: {(exact_matches / test_size * 100):.2f}%")
    print(f"2. Element-wise accuracy: {(element_correct / total_elements * 100):.2f}%")
    print(f"3. Average inversion count: {(total_inversions / test_size):.2f}")

if __name__ == '__main__':
    evaluate()
