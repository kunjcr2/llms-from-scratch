import torch
import torch.nn as nn

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
        
        if target_len is None:
            target_len = n

        logits_list = []
        mask = torch.zeros(batch_size, n, dtype=torch.bool, device=encoder_outputs.device)
        predictions = []

        for t in range(target_len):
            h_t, c_t = self.lstm_cell(decoder_input, (h_t, c_t))
            
            decoder_features = self.W2(h_t).unsqueeze(1)
            energy = torch.tanh(encoder_features + decoder_features)
            scores = self.v(energy).squeeze(2)
            
            # Mask already selected indices
            scores = scores.masked_fill(mask, -1e8)
            logits_list.append(scores)
            
            predicted_idx = scores.argmax(dim=1)
            predictions.append(predicted_idx)
            
            # Teacher forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_indices = targets[:, t]
            else:
                next_indices = predicted_idx
                
            mask = mask.clone()
            mask.scatter_(1, next_indices.unsqueeze(1), True)
            
            batch_indices = torch.arange(batch_size, device=encoder_outputs.device)
            decoder_input = encoder_outputs[batch_indices, next_indices, :]

        logits_tensor = torch.stack(logits_list, dim=1)
        predictions_tensor = torch.stack(predictions, dim=1)
        return logits_tensor, predictions_tensor
