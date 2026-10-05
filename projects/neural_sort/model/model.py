import torch.nn as nn
from .encoder import Encoder
from .pointer_decoder import PointerDecoder

class PointerSortNet(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, nhead, num_layers)
        self.decoder = PointerDecoder(d_model)

    def forward(self, src, teacher_forcing_ratio=0.0, targets=None):
        # src: (batch, n)
        encoder_outputs = self.encoder(src)
        
        target_len = src.size(1)
        logits, predictions = self.decoder(
            encoder_outputs, 
            target_len=target_len, 
            teacher_forcing_ratio=teacher_forcing_ratio, 
            targets=targets
        )
        return logits, predictions
