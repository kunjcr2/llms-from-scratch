import torch
import torch.nn as nn

from block import Block

class GatorGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 8,
        gqa_groups: int = 2,
        max_len: int = 1024,
        d_ff: int = 768,
        eps: float = 1e-5,
        dropout_p: float = 0.0,
        blocks: int = 10,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        self.final_rms = nn.modules.normalization.RMSNorm(d_model, eps)
        self.embed.weight = self.unembed.weight

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    gqa_groups=gqa_groups,
                    max_len=max_len,
                    d_ff=d_ff,
                    eps=eps,
                    dropout_p=dropout_p,
                ) for _ in range(blocks)
            ]
        )

        if device is not None:
          self.to(device)

    def forward(self, x):
        """
        Forward method that takes in the tokens
        """
      # x: (batch, seq_len) of token ids
        h = self.embed(x)                 # (batch, seq_len, d_model)
        for block in self.blocks:         # run each transformer block
            h = block(h)
        h = self.final_rms(h)
        logits = self.unembed(h)          # (batch, seq_len, vocab_size)
        return logits
