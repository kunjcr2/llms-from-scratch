import torch
import torch.nn as nn
import torch.nn.functional as F

from rms_norm import RMSNorm

class MTPModule(nn.Module):
    """
    Multi Token Prediction (MTP) Module
    
    This module predicts multiple future tokens simultaneously rather than just the next token.
    It uses multiple prediction heads, where each head predicts tokens at different future positions.
    """

    def __init__(self, num_heads, d_model, vocab_size):
        """
        Initialize the Multi Token Prediction module
        
        Args:
            num_heads (int): Number of prediction heads (how many future tokens to predict)
            d_model (int): Hidden dimension size of the model
            vocab_size (int): Size of the vocabulary (number of possible tokens)
        """
        super().__init__()
        
        self.num_heads = num_heads      # How many future tokens we'll predict
        self.vocab_size = vocab_size    # Total number of possible tokens
        self.d_model = d_model          # Hidden state dimension
        
        # Normalization layer - helps with training stability
        self.rms = RMSNorm(self.d_model)
        
        # Projection layers - one for each prediction head
        # These combine current token + previous hidden state into new representation
        self.proj = nn.ModuleList([
            nn.Linear(2*d_model, d_model) for _ in range(num_heads)
        ])
        
        # Transformer decoder layers - one for each prediction head
        # These process the projected representations
        self.tfmr_blk = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads) for _ in range(num_heads)
        ])

        # Token embedding and unembedding layers
        self.emb = nn.Embedding(vocab_size, d_model)  # Convert tokens to vectors
        self.unemb = nn.Linear(d_model, vocab_size)   # Convert vectors back to token probabilities

    def forward(self, token_ids, hidden_states):
        """
        Forward pass through the MTP module
        
        Args:
            token_ids: Input token sequence [batch_size, sequence_length]
            hidden_states: Previous hidden states (can be None for first pass)
            
        Returns:
            Tensor of logits for multiple future token predictions
            Shape: [sequence_positions, batch_size, num_heads, vocab_size]
        """
        B, T = token_ids.shape  # B = batch size, T = sequence length

        # Convert token IDs to embeddings
        embed = self.emb(token_ids)

        # Use provided hidden states or start with embeddings
        if hidden_states is None:
            h0_seq = embed  # Start with token embeddings
        else:
            h0_seq = hidden_states  # Use provided hidden states

        outputs = []  # Will store predictions for each sequence position
        
        # Calculate how many positions we can make predictions for
        # We need at least num_heads tokens ahead to predict
        last_input = T - self.num_heads - 1

        # Process each position in the sequence
        for i in range(0, last_input + 1):
            # Get the hidden state at current position
            h_prev = h0_seq[:, i, :]
            logits = []  # Store logits from each prediction head

            # Use each prediction head to predict future tokens
            for k in range(self.num_heads):
                # Get the token embedding k+1 positions ahead
                curr_input = embed[:, i+k+1, :]

                # Apply RMS normalization to both current input and previous hidden state
                rms_curr = self.rms(curr_input)      # Normalize current token
                rms_hidden = self.rms(h_prev)        # Normalize previous hidden state

                # Combine normalized current token and previous hidden state
                combined_seq = torch.cat([rms_curr, rms_hidden], dim=-1)

                # Project the combined representation to model dimension
                proj_seq = self.proj[k](combined_seq)

                # Pass through transformer decoder layer
                # Note: Adding/removing dimensions for transformer layer compatibility
                h_curr = self.tfmr_blk[k](proj_seq.unsqueeze(0)).squeeze(0)
                
                # Generate logits (token predictions) from current hidden state
                curr_logit = self.unemb(h_curr)
                logits.append(curr_logit)

                # Update hidden state for next prediction head
                h_prev = h_curr
            
            # Stack logits from all prediction heads
            logits_k = torch.stack(logits, dim=1)  # [batch_size, num_heads, vocab_size]
            outputs.append(logits_k)
        
        # Stack all position outputs and rearrange dimensions
        out = torch.stack(outputs, dim=0)  # [seq_pos, batch_size, num_heads, vocab_size]
        out = out.contiguous().permute(1,0,2,3).contiguous()  # Rearrange dimensions

        return out
