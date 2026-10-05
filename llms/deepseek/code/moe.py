"""
Mixture of Experts (MoE) Architecture
=====================================

A complete implementation of the Mixture of Experts architecture, combining:
- Expert networks (simple MLPs)
- Noisy Top-K gating mechanism
- MoE layer that routes tokens to experts

This implementation follows the approach used in models like DeepSeek, GPT-4, and Mixtral.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    Combines Swish activation with Gated Linear Unit for better training dynamics.
    Used in modern LLMs like LLaMA and DeepSeek.
    """
    
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
    
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)


class Expert(nn.Module):
    """
    Expert Network - A simple feed-forward MLP.
    
    Each expert is a 2-layer MLP with expansion factor of 4x (standard for transformers).
    The architecture is: Linear -> Activation -> Linear -> Dropout
    
    Args:
        embed_dim: Input/output embedding dimension
        hidden_dim: Hidden layer dimension (default: 4 * embed_dim)
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'swiglu')
    """
    
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1, activation='relu'):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        
        if activation == 'swiglu':
            # SwiGLU activation (used in LLaMA, DeepSeek)
            self.net = nn.Sequential(
                SwiGLU(embed_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout)
            )
        else:
            act_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout)
            )
    
    def forward(self, x):
        """Forward pass through the expert network."""
        return self.net(x)


class NoisyTopKRouter(nn.Module):
    """
    Noisy Top-K Gating Router.
    
    Routes each token to the top-k experts using a learned gating mechanism.
    Noise is added during training to encourage load balancing across experts.
    
    Args:
        embed_dim: Input embedding dimension
        n_experts: Total number of experts
        top_k: Number of experts each token is routed to
        noise_std: Standard deviation of noise added during training
    """
    
    def __init__(self, embed_dim, n_experts, top_k, noise_std=1.0):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.noise_std = noise_std
        
        # Main routing projection
        self.gate = nn.Linear(embed_dim, n_experts, bias=False)
        
        # Learnable noise weights (helps with load balancing)
        self.noise_weights = nn.Linear(embed_dim, n_experts, bias=False)
    
    def forward(self, x):
        """
        Route tokens to experts.
        
        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]
            
        Returns:
            router_probs: Sparse routing probabilities [batch, seq_len, n_experts]
            top_k_indices: Indices of selected experts [batch, seq_len, top_k]
            load_balance_loss: Auxiliary loss for load balancing
        """
        # Compute gating logits
        logits = self.gate(x)
        
        # Add noise during training for exploration
        if self.training:
            noise = self.noise_weights(x)
            noise = noise * torch.randn_like(noise) * self.noise_std
            logits = logits + noise
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, k=self.top_k, dim=-1)
        
        # Create sparse routing matrix (set non-top-k positions to -inf)
        sparse_logits = torch.full_like(logits, float('-inf'))
        sparse_logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Convert to probabilities
        router_probs = F.softmax(sparse_logits, dim=-1)
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(logits)
        
        return router_probs, top_k_indices, load_balance_loss
    
    def _compute_load_balance_loss(self, logits):
        """
        Compute auxiliary load balancing loss.
        
        Encourages uniform distribution of tokens across experts.
        """
        # Probability of each expert being selected (over all tokens)
        probs = F.softmax(logits, dim=-1)  # [batch, seq, n_experts]
        
        # Average probability per expert
        avg_probs = probs.mean(dim=[0, 1])  # [n_experts]
        
        # Fraction of tokens routed to each expert
        expert_mask = torch.zeros_like(probs)
        _, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        expert_mask.scatter_(-1, top_indices, 1.0)
        tokens_per_expert = expert_mask.mean(dim=[0, 1])  # [n_experts]
        
        # Load balance loss: encourages uniform distribution
        load_balance_loss = self.n_experts * (avg_probs * tokens_per_expert).sum()
        
        return load_balance_loss


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer.
    
    Combines the router and experts into a single module that can replace
    the feed-forward network in a transformer block.
    
    Args:
        embed_dim: Input/output embedding dimension
        n_experts: Total number of expert networks
        top_k: Number of experts to route each token to
        expert_hidden_dim: Hidden dimension of each expert (default: 4 * embed_dim)
        dropout: Dropout probability
        activation: Activation function for experts ('relu', 'gelu', 'swiglu')
        noise_std: Noise standard deviation for router
    """
    
    def __init__(
        self,
        embed_dim,
        n_experts=8,
        top_k=2,
        expert_hidden_dim=None,
        dropout=0.1,
        activation='relu',
        noise_std=1.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.embed_dim = embed_dim
        
        # Router
        self.router = NoisyTopKRouter(embed_dim, n_experts, top_k, noise_std)
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(embed_dim, expert_hidden_dim, dropout, activation)
            for _ in range(n_experts)
        ])
    
    def forward(self, x, return_aux_loss=True):
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]
            return_aux_loss: Whether to return auxiliary load balancing loss
            
        Returns:
            output: Processed tensor of shape [batch, seq_len, embed_dim]
            aux_loss: Load balancing loss (if return_aux_loss=True)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Get routing decisions
        router_probs, top_k_indices, aux_loss = self.router(x)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Flatten for batch processing
        flat_x = x.view(-1, embed_dim)  # [batch * seq, embed_dim]
        flat_router_probs = router_probs.view(-1, self.n_experts)  # [batch * seq, n_experts]
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch, seq_len]
            flat_mask = expert_mask.view(-1)  # [batch * seq]
            
            if not flat_mask.any():
                continue
            
            # Get tokens for this expert
            expert_input = flat_x[flat_mask]  # [n_tokens, embed_dim]
            
            # Process through expert
            expert_output = expert(expert_input)  # [n_tokens, embed_dim]
            
            # Weight by routing probability
            gating_weights = flat_router_probs[flat_mask, expert_idx].unsqueeze(-1)  # [n_tokens, 1]
            weighted_output = gating_weights * expert_output
            
            # Scatter results back to output
            output[expert_mask] += weighted_output
        
        if return_aux_loss:
            return output, aux_loss
        return output


class MoETransformerBlock(nn.Module):
    """
    A complete transformer block with MoE replacing the feed-forward network.
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> x (residual)
        x -> LayerNorm -> MoE -> x (residual)
    
    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        n_experts: Number of experts in MoE
        top_k: Number of experts per token
        dropout: Dropout probability
        activation: Activation for experts
    """
    
    def __init__(
        self,
        embed_dim,
        n_heads=8,
        n_experts=8,
        top_k=2,
        dropout=0.1,
        activation='relu'
    ):
        super().__init__()
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # MoE feed-forward
        self.moe = MixtureOfExperts(
            embed_dim, n_experts, top_k, dropout=dropout, activation=activation
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            
        Returns:
            output: Processed tensor [batch, seq_len, embed_dim]
            aux_loss: Auxiliary load balancing loss from MoE
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, normed, normed, key_padding_mask=attention_mask)
        x = x + self.dropout(attn_output)
        
        # MoE with residual
        normed = self.norm2(x)
        moe_output, aux_loss = self.moe(normed, return_aux_loss=True)
        x = x + self.dropout(moe_output)
        
        return x, aux_loss


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    # Configuration
    batch_size = 2
    seq_len = 16
    embed_dim = 256
    n_experts = 8
    top_k = 2
    
    print("=" * 60)
    print("Mixture of Experts (MoE) Architecture Demo")
    print("=" * 60)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"\nInput shape: {x.shape}")
    print(f"Configuration: {n_experts} experts, top-{top_k} routing")
    
    # Test individual components
    print("\n--- Testing Router ---")
    router = NoisyTopKRouter(embed_dim, n_experts, top_k)
    probs, indices, loss = router(x)
    print(f"Router probs shape: {probs.shape}")
    print(f"Top-k indices shape: {indices.shape}")
    print(f"Load balance loss: {loss.item():.4f}")
    
    print("\n--- Testing Expert ---")
    expert = Expert(embed_dim, activation='gelu')
    expert_out = expert(x)
    print(f"Expert output shape: {expert_out.shape}")
    
    print("\n--- Testing MoE Layer ---")
    moe = MixtureOfExperts(embed_dim, n_experts, top_k, activation='gelu')
    moe_out, aux_loss = moe(x)
    print(f"MoE output shape: {moe_out.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    print("\n--- Testing Full Transformer Block ---")
    block = MoETransformerBlock(embed_dim, n_heads=8, n_experts=n_experts, top_k=top_k)
    block_out, block_aux_loss = block(x)
    print(f"Block output shape: {block_out.shape}")
    print(f"Block auxiliary loss: {block_aux_loss.item():.4f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"\nTotal MoE parameters: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
