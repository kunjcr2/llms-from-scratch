"""
Group Relative Policy Optimization (GRPO) Implementation

A simplified implementation demonstrating GRPO concepts for LLM training.
This is a conceptual example - real implementations use HuggingFace TRL or similar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    group_size: int = 4           # Number of responses per prompt
    clip_epsilon: float = 0.2     # PPO clipping parameter
    kl_coef: float = 0.1          # KL divergence penalty coefficient
    max_length: int = 512         # Maximum response length


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute advantages using group-based normalization.
    
    This is GRPO's key innovation: instead of using a learned value function,
    normalize rewards within each group of responses.
    
    Args:
        rewards: Shape (batch_size, group_size) - rewards for each response
    
    Returns:
        advantages: Shape (batch_size, group_size) - normalized advantages
    """
    # Normalize within each group (across responses to same prompt)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True) + 1e-8
    
    advantages = (rewards - mean) / std
    return advantages


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    """
    Compute PPO clipped surrogate loss.
    
    Args:
        log_probs: Log probabilities under current policy
        old_log_probs: Log probabilities under old policy
        advantages: Advantage estimates
        clip_epsilon: Clipping parameter
    
    Returns:
        PPO loss (to be maximized, so negate for gradient descent)
    """
    # Compute probability ratio
    # Shouldnt it be just log_probs/old_log_probs?
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped and unclipped objectives
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    
    # Take minimum
    loss = torch.min(unclipped, clipped)
    
    return loss.mean()


def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between current policy and reference policy.
    
    Args:
        log_probs: Log probabilities under current policy
        ref_log_probs: Log probabilities under reference (base) policy
    
    Returns:
        KL divergence estimate
    """
    # KL(π_θ || π_ref) ≈ E[log π_θ - log π_ref]
    kl = (torch.exp(log_probs) * (log_probs - ref_log_probs)).sum(dim=-1)
    return kl.mean()


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    config: GRPOConfig
) -> tuple[torch.Tensor, dict]:
    """
    Compute complete GRPO loss.
    
    L_GRPO = L_PPO - β * D_KL(π_θ || π_ref)
    
    Args:
        log_probs: Current policy log probs, shape (batch, group_size, seq_len)
        old_log_probs: Old policy log probs (for PPO ratio)
        ref_log_probs: Reference (base model) log probs
        rewards: Rewards for each response, shape (batch, group_size)
        config: GRPO configuration
    
    Returns:
        loss: Scalar loss to minimize
        metrics: Dictionary of training metrics
    """
    # Step 1: Compute group-based advantages (GRPO's key innovation)
    advantages = compute_group_advantages(rewards)
    
    # Expand advantages to match sequence dimension
    # Shape: (batch, group_size) -> (batch, group_size, 1)
    advantages_expanded = advantages.unsqueeze(-1)
    
    # Step 2: Compute PPO clipped loss
    ppo_loss = compute_ppo_loss(
        log_probs.sum(dim=-1),  # Sum log probs over sequence
        old_log_probs.sum(dim=-1),
        advantages,
        config.clip_epsilon
    )
    
    # Step 3: Compute KL penalty (keeps policy close to reference)
    kl_div = compute_kl_divergence(log_probs, ref_log_probs)
    
    # Step 4: Combine losses
    # Negate PPO loss because we want to maximize it
    total_loss = -ppo_loss + config.kl_coef * kl_div
    
    metrics = {
        "ppo_loss": ppo_loss.item(),
        "kl_divergence": kl_div.item(),
        "total_loss": total_loss.item(),
        "mean_advantage": advantages.mean().item(),
    }
    
    return total_loss, metrics


# ============================================================================
# Example: Simplified GRPO Training Loop for LLMs
# ============================================================================

class SimplifiedGRPOTrainer:
    """
    Simplified GRPO trainer demonstrating the core algorithm.
    
    In practice, use HuggingFace TRL's GRPOTrainer:
    https://huggingface.co/docs/trl/main/en/grpo_trainer
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: GRPOConfig,
        lr: float = 1e-6
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
    
    def generate_responses(
        self,
        prompts: list[str],
        tokenizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses per prompt.
        
        Returns:
            responses: Generated token IDs
            log_probs: Log probabilities of generated tokens
        """
        # In real implementation, this would use model.generate()
        # with proper sampling and return log probabilities
        pass
    
    def compute_rewards(
        self,
        prompts: list[str],
        responses: list[str]
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        
        For reasoning tasks, this could be:
        - Correctness check (1 if correct, 0 if wrong)
        - Format reward (bonus for proper reasoning steps)
        - Length penalty
        """
        # Example: Simple correctness reward
        # In practice, use a reward model or ground truth verification
        pass
    
    def train_step(
        self,
        prompts: list[str],
        tokenizer
    ) -> dict:
        """
        Single GRPO training step.
        
        1. Generate G responses per prompt
        2. Compute rewards
        3. Compute group-normalized advantages
        4. Update policy with PPO + KL penalty
        """
        batch_size = len(prompts)
        G = self.config.group_size
        
        # Step 1: Generate responses (G per prompt)
        self.model.eval()
        with torch.no_grad():
            # Generate responses and get log probs
            responses, old_log_probs = self.generate_responses(prompts, tokenizer)
            
            # Get reference model log probs
            ref_log_probs = self.get_log_probs(self.ref_model, responses)
        
        # Step 2: Compute rewards
        rewards = self.compute_rewards(prompts, responses)
        # Shape: (batch_size, group_size)
        
        # Step 3: Training update
        self.model.train()
        
        # Get current log probs (with gradients)
        log_probs = self.get_log_probs(self.model, responses)
        
        # Compute GRPO loss
        loss, metrics = grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
            config=self.config
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return metrics
    
    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Get log probabilities for input tokens."""
        outputs = model(input_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        # Shape: (batch, seq_len)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("GRPO Conceptual Example")
    print("=" * 50)
    
    # Demonstrate group advantage computation
    print("\n1. Group-Based Advantage Computation:")
    print("-" * 40)
    
    # Simulate rewards for 2 prompts, 4 responses each
    rewards = torch.tensor([
        [1.0, 0.8, 0.0, 0.0],  # Prompt 1: 2 correct, 2 wrong
        [0.5, 0.5, 0.5, 1.0],  # Prompt 2: 3 mediocre, 1 good
    ])
    
    print(f"Rewards:\n{rewards}")
    
    advantages = compute_group_advantages(rewards)
    print(f"\nNormalized Advantages:\n{advantages}")
    
    # Demonstrate PPO clipping
    print("\n2. PPO Clipping Example:")
    print("-" * 40)
    
    # Simulate log probabilities
    old_log_probs = torch.tensor([-0.5, -0.8, -1.2, -0.3])
    new_log_probs = torch.tensor([-0.3, -0.6, -1.0, -0.5])  # Policy has changed
    advantages_flat = torch.tensor([1.0, -0.5, 0.2, 0.8])
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    print(f"Probability ratios: {ratio}")
    
    clip_eps = 0.2
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    print(f"Clipped ratios:     {clipped_ratio}")
    
    loss = compute_ppo_loss(new_log_probs, old_log_probs, advantages_flat, clip_eps)
    print(f"\nPPO Loss: {loss:.4f}")
    
    print("\n" + "=" * 50)
    print("For real LLM training, use HuggingFace TRL:")
    print("  from trl import GRPOTrainer, GRPOConfig")
    print("  trainer = GRPOTrainer(model, config, ...)")
