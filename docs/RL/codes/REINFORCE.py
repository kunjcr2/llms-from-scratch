"""
REINFORCE Algorithm Implementation

A simple policy gradient algorithm using Monte Carlo returns.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Simple MLP policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """Select action and return log probability."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_tensor)
        dist = Categorical(probs) # just like torch.multinomial
        action = dist.sample() # <- Sampling
        return action.item(), dist.log_prob(action) # <- Log probability


class REINFORCE:
    """REINFORCE with optional baseline."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True
    ):
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Simple baseline: running mean of returns
        self.baseline = 0.0
        self.baseline_alpha = 0.1
    
    def compute_returns(self, rewards: list[float]) -> torch.Tensor:
        """Compute discounted returns G_t for each timestep."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)
    
    def update(self, log_probs: list[torch.Tensor], rewards: list[float]) -> float:
        """Update policy using collected trajectory."""
        returns = self.compute_returns(rewards)
        
        # Normalize returns (optional but helps stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Apply baseline if enabled
        if self.use_baseline:
            advantages = returns - self.baseline
            self.baseline = (1 - self.baseline_alpha) * self.baseline + \
                           self.baseline_alpha * returns.mean().item()
        else:
            advantages = returns
        
        # Policy gradient: -log_prob * advantage (negative for gradient ascent)
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def train(
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    print_every: int = 50,
    use_baseline: bool = True
):
    """Train REINFORCE on a gym environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(
        state_dim=state_dim,
        action_dim=action_dim,
        use_baseline=use_baseline
    )
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        # Collect trajectory
        while not done:
            action, log_prob = agent.policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Update policy after episode ends
        loss = agent.update(log_probs, rewards)
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode + 1:4d} | Avg Reward: {avg_reward:6.1f} | Loss: {loss:.4f}")
    
    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    print("Training REINFORCE on CartPole-v1...")
    print("-" * 50)
    agent, rewards = train(use_baseline=True)
    print("-" * 50)
    print(f"Final average reward (last 50 episodes): {np.mean(rewards[-50:]):.1f}")
