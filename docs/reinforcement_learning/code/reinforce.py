"""
REINFORCE Algorithm on CartPole-v1

CartPole: keep a pole balanced on a moving cart.
State: 4 numbers (cart position, cart velocity, pole angle, pole velocity)
Actions: push cart LEFT (0) or RIGHT (1)
Reward: +1 for every step the pole stays up (max 500)
Done: pole falls over, or cart goes out of bounds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ─── Policy Network ────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Simple network: takes state, outputs action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)   # Probabilities that sum to 1
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def select_action(self, state) -> tuple[int, torch.Tensor]:
        """Pick an action by sampling from the policy distribution."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_tensor)

        # Sample action proportionally to probabilities
        action = torch.multinomial(probs, num_samples=1).item()

        # Log probability of the chosen action (needed for the REINFORCE update)
        log_prob = torch.log(probs[0, action].clamp(min=1e-8))

        return action, log_prob


# ─── REINFORCE Agent ───────────────────────────────────────────────────────────

class REINFORCE:
    """REINFORCE policy gradient agent."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001):
        self.gamma = 0.99  # How much to value future rewards
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train_one_episode(self, env) -> float:
        state, _ = env.reset()
        log_probs = []
        rewards = []

        # Step 1: Play one full episode
        done = False
        while not done:
            action, log_prob = self.policy.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

        # Step 2: Calculate discounted returns
        # G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Step 3: Normalize returns (helps stabilize training in CartPole
        # because episode lengths vary a lot — short episodes vs long ones)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Step 4: Compute loss: -log_prob * return (negative = maximize reward)
        loss = sum(-lp * G_t for lp, G_t in zip(log_probs, returns))

        # Step 5: Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return sum(rewards)  # Total reward = how long the pole stayed up


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    state_dim  = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    agent = REINFORCE(state_dim=state_dim, action_dim=action_dim, lr=0.001)

    print("Training REINFORCE on CartPole-v1...")
    print("Goal: keep the pole balanced for 500 steps")
    print("Reward per episode = number of steps pole stayed up (max 500)")
    print("-" * 50)

    for episode in range(1000):
        total_reward = agent.train_one_episode(env)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1:4d} | Steps survived: {total_reward:.0f}")

    print("-" * 50)

    # ─── Test the trained agent ────────────────────────────────────────────────
    print("Testing trained agent (greedy — always picks best action):")
    print("-" * 50)

    test_env = gym.make("CartPole-v1")
    state, _ = test_env.reset()
    done = False
    step = 0

    while not done:
        step += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = agent.policy.forward(state_tensor).detach()
        action = int(torch.argmax(probs).item())   # always pick best action

        state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    print(f"Pole stayed up for {step} steps  (max possible: 500)")
    if step >= 475:
        print("✓ Solved! Agent learned to balance the pole.")
    elif step >= 200:
        print("~ Partially learned.")
    else:
        print("✗ Needs more training.")

    env.close()
    test_env.close()