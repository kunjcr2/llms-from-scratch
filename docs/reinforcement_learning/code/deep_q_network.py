"""
Deep Q-Network (DQN) for CartPole with PyTorch

Train an agent to balance a pole on a cart using a simple Multi-Layer Perceptron (MLP).
This is a simpler, easier-to-run alternative to Atari Pong. It runs on a 1D state array
instead of image pixels, making it much easier to understand how DQN works.

Prerequisites:
    pip install torch numpy gymnasium
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

try:
    import gymnasium as gym
except ImportError:
    print("Please install gymnasium: pip install gymnasium")
    raise

class DQN(nn.Module):
    """
    Standard fully connected neural network (MLP) for CartPole.
    
    Input: state space dimension (4 for CartPole: position, velocity, angle, angular velocity)
    Output: action space dimension (2 for CartPole: Left, Right)
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """Forward pass: state → Q-values"""
        return self.fc(x)

class ReplayBuffer:
    """Store and sample experience tuples to break temporal correlations."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN agent for CartPole."""
    
    def __init__(self, state_dim, num_actions, lr=1e-3, gamma=0.99, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.num_actions = num_actions
        
        # Q-network (the one we train)
        self.q_net = DQN(state_dim, num_actions).to(device)
        
        # Target network (frozen copy for stable TD targets)
        self.target_net = DQN(state_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def get_action(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self, batch):
        """One gradient descent step."""
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values: Q(s, a). We use gather to pick the Q-value of the action taken.
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and backprop
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

def train_cartpole():
    """Training loop for CartPole DQN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CartPole DQN on {device}")
    print("-" * 60)
    
    # Create CartPole environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    num_actions = env.action_space.n            # 2
    
    agent = DQNAgent(state_dim, num_actions, device=device)
    replay_buffer = ReplayBuffer(capacity=10000)
    
    episodes = 500
    batch_size = 64
    target_update_freq = 10
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.get_action(state, epsilon)
            
            # Step environment
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            # Add to memory
            is_terminal = done or truncated
            replay_buffer.add(state, action, reward, next_state, is_terminal)
            
            state = next_state
            
            # Train model
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.train_step(batch)
        
        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Target network update
        if episode % target_update_freq == 0:
            agent.update_target_network()
            
        if episode % 10 == 0:
            print(f"Episode: {episode:3d} | Reward: {episode_reward:5.1f} | Epsilon: {epsilon:.3f}")
            
        if episode_reward >= 450:
            print(f"\nSolved! Top performance reached at episode {episode} with reward {episode_reward}")
            break
            
    env.close()
    print("\nTraining complete! To see it play, you can run an evaluation loop with:")
    print("env = gym.make('CartPole-v1', render_mode='human')")

if __name__ == "__main__":
    train_cartpole()
