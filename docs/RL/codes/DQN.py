"""
Deep Q-Network (DQN) for Pong with PyTorch

Train an agent to play Pong from raw pixels using CNNs.
Based on DeepMind's 2015 Nature paper.

Key components:
1. CNN: Extract features from stacked game frames
2. Experience Replay: Random sampling breaks correlations
3. Target Network: Stable TD targets
4. Frame Stacking: 4 frames capture motion/velocity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class PongCNN(nn.Module):
    """
    Convolutional neural network for Pong Q-values.
    
    Architecture (DeepMind 2015):
    Input: 84x84x4 (4 stacked grayscale frames)
    → Conv 32 filters (8x8, stride 4) → ReLU
    → Conv 64 filters (4x4, stride 2) → ReLU  
    → Conv 64 filters (3x3, stride 1) → ReLU
    → FC 512 → ReLU
    → Output: 3 Q-values (UP, DOWN, STAY)
    
    Why this architecture?
    - First conv: Large receptive field, detects edges/basic shapes
    - Second conv: Combines features, finds paddles/ball
    - Third conv: High-level patterns, game state understanding
    - FC layers: Combine spatial info into Q-values
    """
    
    def __init__(self, num_actions=3):
        super(PongCNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 84x84x4 → 20x20x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 20x20x32 → 9x9x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 9x9x64 → 7x7x64
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        """
        Forward pass: frames → Q-values
        
        Args:
            x: (batch, 4, 84, 84) tensor of stacked frames
        Returns:
            (batch, num_actions) Q-values
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class ReplayBuffer:
    """Store and sample experience tuples."""
    
    def __init__(self, capacity=100000):
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
    """DQN agent for playing Pong."""
    
    def __init__(self, num_actions=3, lr=0.00025, gamma=0.99, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.num_actions = num_actions
        
        # Q-network (the one we train)
        self.q_net = PongCNN(num_actions).to(device)
        
        # Target network (frozen copy for stable targets)
        self.target_net = PongCNN(num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Always in eval mode
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
    
    def get_action(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self, batch):
        """
        One gradient descent step.
        
        Loss = (Q(s,a) - TD_target)²
        Where TD_target = r + γ * max_a' Q_target(s', a')
        
        Why target network?
        - TD_target uses target_net (updated every N steps)
        - This prevents "chasing a moving target"
        - Greatly stabilizes training
        """
        states, actions, rewards, next_states, dones = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values: Q(s, a)
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


def preprocess_frame(frame):
    """
    Preprocess Atari frame: 210x160x3 → 84x84 grayscale.
    
    Steps: RGB→gray, crop score, resize, normalize
    """
    gray = np.mean(frame, axis=2).astype(np.uint8)
    cropped = gray[34:194, :]  # Remove score
    # In practice use: cv2.resize(cropped, (84, 84))
    resized = cropped[::2, ::2]  # Simple downsample for demo
    return resized / 255.0


def train_pong():
    """
    Training loop for Pong DQN.
    
    Hyperparameters (from DeepMind paper):
    - Replay buffer: 100k transitions
    - Batch size: 32
    - Learning rate: 0.00025
    - Gamma: 0.99
    - Epsilon: 1.0 → 0.1 over 1M frames
    - Target network update: every 10k steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(num_actions=3, device=device)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Training params
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 1000000  # Decay over 1M frames
    batch_size = 32
    target_update_freq = 10000
    
    print(f"Training Pong DQN on {device}")
    print("-" * 60)
    
    frame_count = 0
    episode = 0
    
    # Simulate training (replace with actual gym environment)
    for step in range(1000):
        # Epsilon decay
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-frame_count / epsilon_decay)
        
        # Simulate state (4 stacked frames: 4x84x84)
        state = np.random.rand(4, 84, 84)
        action = agent.get_action(state, epsilon)
        
        # Simulate environment step
        reward = np.random.choice([-1, 0, 1])  # Pong rewards
        next_state = np.random.rand(4, 84, 84)
        done = random.random() < 0.01
        
        replay_buffer.add(state, action, reward, next_state, done)
        frame_count += 1
        
        # Train when enough samples
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = agent.train_step(batch)
            
            if step % 100 == 0:
                print(f"Step {step} | Epsilon {epsilon:.3f} | Loss {loss:.4f}")
        
        # Update target network
        if frame_count % target_update_freq == 0:
            agent.update_target_network()
            print(f"→ Target network updated at frame {frame_count}")
        
        if done:
            episode += 1
    
    print(f"\nTraining complete: {episode} episodes, {frame_count} frames")


if __name__ == "__main__":
    train_pong()
