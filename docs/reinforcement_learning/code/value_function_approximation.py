"""
Value Function Approximation (VFA)

This script demonstrates Episodic Semi-Gradient SARSA using Non-Linear Function Approximation (Neural Networks).
Instead of a tabular Q-table or a linear weights matrix, we use a simple Multi-Layer Perceptron (MLP)
via PyTorch to approximate Q-values for a continuous state space.

Core ideas demonstrated (as per the notes):
1. Non-linear Approximation: Q(s, a) = NeuralNetwork(s)[a]
2. Feature Learning: The hidden layers act as non-linear feature extractors.
3. Semi-Gradient SARSA: Updating the Neural Network weights based on TD error using Backpropagation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# 1D CONTINUOUS ENVIRONMENT
# ============================================================================
# State: A continuous number between 0.0 and 1.0
# Start State: 0.5
# Actions: 0 (Left), 1 (Right)
# Step size: 0.1 + some random noise
# Terminal states: x <= 0.0 (Reward = -1), x >= 1.0 (Reward = +1)
# Step reward: -0.01 per step to encourage the fastest path
# 
# The optimal action everywhere should just be to go Right!

def step(state, action):
    noise = np.random.normal(0, 0.02)
    step_size = -0.1 if action == 0 else 0.1
    next_state = state + step_size + noise
    
    if next_state <= 0.0:
        return next_state, -1.0, True
    elif next_state >= 1.0:
        return next_state, 1.0, True
    
    return next_state, -0.01, False 

# ============================================================================
# NON-LINEAR FUNCTION APPROXIMATOR (NEURAL NETWORK)
# ============================================================================

class NeuralQFunction(nn.Module):
    """
    Approximates Q(s,a) using a simple MLP in PyTorch.
    Input: [1] (the state x)
    Output: [num_actions] (Q-values for each action)
    """
    def __init__(self, num_actions=2):
        super().__init__()
        # Simple feed-forward network (Non-linear!)
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
        
    def forward(self, state):
        # Convert state scalar to tensor of shape (1, 1) if it isn't already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor([[state]])
        return self.net(state)
        
    def get_q_values(self, state):
        """Helper to get numpy Q-values for decision making during interaction."""
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.numpy()[0]

# ============================================================================
# EPISODIC SEMI-GRADIENT SARSA
# ============================================================================

def epsilon_greedy(q_func, state, epsilon):
    """Epsilon-greedy policy for action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(2)
    q_values = q_func.get_q_values(state)
    return int(np.argmax(q_values))

def train_sarsa_vfa(episodes=1500, lr=0.01, gamma=0.99, epsilon=0.1):
    q_func = NeuralQFunction(num_actions=2)
    optimizer = optim.Adam(q_func.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    print(f"Training Semi-Gradient SARSA (Neural Network) for {episodes} episodes...")
    
    for ep in range(episodes):
        state = 0.5 # Start in the middle
        action = epsilon_greedy(q_func, state, epsilon)
        
        while True:
            next_state, reward, done = step(state, action)
            
            # --- Calculate TD Target ---
            if done:
                td_target = torch.FloatTensor([[reward]])
            else:
                next_action = epsilon_greedy(q_func, next_state, epsilon)
                
                # We use torch.no_grad() to compute the target because it is 
                # a SEMI-gradient method - we treat the target as a constant!
                with torch.no_grad():
                    next_q = q_func.forward(next_state)[0, next_action]
                    td_target = reward + gamma * next_q
                td_target = td_target.view(1, 1) # Match shapes
                
            # --- Calculate Current Q-Value ---
            # Forward pass to get Q_hat(S, A, w)
            # We want gradients to flow through this value!
            current_q = q_func.forward(state)[0, action].view(1, 1)
            
            # --- Network Update ---
            # td_error = (td_target - current_q)^2 for MSE
            loss = loss_fn(current_q, td_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if done:
                break
                
            state = next_state
            action = next_action
            
        if (ep + 1) % 250 == 0:
            print(f"Episode {ep+1}/{episodes} completed.")
            
    print("\n--- Training Complete ---")
    print("\nEvaluating the learned policy across the continuous state space:")
    print("-" * 65)
    print(f"{'State':<10} | {'Q(s, Left)':<15} | {'Q(s, Right)':<15} | {'Action Choice'}")
    print("-" * 65)
    
    for test_state in np.linspace(0.1, 0.9, 9):
        q_values = q_func.get_q_values(test_state)
        best_action = np.argmax(q_values)
        action_str = "Right (->)" if best_action == 1 else "Left (<-)"
        
        print(f"x = {test_state:.1f}  | {q_values[0]:<15.3f} | {q_values[1]:<15.3f} | {action_str}")
        
    return q_func

if __name__ == "__main__":
    print("=== Value Function Approximation (VFA) with Non-Linear/Deep Methods ===")
    train_sarsa_vfa(episodes=500, lr=0.01, gamma=0.95, epsilon=0.1)
