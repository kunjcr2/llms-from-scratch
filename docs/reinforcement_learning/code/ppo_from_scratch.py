import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

# ==========================================
# 1. Hyperparameters
# ==========================================
# These parameters govern how PPO trains.
LR = 3e-4               # Learning rate for optimizer
GAMMA = 0.99            # Discount factor for future rewards
CLIP_RATIO = 0.2        # Epsilon in the PPO-Clip objective to prevent large policy updates
UPDATE_EPOCHS = 10      # Number of times to update networks per collected batch
BATCH_SIZE = 64         # Mini-batch size for training
MAX_EPISODES = 500      # Total episodes to train
MAX_TIMESTEPS = 200     # Max steps per episode (CartPole-v1 max is 500, but we can set 200/500 depending on gym version)

# ==========================================
# 2. Neural Network (Actor-Critic)
# ==========================================
class ActorCritic(nn.Module):
    """
    This network has two heads:
    1. Actor: Outputs the probabilities of each action (policy).
    2. Critic: Outputs the estimated value of the state V(s).
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Shared feature extractor (optional, but common)
        # Here we keep separate networks for simplicity and stability.
        
        # Actor Network: State -> Action Probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(), # Tanh is often preferred in PPO over ReLU for stability
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # Output probability distribution over discrete actions
        )
        
        # Critic Network: State -> Value (Scalar)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError("Use act() or evaluate() instead of forward.")
        
    def act(self, state):
        """
        Samples an action from the policy for the given state.
        Returns the action, its log probability, and the state value.
        """
        action_probs = self.actor(state)
        
        # Manually sample from the probability distribution
        action = torch.multinomial(action_probs, 1).squeeze(-1)
        
        # Calculate log probabilities directly
        log_probs = torch.log(action_probs + 1e-10) # Add small epsilon to prevent log(0)
        
        # Gather the log probability of the sampled action
        if action_probs.dim() == 1:
            action_logprob = log_probs[action]
        else:
            action_logprob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        state_val = self.critic(state)
        
        return action.item(), action_logprob.item(), state_val.item()
    
    def evaluate(self, state, action):
        """
        Evaluates a batch of (state, action) pairs during the PPO update.
        Returns the action log probabilities, state values, and entropy.
        """
        action_probs = self.actor(state)
        
        # Calculate log probabilities directly
        log_probs = torch.log(action_probs + 1e-10)
        
        # Gather the log probabilities of the specified actions
        action_logprobs = log_probs.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        
        # Manually calculate entropy: -sum(p * log(p))
        dist_entropy = -(action_probs * log_probs).sum(dim=-1)
        
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

# ==========================================
# 3. Rollout Buffer
# ==========================================
class RolloutBuffer:
    """
    Stores trajectories (states, actions, rewards, etc.) collected by the policy.
    These are used to compute advantages and perform the PPO update.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

# ==========================================
# 4. PPO Agent
# ==========================================
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        
        # The 'old' policy is used to compute the probability ratio (pi_theta / pi_theta_old)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = RolloutBuffer()
        
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action, action_logprob, state_val = self.policy_old.act(state_tensor)
            
            # Store data in buffer for training
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            
        return action

    def update(self):
        """
        Performs the PPO update using the data collected in the buffer.
        """
        # 1. Convert lists to tensors
        old_states = torch.FloatTensor(np.array(self.buffer.states))
        old_actions = torch.LongTensor(np.array(self.buffer.actions))
        old_logprobs = torch.FloatTensor(np.array(self.buffer.logprobs))
        old_state_values = torch.FloatTensor(np.array(self.buffer.state_values)).squeeze()
        
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        
        # 2. Compute Rewards-To-Go and Standard Advantage (no GAE)
        returns = []
        discounted_reward = 0
        
        # Process rewards backwards to calculate discounted returns
        for step in reversed(range(len(rewards))):
            if is_terminals[step]:
                discounted_reward = 0
            
            discounted_reward = rewards[step] + (GAMMA * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns) # converting to tensors
        
        # Standard Advantage: A(s,a) = G_t - V(s) = Q(a_t, s_t) - V(s_t)
        # Note: We detach old_state_values because they are treated as target baseline constants
        advantages = returns - old_state_values.detach()
        
        # Normalize advantages (standard practice for stability) - GRPO DOES THAT, but okayyyyyy
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 3. Optimize policy for K epochs
        for _ in range(UPDATE_EPOCHS):
            # Evaluate old actions and states using the CURRENT policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # Probability Ratio = pi_theta(a|s) / pi_theta_old(a|s)
            # In log space: exp(log_prob - old_log_prob)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # 4. Compute Actor Loss (PPO-Clip)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * advantages
            
            # Actor loss is negative because we want to MAXIMIZE the objective (gradient ascent)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 5. Compute Critic Loss (Mean Squared Error)
            critic_loss = F.mse_loss(state_values, returns)
            
            # 6. Total Loss = Actor Loss + Critic Loss - Entropy Bonus
            # (Entropy bonus encourages exploration)
            entropy_bonus = 0.01 * dist_entropy.mean()
            loss = actor_loss + critic_loss - entropy_bonus
            
            # 7. Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights to the old policy network
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer after training
        self.buffer.clear()

# ==========================================
# 5. Training Loop
# ==========================================
def main():
    # Create the environment. CartPole is simple and fast to train.
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim)
    
    print_freq = 20 # Print average reward every 20 episodes
    running_reward = 0
    
    time_step = 0
    update_timestep = 2000 # Update network after this many timesteps collected
    
    for episode in range(1, MAX_EPISODES + 1):
        # Reset environment (gymnasium > 0.26 API)
        state, _ = env.reset()
            
        ep_reward = 0
        
        for t in range(MAX_TIMESTEPS):
            time_step += 1
            
            # 1. Select action
            action = agent.select_action(state)
            
            # 2. Step environment (gymnasium > 0.26 API)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
                
            # 3. Store reward and termination status
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            state = next_state
            ep_reward += reward
            
            # 4. Train Agent
            # If we've collected enough data, perform PPO update
            if time_step % update_timestep == 0:
                agent.update()
                
            if done:
                break
                
        running_reward += ep_reward
        
        # Logging
        if episode % print_freq == 0:
            avg_reward = running_reward / print_freq
            print(f"Episode {episode}\t Average Reward: {avg_reward:.2f}")
            running_reward = 0
            
            # Simple early stopping condition for CartPole
            if avg_reward > 195.0:
                print("Solved CartPole!")
                break
                
    env.close()

if __name__ == '__main__':
    main()
