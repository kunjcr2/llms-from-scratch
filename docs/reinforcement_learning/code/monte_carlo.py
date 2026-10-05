"""
Monte Carlo Methods for Reinforcement Learning

Monte Carlo: Wait for episode to finish, learn from actual outcomes.
Update Q-values from complete episode using actual returns.

How it works:
1. Episode finishes (e.g., game ends)
2. Calculate return G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
   (gamma discounts future rewards: γ=0.9 means future worth 90% of now)
3. Update Q(s,a) = average of all returns seen from that (s,a)

Pros: Simple, unbiased (uses real outcomes)
Cons: Must wait for complete episodes
"""

import numpy as np


def monte_carlo_update(episode, Q, gamma=0.99):
    """
    Update Q-values from complete episode using actual returns.
    
    Args:
        episode: [(state, action, reward), ...] from one complete episode
        Q: Q-table to update
        gamma: Discount factor (0 to 1, typically 0.9-0.99)
    """
    returns_sum = {}  # Accumulated returns for each (state, action)
    visit_count = {}  # How many times we've seen each (state, action)
    G = 0  # Return (cumulative discounted reward)

    # Work backwards through episode to calculate returns
    for state, action, reward in reversed(episode):
        G = reward + gamma * G  # Add current reward to discounted future
        
        sa_pair = (state, action)
        if sa_pair not in visit_count:
            returns_sum[sa_pair] = 0
            visit_count[sa_pair] = 0
        
        returns_sum[sa_pair] += G
        visit_count[sa_pair] += 1
        
        # Q-value = average return from this state-action pair
        Q[state, action] = returns_sum[sa_pair] / visit_count[sa_pair]
    
    return Q


# ============================================================================
# GRID WORLD SIMULATOR
# ============================================================================

def step(state, action, grid_size=4):
    """
    Take one step in a 4x4 grid world.
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal (state 15), episode ends at goal.
    Returns: (next_state, reward, done)
    """
    row, col = divmod(state, grid_size)

    if action == 0:    # up
        row = max(row - 1, 0)
    elif action == 1:  # right
        col = min(col + 1, grid_size - 1)
    elif action == 2:  # down
        row = min(row + 1, grid_size - 1)
    elif action == 3:  # left
        col = max(col - 1, 0)

    next_state = row * grid_size + col
    done = (next_state == grid_size * grid_size - 1)  # goal = state 15
    reward = 10 if done else -1
    return next_state, reward, done


def generate_episode(Q, epsilon=0.1, max_steps=200, grid_size=4):
    """
    Run one episode using an epsilon-greedy policy derived from Q.
    Returns: [(state, action, reward), ...]
    """
    state = 0  # always start at top-left corner
    episode = []

    for _ in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(Q.shape[1])   # explore
        else:
            action = np.argmax(Q[state])              # exploit

        next_state, reward, done = step(state, action, grid_size)
        episode.append((state, action, reward))
        state = next_state

        if done:
            break

    return episode


# ============================================================================
# EXAMPLE USAGE  –  multiple episodes
# ============================================================================

def monte_carlo_example(num_episodes=500, gamma=0.9, epsilon=0.2):
    """
    4x4 grid: Start at (0,0), Goal at (3,3)
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal

    Runs num_episodes episodes:
      - Generate episode using epsilon-greedy policy
      - Update Q-table with monte_carlo_update
    """
    # Our Q-table is initialized to zeros
    Q = np.zeros((16, 4))

    episode_lengths = []

    for ep in range(num_episodes):
        episode = generate_episode(Q, epsilon=epsilon)
        Q = monte_carlo_update(episode, Q, gamma=gamma)
        episode_lengths.append(len(episode))

        # Print progress every 100 episodes
        if (ep + 1) % 100 == 0:
            avg_len = np.mean(episode_lengths[-100:])
            print(f"Episode {ep+1:4d} | avg steps (last 100): {avg_len:.1f}")

    print("\n--- Training complete ---")
    print(f"Q(state=0, right):  {Q[0, 1]:.3f}   (expected return: go right from start)")
    print(f"Q(state=14, right): {Q[14, 1]:.3f}   (expected return: one step from goal)")
    print(f"\nBest action per state (0=up, 1=right, 2=down, 3=left):")
    action_names = ["up", "right", "down", "left"]
    for s in range(16):
        row, col = divmod(s, 4)
        best = np.argmax(Q[s])
        print(f"  state ({row},{col}): {action_names[best]}")


if __name__ == "__main__":
    monte_carlo_example(num_episodes=1000)
