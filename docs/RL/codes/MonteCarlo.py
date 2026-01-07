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
# EXAMPLE USAGE
# ============================================================================

def monte_carlo_example():
    """
    4x4 grid: Start at (0,0), Goal at (3,3)
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal
    """
    # States 0-15 (flattened 4x4 grid), 4 actions
    Q = np.zeros((16, 4))
    
    # One complete episode: [(state, action, reward), ...]
    episode = [(0,1,-1), (1,1,-1), (2,2,-1), (6,2,-1), (10,1,-1), (11,2,-1), (15,0,10)]
    Q = monte_carlo_update(episode, Q, gamma=0.9)
    
    print("Monte Carlo learned Q(0,1):", Q[0,1])
    print("This represents the expected return from state 0, taking action 'right'")


if __name__ == "__main__":
    monte_carlo_example()
