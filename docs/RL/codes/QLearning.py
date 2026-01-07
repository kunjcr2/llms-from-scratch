"""
Q-Learning (Temporal Difference) Implementation

Q-Learning: Learn immediately from each step using estimates.
- Q(s,a): Expected cumulative reward for action a in state s
- Goal: Learn Q-values to find best policy (action selection strategy)

Core idea: Q(s,a) should equal reward + discounted best future Q-value
Update rule: Q(s,a) ← Q(s,a) + α * [TD_target - Q(s,a)]
                                     └── TD_error ──┘

Where TD_target = r + γ * max_a' Q(s', a')  if not done
                = r                          if done

Pros: Learns immediately, works with continuing tasks, fast
Cons: Uses estimates (bootstrapping), can be unstable
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon, num_actions):
    """
    Balance exploration (try new things) vs exploitation (use best known action).
    
    With probability epsilon: random action (explore)
    Otherwise: action with max Q-value (exploit)
    """
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q[state])


def q_learning_update(state, action, reward, next_state, Q, 
                      alpha=0.1, gamma=0.99, done=False):
    """
    Update Q-value immediately using Temporal Difference learning.
    
    Args:
        state, action: What you did
        reward: Immediate reward received
        next_state: Where you ended up
        Q: Q-table to update
        alpha: Learning rate (0 to 1, typically 0.01-0.5)
        gamma: Discount factor
        done: True if episode ended (no next state)
    """
    if done:
        td_target = reward  # No future rewards
    else:
        # Best possible future value from next_state
        td_target = reward + gamma * np.max(Q[next_state])
    
    # How wrong was our Q-value?
    td_error = td_target - Q[state, action]
    
    # Move Q-value toward target
    Q[state, action] += alpha * td_error
    
    return Q


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def q_learning_example():
    """
    4x4 grid: Start at (0,0), Goal at (3,3)
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal
    """
    # States 0-15 (flattened 4x4 grid), 4 actions
    Q = np.zeros((16, 4))
    
    # Step by step updates: (state, action, reward, next_state, done)
    transitions = [
        (0, 1, -1, 1, False),    # From state 0, go right to state 1
        (1, 1, -1, 2, False),    # From state 1, go right to state 2
        (15, 0, 10, 15, True)    # At goal state 15, episode ends
    ]
    
    for s, a, r, s_next, done in transitions:
        Q = q_learning_update(s, a, r, s_next, Q, alpha=0.1, gamma=0.9, done=done)
    
    print("Q-Learning learned Q(0,1):", Q[0,1])
    print("This value is updated incrementally after each step")


if __name__ == "__main__":
    q_learning_example()
