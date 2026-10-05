"""
Temporal Difference (TD) Methods for Reinforcement Learning

TD methods combine Monte Carlo (learn from experience) and Dynamic Programming
(bootstrap from estimates). Unlike Monte Carlo, updates happen after every step —
no need to wait for the episode to end.

TD Update Rule:
    V(S) <- V(S) + alpha [ R + gamma * V(S') - V(S) ]
          - R + gamma * V(S')   : TD Target  (immediate reward + discounted next-state estimate)
          - TD Error       : difference between TD target and current estimate

Implemented here:
    1. TD(0) Prediction  – estimates V(s) for a fixed policy
"""

import numpy as np


# ============================================================================
# CLIFF WALKING ENVIRONMENT (4x12 Grid)
# ============================================================================
# Start: Bottom-Left (3, 0)
# Goal: Bottom-Right (3, 11)
# Cliff: Bottom row, columns 1 to 10. Stepping here = -100 reward and reset to Start.
# Normal step: -1 reward.
# Actions: 0=up, 1=right, 2=down, 3=left

ROWS = 4
COLS = 12
START_STATE = 3 * COLS + 0     # 36
GOAL_STATE = 3 * COLS + 11     # 47
CLIFF_STATES = set(range(START_STATE+1, GOAL_STATE))

def step(state, action):
    row, col = divmod(state, COLS)

    if action == 0:    row = max(row - 1, 0)         # up
    elif action == 1:  col = min(col + 1, COLS - 1)  # right
    elif action == 2:  row = min(row + 1, ROWS - 1)  # down
    elif action == 3:  col = max(col - 1, 0)         # left

    next_state = row * COLS + col

    if next_state in CLIFF_STATES:
        return START_STATE, -100, False  # Fall off cliff
    
    if next_state == GOAL_STATE:
        return next_state, -1, True      # Reach goal

    return next_state, -1, False         # Normal step


# ============================================================================
# TD(0) PREDICTION  –  estimate V(s) under a random policy
# ============================================================================

def td0_prediction(num_episodes=500, alpha=0.1, gamma=0.9, max_steps=200):
    """
    TD(0): estimate the state-value function V(s) for a random policy.

    Update after every single step (online):
        V(S) <- V(S) + alpha [ R + gamma * V(S') - V(S) ]

    Uses a random (uniform) policy to keep the focus on the TD update rule.
    """
    V = np.zeros(ROWS * COLS)   # state-value table

    print("State values (Before) (Start=bottom-left, Goal=bottom-right):")
    print(V.reshape(ROWS, COLS).round(1))
    print()

    for ep in range(num_episodes):
        state = START_STATE
        steps = 0

        while True:
            # Random because we are just evaluating the value functions
            action = np.random.randint(4)                              # random policy
            next_state, reward, done = step(state, action)

            # TD target = immediate reward + discounted next-state value
            td_target = reward + gamma * V[next_state] * (not done)
            td_error  = td_target - V[state]
            V[state] += alpha * td_error                               # TD(0) update

            state = next_state
            steps += 1
            if done or steps >= max_steps:
                break

    return V

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== TD(0) Prediction on Cliff Walking (random policy) ===")
    V = td0_prediction(num_episodes=1000, alpha=0.1, gamma=1.0)
    print("State values (After) (Start=bottom-left, Goal=bottom-right):")
    print(V.reshape(ROWS, COLS).round(1))
