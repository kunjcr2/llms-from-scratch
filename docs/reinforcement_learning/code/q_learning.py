r"""
Q-Learning (Off-Policy TD Control) Implementation

Q-Learning: Learn the optimal policy off-policy.
- Q(S, A): Expected cumulative reward for action A in state S
- Goal: Learn Q-values to find the optimal policy regardless of the agent's exploratory actions.

Core idea: Q(S, A) is updated using the maximum possible Q-value of the next state,
meaning it assumes the agent will take the *best* action next, even if it actually exploring.

Update rule:
    Q(S, A) ← Q(S, A) + α * [ R + γ * max_a Q(S', a) - Q(S, A) ]
                             \_______ TD Target _______/
"""

import numpy as np

# ============================================================================
# CLIFF WALKING ENVIRONMENT (4x12 Grid)
# ============================================================================
# A classic environment to show how Q-Learning differs from SARSA.
# Start: Bottom-Left (3, 0)
# Goal: Bottom-Right (3, 11)
# Cliff: Bottom row, columns 1 to 10. Stepping here = -100 reward and reset to Start.
# Normal step: -1 reward.
# Actions: 0=up, 1=right, 2=down, 3=left

ROWS = 4
COLS = 12
START_STATE = 3 * COLS + 0     # 36
GOAL_STATE = 3 * COLS + 11     # 47
CLIFF_STATES = set(range(START_STATE+1, GOAL_STATE)) # START_STATE+1 to GOAL_STATE-1

def step(state, action):
    row, col = divmod(state, COLS)

    if action == 0:    row = max(row - 1, 0)         # up
    elif action == 1:  col = min(col + 1, COLS - 1)  # right
    elif action == 2:  row = min(row + 1, ROWS - 1)  # down
    elif action == 3:  col = max(col - 1, 0)         # left

    next_state = row * COLS + col

    if next_state in CLIFF_STATES:
        return START_STATE, -100, False  # Fall off cliff: -100 reward, go back to start
    
    if next_state == GOAL_STATE:
        return next_state, -1, True      # Reach goal: -1 reward, episode ends

    return next_state, -1, False         # Normal step: -1 reward, episode continues


def epsilon_greedy(Q, state, epsilon):
    """Pick action epsilon-greedily."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))


# ============================================================================
# Q-LEARNING ALGORITHM
# ============================================================================

def q_learning_update(Q, state, action, reward, next_state, alpha, gamma, done):
    """
    One Q-Learning update step.

    Q-learning update rule:
        Q(S, A) ← Q(S, A) + α * [ R + γ * max_a Q(S', a) - Q(S, A) ]

    'Off-policy': The TD target uses the maximum Q-value of the next state,
    assuming the greedy (best) action is taken next, even though the actual action 
    we might take next in the loop is exploratory (epsilon-greedy).
    """
    # max_a Q(S', a) is the best future value we can hope for from the next state
    best_next_value = np.max(Q[next_state]) # main point that makes it different
    
    td_target = reward + gamma * best_next_value * (not done)
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    
    return Q


def q_learning_example(num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Runs Q-Learning on the Cliff Walking environment.
    """
    Q = np.zeros((ROWS * COLS, 4))
    rewards_per_episode = []

    print("\nUnlearned Policy (0=↑, 1=→, 2=↓, 3=←):")
    action_chars = ['↑', '→', '↓', '←']
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            s = r * COLS + c
            if s == START_STATE:
                row_str += " S "
            elif s == GOAL_STATE:
                row_str += " G "
            elif s in CLIFF_STATES:
                row_str += " C "
            else:
                best_action = int(np.argmax(Q[s]))
                row_str += f" {action_chars[best_action]} "
        print(row_str)
    print()

    for ep in range(num_episodes):
        state = START_STATE
        total_reward = 0

        while True:
            # 1. Choose A from S using policy derived from Q (epsilon-greedy)
            action = epsilon_greedy(Q, state, epsilon)
            
            # 2. Take action A, observe R, S'
            next_state, reward, done = step(state, action)
            
            # 3. Update Q(S,A) using Q-Learning rule
            Q = q_learning_update(Q, state, action, reward, next_state, alpha, gamma, done)
            
            state = next_state
            total_reward += reward

            if done:
                break
                
        rewards_per_episode.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {ep+1:4d} | avg reward (last 100): {avg_reward:.1f}")

    print("\n--- Training complete ---")
    
    # Print the learned optimal policy visually
    print("\nLearned Optimal Policy (0=↑, 1=→, 2=↓, 3=←):")
    action_chars = ['↑', '→', '↓', '←']
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            s = r * COLS + c
            if s == START_STATE:
                row_str += " S "
            elif s == GOAL_STATE:
                row_str += " G "
            elif s in CLIFF_STATES:
                row_str += " C "
            else:
                best_action = int(np.argmax(Q[s]))
                row_str += f" {action_chars[best_action]} "
        print(row_str)

    return Q


if __name__ == "__main__":
    print("=== Q-Learning on Cliff Walking (4x12) ===")
    q_learning_example(num_episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1)
