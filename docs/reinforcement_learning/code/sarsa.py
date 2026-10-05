'''
SARSA (TD Control) Implementation

SARSA: On-policy TD Control.
- Q(S, A): Expected cumulative reward for action A in state S
- Goal: Learn Q-values to find the optimal policy while exploring.

Core idea: Q(S, A) is updated using the Q-value of the next state and the actual next action chosen by the current policy.

Update rule:
    Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
                                  |____ TD Target ____|
'''

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


def epsilon_greedy(Q, state, epsilon):
    """Pick action epsilon-greedily."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))


# ============================================================================
# SARSA ALGORITHM
# ============================================================================

def sarsa_update(Q, state, action, reward, next_state, next_action, alpha, gamma, done):
    """
    One SARSA update step.
    'On-policy': the next action A' is sampled from the same policy being improved.
    """
    td_target = reward + gamma * Q[next_state, next_action] * (not done)
    td_error  = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return Q


def sarsa_example(num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
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
        state  = START_STATE
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0

        while True:
            next_state, reward, done = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q = sarsa_update(Q, state, action, reward, next_state, next_action, alpha, gamma, done)

            state  = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {ep+1:4d} | avg reward (last 100): {avg_reward:.1f}")

    print("\n--- Training complete ---")
    
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
    print("=== SARSA on Cliff Walking (4x12) ===")
    sarsa_example(num_episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1)
