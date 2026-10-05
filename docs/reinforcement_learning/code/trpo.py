"""
TRPO on a 5x5 Grid World — built from scratch.

The grid looks like this:

  . . . . .
  . # . # .
  . . . . .
  . # . # .
  S . . . G

S = start (bottom left)
G = goal  (bottom right)
# = wall  (can't walk into)

Agent actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

The agent gets:
  +10  for reaching the goal
  -0.1 for every step (so it learns to be fast)
  -1   for walking into a wall (bounces back)

We train using TRPO. After training, we print what path the agent takes.
"""

import torch
import torch.nn as nn
import numpy as np


# PART 1: THE ENVIRONMENT
# This is your "gym". A simple class that knows the grid rules.


class GridWorld:
    def __init__(self):
        self.size = 5

        # Walls: list of (row, col) cells the agent cannot enter
        self.walls = {(1,1), (1,3), (3,1), (3,3)}

        self.start = (4, 0)   # bottom-left
        self.goal  = (4, 4)   # bottom-right

        self.agent = self.start

    def reset(self):
        self.agent = self.start
        return self._obs()

    def _obs(self):
        # The agent's observation is just its (row, col) as floats.
        # We normalize to [0,1] so the network doesn't see raw grid numbers.
        r, c = self.agent
        return np.array([r / 4.0, c / 4.0], dtype=np.float32)

    def step(self, action):
        r, c = self.agent

        # Figure out where the agent wants to go
        if   action == 0: new_r, new_c = r-1, c   # UP
        elif action == 1: new_r, new_c = r+1, c   # DOWN
        elif action == 2: new_r, new_c = r,   c-1 # LEFT
        elif action == 3: new_r, new_c = r,   c+1 # RIGHT

        # Hit the grid boundary?
        out_of_bounds = not (0 <= new_r < self.size and 0 <= new_c < self.size)

        # Hit a wall?
        hit_wall = (new_r, new_c) in self.walls

        if out_of_bounds or hit_wall:
            # Bounce back — stay where you are, take a penalty
            reward = -1.0
            done = False
        else:
            # Move the agent
            self.agent = (new_r, new_c)
            if self.agent == self.goal:
                reward = 10.0
                done = True
            else:
                reward = -0.1   # small step penalty to encourage speed
                done = False

        return self._obs(), reward, done

    def render(self):
        # Print the grid nicely so you can see what's happening
        symbols = {self.goal: 'G', self.start: 'S'}
        print()
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if (r, c) == self.agent:
                    row_str += "A "
                elif (r, c) in self.walls:
                    row_str += "# "
                elif (r, c) in symbols:
                    row_str += symbols[(r,c)] + " "
                else:
                    row_str += ". "
            print(row_str)
        print()



# PART 2: THE POLICY NETWORK
#
# This is the "brain" of the agent.
# Input:  2 numbers (normalized row, col)
# Output: 4 numbers → probabilities for each action (up/down/left/right)
#
# nn.Softmax at the end makes sure probabilities sum to 1.


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4),        # 4 actions
            nn.Softmax(dim=-1)       # turn raw scores into probabilities
        )

    def forward(self, x):
        return self.net(x)  # returns [prob_up, prob_down, prob_left, prob_right]



# PART 3: RUN ONE EPISODE
#
# Let the agent play one full episode using the current policy.
# Collect everything we need for the TRPO update:
#   - what state was the agent in
#   - what action did it take
#   - what reward did it get
#   - what was the probability of that action under the current policy


def run_episode(env, policy, max_steps=100):
    obs = env.reset()
    
    states      = []  # observations at each step
    actions     = []  # action taken at each step
    rewards     = []  # reward received at each step
    log_probs   = []  # log probability of the chosen action (needed for TRPO)

    for _ in range(max_steps):
        # Convert observation to tensor
        obs_t = torch.FloatTensor(obs)

        # Ask the policy: what's the probability of each action here?
        with torch.no_grad():
            probs = policy(obs_t)

        # Sample an action — don't always take the best one, sample randomly
        # weighted by the probabilities. This is how the agent explores. - They (indeces) come out sorted.
        action = torch.multinomial(probs, 1).item()

        # Remember the log probability of this action.
        # Why log? Because multiplying many small probabilities underflows to zero.
        # Log turns multiplication into addition. Numerically stable.
        log_prob = torch.log(probs[action])

        # Take the action in the environment
        next_obs, reward, done = env.step(action)

        states.append(obs_t)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

        obs = next_obs
        if done:
            break

    return states, actions, rewards, log_probs



# PART 4: COMPUTE RETURNS
#
# After an episode, for each step we compute the "return" —
# total discounted future reward from that step onwards.
#
# Why discounted? Future rewards are worth less than immediate ones.
# gamma=0.99 means a reward 100 steps later is worth 0.99^100 ≈ 37% now.
#
# Example with gamma=0.99 and rewards [−0.1, −0.1, +10]:
#   step 2 return = 10
#   step 1 return = −0.1 + 0.99×10 = 9.8
#   step 0 return = −0.1 + 0.99×9.8 = 9.6


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):   # walk backwards through the episode
        G = r + gamma * G
        returns.insert(0, G)       # insert at front to maintain order

    returns = torch.FloatTensor(returns)

    # Normalize: zero mean, unit variance.
    # This keeps the gradient updates stable — without this,
    # returns from lucky episodes (high rewards) dominate unfairly.
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns



# PART 5: THE SURROGATE OBJECTIVE  ← THE HEART OF TRPO
#
# Remember from the lecture:
#   L(π_old, π_new) = E[ (π_new / π_old) × advantage ]
#
# The ratio π_new/π_old is the importance weight.
# In log space: ratio = exp(log π_new − log π_old)
#
# Why advantage instead of return?
# Advantage = return − baseline. It asks "was this action BETTER THAN AVERAGE?"
# We use the mean return as a simple baseline here.
# This reduces variance in the gradient estimate.


def surrogate_objective(policy, states, actions, old_log_probs, returns):
    # Stack all states into one batch tensor
    states_t = torch.stack(states)

    # Get new policy's probabilities for all states at once
    probs = policy(states_t)   # shape: [num_steps, 4]

    # Get log probability of the specific action taken at each step
    actions_t = torch.LongTensor(actions)
    # Batch things
    new_log_probs = torch.log(probs.gather(dim=1, index=actions_t.unsqueeze(1)).squeeze(1))

    # Stack old log probs
    old_log_probs_t = torch.stack(old_log_probs).detach()

    # Importance ratio: how much more/less likely is the new policy
    # to take the same action the old policy took?
    ratio = torch.exp(new_log_probs - old_log_probs_t)

    # Advantage = return − mean(return)
    # Simple baseline: just the mean. Tells us "was this step above average?"
    advantages = returns - returns.mean()

    # Surrogate = weighted advantage
    # If ratio > 1: new policy likes this action more → scaled up
    # If ratio < 1: new policy likes this action less → scaled down
    return (ratio * advantages).mean()



# PART 6: KL DIVERGENCE
#
# Measures how different the new policy is from the old policy.
# KL(old || new) = Σ old(a) × log(old(a) / new(a))
#
# When KL = 0: policies are identical
# When KL is large: policies behave very differently
#
# TRPO says: only accept the new policy if KL ≤ δ (the trust region)


def compute_kl(policy, old_probs_detached, states):
    states_t = torch.stack(states)
    new_probs = policy(states_t)

    # KL divergence formula, averaged across all states in the episode
    kl = (old_probs_detached * torch.log(old_probs_detached / (new_probs + 1e-8))).sum(dim=1)
    return kl.mean()


# PART 7: THE TRPO UPDATE  ← WHERE THE MAGIC HAPPENS
#
# Full TRPO uses conjugate gradients + Fisher matrix inverse.
# That's mathematically heavy and 200+ lines of code.
#
# Instead, we use a clean approximation that captures the exact
# same spirit: gradient ascent with a backtracking line search.
#
# Here's the flow:
#   1. Compute the gradient of the surrogate objective
#   2. Try a step in that direction
#   3. Check: did KL stay below δ? Did the objective improve?
#   4. If yes → keep the step. Done.
#   5. If no  → shrink the step by half. Go to 3.
#
# This backtracking line search IS the trust region enforcement.
# It's the same idea as full TRPO, just without the Fisher matrix math.


def trpo_update(policy, states, actions, old_log_probs, returns,
                delta=0.01, max_backtracks=10):

    #  Step 1: get old policy probs (frozen, for KL later) 
    states_t = torch.stack(states)
    with torch.no_grad():
        old_probs = policy(states_t).detach()

    #  Step 2: compute surrogate and its gradient 
    loss = surrogate_objective(policy, states, actions, old_log_probs, returns)

    # Compute gradient of surrogate w.r.t. all policy parameters
    grads = torch.autograd.grad(loss, policy.parameters())

    # Flatten all gradients into one long vector
    flat_grad = torch.cat([g.view(-1) for g in grads])

    #  Step 3: save current parameters (so we can restore if needed) 
    old_params = torch.cat([p.data.view(-1) for p in policy.parameters()])

    #  Step 4: backtracking line search 
    # Start with step_size=1.0, halve it each time until KL ≤ δ and loss improves.
    step_size = 1.0
    old_loss  = loss.item()

    for attempt in range(max_backtracks):
        # Propose new parameters: move in gradient direction by step_size
        new_params = old_params + step_size * flat_grad

        # Load these new parameters into the policy
        idx = 0
        for p in policy.parameters():
            size = p.numel()
            p.data.copy_(new_params[idx: idx + size].view(p.shape))
            idx += size

        # Check: how different is the new policy from the old?
        kl = compute_kl(policy, old_probs, states).item()

        # Check: is the surrogate objective actually better?
        new_loss = surrogate_objective(
            policy, states, actions, old_log_probs, returns
        ).item()

        if kl <= delta and new_loss > old_loss:
            # Both conditions met. Keep this update.
            return True

        # Either KL too large or objective didn't improve.
        # Shrink step and try again.
        step_size *= 0.5

    # All attempts failed. Restore old parameters.
    idx = 0
    for p in policy.parameters():
        size = p.numel()
        p.data.copy_(old_params[idx: idx + size].view(p.shape))
        idx += size

    return False



# PART 8: THE TRAINING LOOP
#
# This is where everything comes together.
#
# For each episode:
#   1. Run the episode with the current policy
#   2. Compute returns
#   3. Do a TRPO update
#   4. Print the reward


def train():
    env    = GridWorld()
    policy = PolicyNet()

    num_episodes = 800
    print("Training TRPO on 5×5 GridWorld\n")
    print(f"{'Episode':>8}  {'Total Reward':>13}  {'Steps':>6}")
    print("-" * 35)

    for episode in range(num_episodes):

        #  Run one episode 
        states, actions, rewards, log_probs = run_episode(env, policy)

        total_reward = sum(rewards)
        steps = len(rewards)

        #  Compute returns for each step 
        returns = compute_returns(rewards)

        #  TRPO update 
        trpo_update(policy, states, actions, log_probs, returns)

        #  Print progress every 50 episodes 
        if (episode + 1) % 50 == 0:
            print(f"{episode+1:>8}  {total_reward:>13.2f}  {steps:>6}")

    print("\nTraining done.\n")
    return policy



# PART 9: WATCH THE TRAINED AGENT
#
# After training, let the agent play one episode greedily
# (always pick the highest probability action, no randomness)
# and print the grid at each step so you can see the path.


def watch(policy):
    env = GridWorld()
    obs = env.reset()
    print("=== Trained agent's path ===")
    env.render()

    for step in range(50):
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            probs  = policy(obs_t)
            action = torch.argmax(probs).item()   # greedy: best action only

        action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
        obs, reward, done = env.step(action)

        print(f"Step {step+1}: {action_name}  (reward: {reward:.1f})")
        env.render()

        if done:
            print("Goal reached!")
            break



# RUN
if __name__ == "__main__":
    trained_policy = train()
    watch(trained_policy)