# Deep Q-Network (DQN)

⸻

## 1. The Motivation

In traditional Q-learning, we had a Q-table that stored the expected long-term reward for every state–action pair.
That works fine for simple problems like grid worlds, but when we deal with environments like the game of Pong, it becomes impossible.

Each frame of Pong has 210 × 160 × 3 pixels.
If every pixel can take 256 values, the total number of possible states is astronomically large.
We cannot store that in a table.

So we ask a new question:

Can we train a neural network to approximate these Q-values instead of storing them in a table?

That is exactly what a Deep Q-Network (DQN) does.

⸻

## 2. What the Agent Sees and Does

In Pong:
	•	State (s): one game frame (later, we'll use four stacked frames so the agent sees motion).
	•	Actions (a): move paddle up, move paddle down, or do nothing.
	•	Reward (r):
	•	+1 if we score a point,
	•	–1 if the opponent scores,
	•	0 otherwise.

So the agent receives a frame (the observation), chooses one of three actions, and receives a reward from the environment.

⸻

## 3. The Goal

We want to learn the Q-function:

Q(s, a) = \text{expected total future reward if we take action a in state s}

Once we know Q(s, a) for all possible actions, we simply choose the action with the highest Q-value.

In practice, we'll train a neural network that takes in the game frame as input and predicts Q-values for all actions as output.

⸻

## 4. The Learning Rule

In standard Q-learning, we updated the table like this:

Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]

In DQN, we keep the same idea, but instead of updating numbers in a table, we train a neural network so that its prediction Q_\theta(s,a) matches the target:

\text{Target} = r + \gamma \max_{a'} Q_{\theta^-}(s', a')

Here:
	•	Q_\theta is our main network,
	•	Q_{\theta^-} is a target network (a frozen copy used to stabilize training).

We minimize the difference between the predicted Q-value and this target using mean squared error.

\text{Loss} = ( \text{Target} - Q_\theta(s,a) )^2

⸻

## 5. The Four Major Ideas That Make DQN Work

### (1) Epsilon-Greedy Exploration

At the start, the agent doesn't know what to do.
If it always takes the "best" action according to its (untrained) network, it won't explore new possibilities.

To fix this, we use an epsilon-greedy policy:
	•	With probability ε → take a random action (exploration)
	•	With probability (1 – ε) → take the action with the highest Q-value (exploitation)

We start with ε = 1 (completely random) and slowly reduce it as the agent becomes smarter.

Example (simplified code):

```python
import random, torch

def select_action(net, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    q_values = net(state)          # predict Q-values for all actions
    return int(q_values.argmax())  # choose action with max Q-value
```

⸻

### (2) Replay Buffer (Experience Replay)

When playing, consecutive frames are highly correlated.
If we train directly on them, the network becomes unstable.

To fix this, we store all past experiences (state, action, reward, next_state) in a large replay buffer.
During training, we pick random batches from this buffer, breaking the correlation and stabilizing learning.

Simplified view:
```python
buffer = []
# Store transitions
buffer.append((s, a, r, s_next))
# Later, sample randomly
batch = random.sample(buffer, batch_size)
```

⸻

### (3) Target Network

If we use the same network to calculate both the prediction and the target, both keep changing together, making training unstable.

To avoid this, we make a copy of the network, called the target network.
It is updated only once every few thousand steps.
This gives us a fixed target for a while.
```python
def update_target(target_net, main_net):
    target_net.load_state_dict(main_net.state_dict())
```

⸻

### (4) Frame Stacking

A single Pong frame doesn't tell us if the ball is moving up or down.
To understand motion, we stack the last 4 frames together.
This gives the agent a sense of direction and speed.

So, the state becomes a tensor of shape [4, 84, 84] (four grayscale frames).

⸻

## 6. The Neural Network Architecture

Because our input is an image, we use a Convolutional Neural Network (CNN).
This helps the agent understand visual patterns such as the ball and paddle.

Typical structure:

Input: 4 stacked grayscale frames (84×84×4)
Conv(32 filters, 8×8, stride 4) → ReLU
Conv(64 filters, 4×4, stride 2) → ReLU
Conv(64 filters, 3×3, stride 1) → ReLU
Flatten
Linear(512) → ReLU
Linear(3) → outputs Q-values for each action

Each output represents the Q-value for one action: up, down, or no-op.

Minimal code sketch:
```python
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = x / 255.0                       # normalize pixels
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

⸻

## 7. The Step-by-Step Algorithm
	1.	Initialize:
	•	Main Q-network (θ)
	•	Target network (θ⁻ = θ)
	•	Replay buffer (empty)
	•	ε = 1.0
	2.	Play:
	•	Select action using ε-greedy.
	•	Execute action → observe reward and next state.
	•	Store (s, a, r, s') in the replay buffer.
	3.	Sample:
	•	Randomly sample a batch from the buffer.
	•	Compute target:
y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
	•	Compute loss:
L = (y - Q_{\theta}(s, a))^2
	•	Backpropagate to update the main network.
	4.	Sync:
	•	Every few thousand steps, copy weights from main network to target network.
	5.	Decay:
	•	Gradually reduce ε to shift from exploration to exploitation.

⸻

## 8. Intuitive Understanding

Initially, the paddle moves randomly.
By chance, it sometimes hits the ball and gets a positive reward.
That reward makes the network increase the Q-values of the moves that led up to the hit.
Gradually, the network learns which actions lead to success and begins to repeat them.

This process continues until the agent learns a consistent strategy to hit the ball back and eventually win most games.

⸻

## 9. Key Points to Remember
	•	DQN replaces the Q-table with a neural network that predicts Q-values.
	•	Training is done by minimizing the difference between the predicted Q-value and the Bellman target.
	•	The replay buffer and target network are essential for stability.
	•	Frame stacking helps the network understand motion.
	•	The Bellman equation is at the heart of the entire learning process.
