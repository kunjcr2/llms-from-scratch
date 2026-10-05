# World Models From Scratch - Lecture 2

## Building the World-Model Toolkit

Lecture 2 introduces fundamental building blocks used to construct the three major parts of a world-model system:

- **Simulator:** models how the environment changes after an action.
- **Planner:** chooses actions that are expected to produce good future outcomes.
- **Renderer:** generates human-interpretable observations such as images or video.

The main tools covered are compression, reward and value functions, imagined rollouts, and policy-learning methods.

## 1. From Lecture 1 to Lecture 2

The agent-environment loop can be summarized as follows:

1. The environment provides an observation.
2. The agent selects an action.
3. The environment uses that action to produce a new state.
4. The agent receives a new observation and possibly a reward.

In Lecture 1, the world model was introduced as a replacement for the environment. It receives an action and predicts the next state or observation. Lecture 2 explains how to make this prediction tractable and how an agent can use it to plan.

The environment usually provides only a partial observation of its true state. A useful world model therefore needs a compact representation that preserves the information relevant to predicting the future.

## 2. Compression: From Pixels to Latents

Raw observations, especially images, contain a very large number of values. Passing every pixel through the simulator is expensive and often unnecessary. Instead, world models commonly compress an observation into a lower-dimensional representation called a **latent**.

The basic pipeline is:

$$
\text{observation } o_t \xrightarrow{\text{encoder}} z_t \xrightarrow{\text{dynamics model}} \hat{z}_{t+1}
$$

where $z_t$ is the latent representation of the observation.

### Sketch-Artist Analogy

Consider a witness describing a face to a sketch artist:

- The actual face contains millions of pixels.
- The witness remembers only meaningful features such as the shape of the nose, eyes, hair, and face.
- The witness is analogous to an **encoder** that produces a compact description.
- The sketch artist is analogous to a **decoder** that reconstructs the face from that description.
- The quality of the reconstruction indicates whether the compact description retained the important information.

The goal is not necessarily to preserve every pixel exactly. The representation should preserve the features needed to reconstruct the observation or predict what happens next.

### Autoencoders

An autoencoder consists of two main components:

- **Encoder:** maps an input observation to a compact latent vector.
- **Decoder:** reconstructs the observation from that latent vector.

For an image $x$, the process is:

$$
z = E(x), \qquad \hat{x} = D(z) = D(E(x))
$$

The model is trained so that the reconstruction $\hat{x}$ remains close to the original image $x$.

For example, a $200 \times 200$ image contains 40,000 pixel positions before accounting for color channels. An encoder might represent it using only 32 latent values. This is a substantial compression, but the latent must still retain the information that matters for the task.

Other approaches can also create compressed visual representations, including:

- Variational autoencoders.
- Convolutional neural networks.
- Vision Transformers.

### Latent Space and Similarity

Images that are semantically similar should ideally be close together in latent space. For example, images of smiling faces may form one cluster, while images of angry faces may form another.

This makes the latent space useful for comparison. Given two latent vectors, their similarity can be estimated using measures such as cosine similarity:

$$
\operatorname{cosine\ similarity}(z_1,z_2) =
\frac{z_1 \cdot z_2}{\|z_1\|\|z_2\|}
$$

In a high-dimensional pixel space, comparing images directly can be difficult because small pixel-level changes may produce large numerical differences. A well-structured latent space makes meaningful similarities easier to identify.

### Why World Models Use Latent Space

Predicting every pixel at every time step is computationally expensive. Most world models therefore follow this pattern:

1. Encode the sensor observation once.
2. Predict the next latent representation rather than every raw pixel.
3. Use the latent representation for downstream simulation and planning.
4. Decode back to pixels only when a human-viewable output is needed.

Some models operate entirely in latent space and do not need a decoder during their main training or planning process. The model may still render pixels for visualization, but its internal predictions remain latent vectors.

There are exceptions. Some systems predict directly in pixel space, but latent-space prediction is generally more efficient because it reduces the dimensionality of the dynamics problem.

### Core Recipe

The central recipe for many world models is:

> **Compress, then predict.**

Compression is especially important for the simulator because it provides a manageable representation in which future dynamics can be modeled.

## 3. Rewards Over Time

An agent should not judge an action only by its immediate reward. Some actions have a low or negative immediate reward but lead to a much better future.

### Examples

- In chess, sacrificing a queen may lead to winning the game several moves later.
- Studying for an exam may be unpleasant immediately but produce a larger future benefit.
- A sports coach may keep a player who performs poorly at first if that player is expected to contribute over the whole season.

The total future reward associated with a trajectory is called the **return**.

For a sequence of rewards, the undiscounted return from time $t$ is:

$$
G_t = r_{t+1} + r_{t+2} + r_{t+3} + \cdots
$$

The planner should generally choose actions that maximize long-term return rather than only the next reward.

## 4. Discounted Return

Future rewards are usually discounted by a factor $\gamma$, where $0 \leq \gamma \leq 1$. The discounted return is:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \cdots
$$

For example, if the rewards are 50, 40, and 30 and $\gamma=0.9$, the estimated return is:

$$
G_t = 50 + 0.9(40) + 0.9^2(30) + \cdots
$$

Discounting gives greater weight to rewards that happen sooner. Its main motivations are:

- The future is uncertain.
- Predictions farther into the future are less reliable.
- A reward available now may be more useful than the same nominal reward much later.

The discount factor decreases the contribution of later rewards: $0.9^2=0.81$, $0.9^3=0.729$, and so on.

## 5. Value Functions

### State Value

The **value function** estimates the expected return from a state when following a policy. It answers:

> How good is this state, on average, for the remainder of the episode?

The state-value function is commonly written as:

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t=s]
$$

For example, a chess grandmaster can look at a position and judge that it is winning before the game has finished. This judgment is an estimate of the expected future return from that state.

The value of a state is useful, but it does not by itself identify which particular action should be taken when several actions are available.

### Action-Value Function

The **action-value function**, usually called the $Q$-function, estimates the expected return from taking a particular action in a particular state and then following a policy:

$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]
$$

This gives one score for every state-action pair. If the available actions have values 100, 150, 200, and 5, the planner should choose the action with value 200.

The greedy action-selection rule is:

$$
 a_t = \arg\max_a Q(s_t,a)
$$

This is why action values are especially useful to a planner: they directly compare the possible actions at the current state.

## 6. Planning with World Models

The simulator and planner work together:

- The **simulator** predicts what will happen after an action.
- The **planner** evaluates those possible futures and selects an action.

Suppose an agent is playing a game and can move left, move right, move forward, or shoot. The planner can simulate a possible trajectory for each action, estimate its return, and select the action associated with the best expected outcome.

Conceptually, the process is:

1. Start from the current state or latent representation.
2. Try possible actions inside the simulator.
3. Roll out each action into one or more future trajectories.
4. Estimate the return or value at the end of each trajectory.
5. Select the action with the highest expected value.

The planner is therefore the component that determines which action the agent should take for a given state.

## 7. Rollouts and Dreaming

A **rollout** is a simulated sequence of states, actions, and rewards. It represents one possible future generated by the world model.

If the agent performs many rollouts inside the simulator, it can compare alternative futures before acting in the real environment. This process is often described as **dreaming** or **imagined training**.

In a learned environment, the agent can:

- Try actions without physically interacting with the real world.
- Make mistakes cheaply.
- Observe the simulated consequences.
- Update its policy based on the returns it receives.

This is useful for a robot folding clothes. Instead of physically attempting the task hundreds of times, the robot can train in a learned simulator. Once its policy becomes sufficiently good, it can be transferred to the real robot.

The quality of this transfer depends heavily on the quality of the world model. If the simulator predicts unrealistic consequences, the agent may learn a policy that works in imagination but fails in reality.

### Dreaming in the 2018 World Models Work

The 2018 _World Models_ work demonstrated the idea of training an agent inside a learned, hallucinated environment. The agent could generate imagined game trajectories, evaluate them, and improve its policy without continuously interacting with the real game.

In this interpretation:

- The world model supplies the simulated environment.
- The planner or policy supplies the agent's behavior.
- Dreaming consists of repeated imagined interaction between the two.

## 8. Policy-Improvement Methods

A **policy** specifies how an agent chooses actions from states or observations:

$$
\pi(a \mid s) = \text{probability of selecting action } a \text{ in state } s
$$

Two broad methods discussed in the lecture are Q-learning and actor-critic methods.

### Q-Learning

Q-learning learns action values. Once the agent has an estimate of $Q(s,a)$ for every available action, it can choose the action with the largest value:

$$
a^* = \arg\max_a Q(s,a)
$$

The $Q$-function can therefore serve as the basis for a policy.

### Actor-Critic Methods

Actor-critic methods divide the work between two learned components:

- **Actor:** proposes an action.
- **Critic:** evaluates the action or the state and provides feedback.

The actor improves its behavior using the critic's evaluation. Through repeated trial and error, both components become better at selecting actions that lead to high returns.

Many world-model systems use actor-critic methods for planning and train them inside imagined environments. The agent can rehearse thousands of episodes cheaply, while the critic scores the imagined outcomes.

## 9. Where Reinforcement Learning Fits

World models and reinforcement learning play different but connected roles:

- The world model provides the simulator and predicts environment dynamics.
- Reinforcement learning provides tools for improving the planner or policy.
- The value function estimates long-term outcomes.
- The action-value function helps choose among actions.

The world model lets the agent look ahead. Reinforcement learning helps the agent decide which imagined outcomes are desirable and how to improve its policy.

This connection is often called **model-based reinforcement learning**: the agent learns or uses a model of the environment and then improves its behavior through simulated experience.

## 10. Key Takeaways

1. World-model systems are built from simulator, planner, and renderer components.
2. Compression transforms high-dimensional observations such as images into compact latent representations.
3. Autoencoders use an encoder and decoder to test whether the latent retains important information.
4. Many simulators follow the recipe: **compress, then predict**.
5. Return is the cumulative future reward, while discounted return gives less weight to rewards farther in the future.
6. The value function estimates the expected return from a state.
7. The action-value function estimates the expected return for a specific state-action pair and is directly useful for action selection.
8. Rollouts are imagined trajectories generated by a simulator.
9. Dreaming allows an agent to train through simulated interaction before deployment in the real world.
10. Q-learning and actor-critic methods are two ways to improve the policy used by the planner.
