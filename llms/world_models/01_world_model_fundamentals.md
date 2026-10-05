# World Models From Scratch - Lecture 1

## 1. What Is a World Model?

A world model is a learned or computed model of an environment. It can act as a substitute for the environment by taking an action as input and predicting what happens next.

At a high level, world models help an agent simulate possible futures before taking actions in the real environment. This is similar to how humans use an internal model of the world to evaluate alternatives and choose an action.

## 2. Agent-Environment Interface

The basic interaction loop contains two components:

- **Agent:** observes the environment and chooses actions.
- **Environment:** receives an action, updates its state, and returns a new observation and reward.

At time $t$:

1.  The agent receives an observation $o_t$ (or, in a fully observable setting, the state $s_t$).
2.  It selects an action $a_t$.
3.  The environment transitions to a new state $s_{t+1}$.
4.  The environment returns an observation $o_{t+1}$ and a reward $r_{t+1}$.

The loop can be summarized as:

$$
a_t = \pi(o_t), \qquad s_{t+1} = f(s_t, a_t), \qquad o_{t+1} = g(s_{t+1})
$$

In a model-free setting, the agent does not have access to the environment's transition rules. A world model provides an approximation of those rules.

### Example: Robot Folding Clothes

- **Agent:** a robot with an AI controller.
- **Environment:** the cloth and table, or the whole room around the robot.
- **Observation:** camera images or other sensor readings.
- **Action:** movements of the robot's joints, including their positions and movement rates.
- **Reward:** a positive signal when the cloth is successfully folded.

The robot repeatedly observes the cloth, takes an action, observes the updated situation, and continues until the task is complete.

### Example: Platformer Game

- **Observation:** an image of the game screen, represented computationally as pixel values or a compressed latent representation.
- **Actions:** move left, move right, jump, or do nothing.
- **Reward:** a signal for collecting items, reaching a goal, or completing the level.

The agent selects one of the available actions based on the current observation. The environment determines how the game changes after that action.

## 3. State Versus Observation

These terms are related but not identical.

### State

The **state** is the complete internal situation of the environment: everything needed to determine what happens next. It may include positions, velocities, hidden variables, object configurations, and other information that the agent cannot directly access.

### Observation

The **observation** is the information that the agent receives through its sensors. It is usually only a partial representation or subset of the true state.

For example, a robot's camera may show where a cloth appears to be, but it may not reveal the exact forces, folds, friction, or hidden parts of the cloth.

The world may know its complete state, while the agent only receives an observation:

$$
o_t = g(s_t)
$$

The goal is often to act successfully using limited observations rather than complete state information.

## 4. Fully and Partially Observable Environments

- **Fully observable environment:** the observation contains all information necessary to determine the current state.
- **Partially observable environment:** important parts of the state are hidden from the agent.

Most real-world problems are partially observable because no sensor captures the entire state.

### Chess and Poker

- **Chess:** generally treated as fully observable because the board and all pieces are visible.
- **Poker:** partially observable because the opponent's cards are hidden.

A partially observable environment is often modeled as a **POMDP**, or partially observable Markov decision process.

### Example: Pong

A single screenshot of Pong shows the ball's position, but not its velocity. Therefore, it may be impossible to tell whether the ball is moving left or right from one frame alone.

A common solution is to stack several consecutive frames. The sequence provides temporal information, allowing the agent to infer motion and estimate hidden variables such as velocity.

In general, a sequence of observations can provide a better proxy for the state than one observation:

$$
h_t = (o_{t-k+1}, \ldots, o_t)
$$

where $h_t$ is a history of recent observations.

## 5. Rewards

A reward is feedback from the environment about progress toward a goal.

- Rewards may be immediate or delayed.
- A folding robot may receive a reward only after the cloth is successfully folded.
- A game-playing agent may receive a reward after collecting an item or finishing a game.
- In language-model reinforcement learning, the generated tokens form a trajectory and the response can be assigned a reward based on its quality.

Delayed rewards make the problem harder because the agent must connect earlier actions with a later outcome.

## 6. World Models as Environment Simulators

The world model belongs primarily on the environment side of the agent-environment interface.

Instead of using a real game engine or manually written physics rules, a world model can learn to predict the next state or observation after an action:

$$
\hat{s}_{t+1} = \hat{f}(s_t, a_t)
$$

or, when predicting observations directly:

$$
\hat{o}_{t+1} = \hat{f}(o_t, a_t)
$$

The agent can then interact with the model as if it were interacting with the real environment. It can simulate multiple possible action sequences and use the predicted outcomes to select a policy.

The world model may be:

- A component that replaces part of the environment.
- A complete learned simulator for the environment.

This is useful in reinforcement learning because planning can happen inside the learned model without requiring every experiment to occur in the real world.

## 7. Brief History of World Models

### Kenneth Craik, 1943

Kenneth Craik proposed that an organism can carry a small-scale model of the world, use it to test alternatives, determine which option is best, and respond to future situations before they occur.

This idea is close to the modern view of world models: construct an internal model, simulate possible futures, and use those simulations to guide actions.

### Tolman's Rat Maze Experiments

Rats that explored a maze could later find shortcuts to a reward, including paths they had not previously walked. This suggested that they learned an internal map or model of the maze rather than merely memorizing rewarded turns.

### Kalman Filter

A Kalman filter estimates a hidden state from noisy sensor measurements and predictions. It illustrates the distinction between observations and state estimation:

- Sensors provide noisy observations.
- A dynamics model predicts the next state.
- The prediction is corrected using new observations.

### Sutton's Model-Based Reinforcement Learning

Richard Sutton described replacing the environment with a model and trying out actions in that model to find a good policy. This is the central motivation behind model-based reinforcement learning.

### 2018 and Beyond

The 2018 _World Models_ work demonstrated an agent learning a compact representation of an environment and performing simulations in a learned latent space. Since then, world-model research has expanded to systems such as Dreamer and newer models that simulate games, robotics, driving, and broader environments.

## 8. Renderer, Simulator, and Planner

A useful taxonomy divides world-model systems into three related capabilities.

### Renderer

A renderer generates realistic observations, such as images or video frames, from an internal state. Video-generation systems are examples of models with strong rendering capabilities.

### Simulator

A simulator models how the environment changes after an action. It answers questions such as:

> What will happen if I push the cup at this location?

The simulator is the core component because it captures environment dynamics.

### Planner

A planner searches over possible actions and selects a sequence that achieves a goal. It answers questions such as:

> What should the robot do to fold the cloth successfully?

Planning usually depends on a simulator so that candidate actions can be evaluated.

### Relationship Between the Components

- A renderer needs a model of dynamics to produce meaningful future observations.
- A planner needs a simulator to evaluate possible futures.
- A system may combine simulator and renderer, simulator and planner, or all three.
- The simulator is the central component in this taxonomy.

Examples discussed in the lecture include:

- Sora-like video models: primarily renderers.
- Dreamer and similar systems: simulators with latent-state prediction.
- Genie-like systems: combinations of simulation, rendering, and planning capabilities.

When reading a world-model paper, it is more informative to identify the specific capabilities it provides rather than simply labeling it a world model.

## 9. Key Takeaways

1.  The agent-environment loop consists of observations, actions, state transitions, and rewards.
2.  State is the environment's complete internal situation; observation is what the agent can sense.
3.  Most real-world tasks are partially observable and can be modeled as POMDPs.
4.  Stacking frames or maintaining observation history can help infer hidden state variables such as velocity.
5.  A world model acts as a learned stand-in for the environment and predicts future states or observations after actions.
6.  World models enable simulation and planning without repeatedly interacting with the real environment.
7.  The main taxonomy is renderer, simulator, and planner, with the simulator forming the foundation for the other capabilities.
