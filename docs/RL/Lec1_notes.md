Supervised Learning - Data and labels
Unsupervised Learning - Not labeled - clustering of data and such
Reinforcement learning - Models learn from interactions with the evironment

Both top 2 learning generalize to a particuar environment but NOT RL.

AlphaGo, AlphaZero - A goal is there

> Agent -> (interaction) -> Environment -> Goal

### 4 main elements of RL systems: 
1. **Policy**: It deifines Agent's way of behaving at a given time. It is state to action. Action based on current state of environment.
2. **Reward Signals**: Defines the goal in RL problem. At each time step, the env sends RL agent, a single number - "Reward". The objective of the agent is to maximize the reward recieved overtime.
3. **Value function**: This is reward but for a looooooooooooong term. Reward is immediate response while value function is something that keeps a long term value of that agent. | _(Karun nair playing in all 5 match, after thinking he will perform better in next match after bad first match. Reward is first match - poor. Value, consistent 40s - Good.)_ | 
4. **Model of the environment**: It is something that mimics the behavior of Environment.
5. **State**: Situation of the environment.

**Exploitation** vs **Exploration**: First means that we select the state where value function's value is highest. Second means we deliberately choose the one with "non-highest" state cuz we wanna explore other states as well.

- Wee change the value function values based on next value by backpropogating. IF we can win a tic tac toe game by placing X at a specific place then value function for that state will go UP as it is a winning position. And decrease where we can loose.