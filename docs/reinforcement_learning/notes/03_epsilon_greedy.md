# Epsilon Greedy Policy

Few small concepts. **Greedy Policy** is where we choose the action with highest value. **Epsilon greedy policy** is where we choose action with highest value but for small amount of steps, we take another value which is not highest.

- **Process**: What happens is Agent is given a state whihc based on the policy and value function of next steps to be taken, makes an actioon. Which is then passed through the environment and environment sends back a new state and a reward.

- **Policy** is denoted by PI.

- **Expected Return**: The goal is not about maximizing the reward, its about maximizing the RETURN which is addition of all the rewards from that point to the end of episode. It is called as _Expected Returns_. Denoted by **G_t**.

$G_t$ = $R_{t+1}$ + $R_{t+2}$ + ... + $R_T$

- **Discounting**: Rewards that are gained earlier are mre valuable compared to those which are given after a long time. We do `R_1 + gamma*R_2 + gamma*R3 + ...` instead of `R_1 + R_2 + R_3`; etc.

- **Discounted expected Return**:

$G_t$ = $R_{t+1}$ + r\*$R_{t+2}$ + $r^2R_{t+2}$ + ...

- Gamma is a discount rate.
