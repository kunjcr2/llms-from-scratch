### Lec5 - Multi Arm Bandits

- Lets say pulling a lever is an action for that "7-7-7" game or "One arm bandit". But instead of 1 lever, we have multiple.
- Every action `a` has a true value as q(a), which is a reward of that action. And since it wont be constant, we can call it $Q_t(a)$.
- Estimated reward would for action a would be average of ALL the rewards that were gained in past afteer doing action a. It would be-

$Q_t(a) = (R_1+R_2+...+R_{N_t})/N_t$

- BRO IS TALKING ABOUT `GREEDY ACTION`. We use a small epsilon (lets say 0.1) that makes 10 moves go NON GREEDY out of 100 moves.
- haha lol.
- Its a great practical way to show how agent learns in the environment.

> # **UPCOMING REINFORCEMENT LEARNING STUFF WILL BE IN ../RL**