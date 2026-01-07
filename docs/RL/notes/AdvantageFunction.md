# Advantage Function

### Advantage Function

•	Definition: `A(s,a) = Qpi(s,a) - Vpi(s)` (1:15:04).
•	Intuition: Measures how much better or worse a specific action `a` in state `s` is compared to the average expected return from that state (1:16:15).
•	Positive advantage: Action is better than average, increase its probability.
•	Negative advantage: Action is worse than average, decrease its probability.
•	Estimation (Temporal Difference Error): Advantage can be estimated using the immediate reward and the value function of the next state: `A(s,a) = Rt + gamma * Vpi(St+1) - Vpi(S_t)` (1:22:30).
•	N-step Backups and GAE:
•	One-step backup: Low variance, high bias.
•	Monte Carlo (N-step) backup: High variance, low bias (1:27:03).
•	Generalized Advantage Estimation (GAE): Uses a `lambda` parameter to balance bias and variance by combining different n-step returns (1:25:00). `lambda=0` is one-step TD, `lambda=1` is Monte Carlo (1:26:19).
