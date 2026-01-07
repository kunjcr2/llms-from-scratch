# Policy Gradient Theorem

### Policy Gradient Methods

•	Objective: Directly learn a policy (probability distribution over actions given a state) without explicitly calculating Q-values. This is more scalable for complex problems than tabular methods or Q-value approximation with neural networks (0:40, 5:15, 6:16).
•	Parametrized Policy: The policy is represented by a function with parameters (theta), often like weights in a neural network (7:33, 9:35).
•	Softmax Function: Used to convert numerical preferences for actions into probabilities between 0 and 1 (10:30).

### Policy Gradient Theorem

•	Goal: Maximize a performance measure J(theta), which is typically the value function of the initial state (V_pi(s0)), representing the expected cumulative rewards (13:13, 16:21).
•	Gradient Ascent: To maximize J(theta), we use gradient ascent: `thetat+1 = thetat + alpha * grad(J(theta))` (14:10).
•	The Challenge: Calculating `grad(J(theta))` is difficult because the policy's performance depends on states visited and actions taken, which are stochastic (15:47).
•	The Breakthrough (1999): The Policy Gradient Theorem provided a formula for `grad(J(theta))` (18:03).
•	Log-Derivative Trick: A key mathematical trick used in the derivation: `grad(P(tau)) = P(tau) * grad(log P(tau))` (23:14).
•	Core Formula: `grad(J(theta)) = E[sum{t=0 to T} grad(log pi(at|s_t, theta)) * G(tau)]` (29:38).
•	This means: sample trajectories, sum the gradients of the log-probabilities of actions taken, and multiply by the total return (G) of the trajectory.
•	Intuition: If an action leads to a positive return (G is positive), increase its probability; if it leads to a negative return (G is negative), decrease its probability (33:57, 34:49).
