# REINFORCE Algorithm

### REINFORCE Algorithm

•	First Policy Gradient Algorithm: Directly uses the derived formula to update policy parameters (37:12).
•	Policy Update Rule: `thetat+1 = thetat + alpha * E[sum{t=0 to T} grad(log pi(at|s_t, theta)) * G(tau)]` (36:36).
•	Problem: High Variance. The return `G(tau)` for each trajectory can vary wildly, leading to unstable gradient estimates (38:28, 40:21).

### REINFORCE with Baseline

•	Solution to Variance: Subtract a baseline `B(s)` from the return `G(tau)` (41:50).
•	Modified Formula: `grad(J(theta)) = E[sum{t=0 to T} grad(log pi(at|s_t, theta)) * (G(tau) - B(s))]` (42:12).
•	Benefits: Reduces variance without introducing bias (as `B(s)` doesn't depend on actions) (42:49).
•	Common Baseline: The state-value function `V_pi(s)` is often used as the baseline (45:41).
•	Actor-Critic Formulation: When the baseline is `V_pi(s)`, the method is often called actor-critic. The "actor" (policy) takes actions, and the "critic" (value function) evaluates the actions (46:33).
