# World Models From Scratch - Lecture 10

## DINO-WM: Planning in Feature Space

Lecture 10 introduces **DINO-WM**, an action-conditioned world model that predicts future observations in a learned feature space instead of pixel space. The model is used with model predictive control (MPC) to plan actions in imagination.

## 1. Encode Observations into Features

Images contain too many pixels to predict directly. DINO-WM uses a pretrained **DINOv2** vision encoder to convert each image into feature tokens. The encoder is frozen, so training focuses on the dynamics predictor rather than relearning visual representations.

The benefit is both computational and conceptual:

- The predictor works on a much smaller set of meaningful visual tokens.
- It avoids treating every pixel detail as equally important.
- The model can focus on objects, geometry, and other features useful for predicting dynamics.

Actions and proprioceptive data are projected into the same feature dimension as the visual tokens.

## 2. Transformer Dynamics Predictor

The predictor receives a history of recent observations and actions. For each time step, the input contains:

- Visual tokens from the DINO encoder.
- Action tokens.
- Proprioceptive tokens.
- Spatial and temporal position embeddings.

The temporal dimension is important because the same image can mean different things depending on whether it is before or after an action. A **block-wise causal attention mask** prevents a time step from attending to future time steps while allowing tokens within the same step to attend bidirectionally.

The predictor learns

$$
\hat z_{t+1} = g(z_{t-H+1:t}, a_{t-H+1:t}),
$$

where $z$ is the encoded visual state, $a$ is the action, and $H$ is the observation horizon.

## 3. Training and Rollouts

Training uses expert trajectories containing observation-action-next-observation tuples. The model learns to predict the next latent observation. During evaluation, the model can roll out multiple future states autoregressively: predicted observations are fed back as inputs while candidate actions are supplied or selected.

The decoder is optional. Predictions can be compared directly in feature space, which avoids reconstructing pixels and keeps the objective aligned with the representation used for planning.

## 4. Planning with MPC

DINO-WM becomes useful for control when combined with a planner:

1. Start with the current observation and a desired goal observation.
2. Sample candidate action sequences.
3. Roll each sequence forward inside the learned world model.
4. Compare the imagined final states with the goal in feature space.
5. Execute the best action or short action sequence.
6. Replan as new observations arrive.

This is model predictive control. The expensive search happens in imagination rather than through repeated interaction with the real environment. A low-level controller converts the selected action plan into physical motion.

## 5. PushT Example

In the PushT task, the model is trained on expert episodes showing how to move one T-shaped block into alignment with another. After training, MPC can plan from a held-out start frame to a goal frame even though goal reaching was not explicitly part of the training objective.

The main limitation is compute: sampling and evaluating many imagined trajectories at test time can be slow. The model also requires action-labeled interaction data during training, although only limited real-environment interaction is needed compared with trial-and-error planning.
