# World Models From Scratch - Lecture 3

## Building the First World Model

This lecture builds the simulator component of a world model from scratch. The goal is to learn how an environment changes after an action, without writing a hand-coded physics engine.

The model learns to predict the next frame of a simple game from:

- The current observation.
- The player's current action.
- A memory of what happened in earlier time steps.

The lecture focuses only on the simulator. Rendering is used to visualize predictions, and planning is left for later lectures.

## 1. Mini Pong

The example environment is a small version of Pong:

- The screen is $32 \times 32$ pixels with three color channels.
- The paddle can move left, move right, or stay still.
- The ball moves one pixel per step.
- The ball bounces off the walls and the paddle.
- A collision with the paddle can change the ball's direction.

The rules could be implemented directly with a physics engine, but the objective here is different: learn the dynamics with neural networks.

The simulator should learn a function of the form:

$$
\hat{o}_{t+1} = f(o_t, a_t, \text{memory}_t)
$$

where $o_t$ is the current frame and $a_t$ is the action taken at that time step.

## 2. The Partial-Observability Problem

A single frame does not reveal the ball's velocity. If the ball is in one location, it could be moving upward or downward. The current image tells us the ball's position, but not its direction of motion.

Therefore, the current frame is an observation rather than the complete state. The missing information is contained in the history of previous frames.

This creates an important architectural requirement:

> The prediction network needs a persistent memory that accumulates information across time steps.

Without memory, a network receiving only one frame and one action cannot reliably predict the next frame in situations where the same visual position can lead to different futures.

## 3. Training Dataset

The training data consists of random gameplay rather than expert gameplay:

- **Episodes:** 200.
- **Time steps per episode:** 120, from $t=0$ through $t=119$.
- **Total frame-action pairs:** $200 \times 120 = 24{,}000$.

Each training item contains:

1. The frame observed at a time step.
2. The action taken at that time step.

The player chooses actions randomly. This is sufficient because the simulator is learning the environment's dynamics, not the best playing strategy. The physics of Pong remains the same whether the player acts intelligently or randomly.

Random exploration also avoids requiring an expert to collect trajectories. The world model needs to observe enough possible states and actions to learn how the environment responds.

## 4. Vision Component: Compressing the Frame

The first component is the vision module, $V_{\phi,\psi} = (E_\phi, D_\psi)$, consisting of an encoder and a decoder. The encoder compresses the approximately 3,000 values in a frame into a compact code with 12 values:

$$
z_t = E_\phi(o_t)
$$

Here, $z_t$ is called a **code**, **latent**, or latent representation. It is a compact way to represent the image while preserving its important information.

This is the same encoder-decoder idea introduced in Lecture 2:

$$
o_t \xrightarrow{E_\phi} z_t \xrightarrow{D_\psi} \hat{o}_t
$$

The encoder acts like the witness describing a face, while the decoder acts like the sketch artist reconstructing it from the description.

The downstream simulator operates primarily on the compact code rather than the full pixel array.

### Variational Autoencoder

The vision module is trained as a **variational autoencoder (VAE)**. Its purpose is to produce a compact latent representation from which the original frame can be reconstructed.

The reconstruction loss compares the original frame with the decoded frame:

$$
\mathcal{L}_{\text{reconstruction}} = \mathcal{L}(o_t, D_\psi(E_\phi(o_t)))
$$

The encoder and decoder parameters are updated to reduce this loss.

The VAE does not need to reproduce every pixel perfectly. It needs to preserve the information that matters for representing the game state and predicting future frames.

### Weighted Pixel Loss

Uniformly weighting every pixel creates a problem in Mini Pong. The paddle occupies many pixels, while the ball occupies only a few bright pixels. A standard reconstruction loss can therefore learn to reconstruct the paddle accurately while effectively ignoring the ball, because errors on the small ball contribute little to the total loss.

The solution is to assign greater weight to important pixels, especially bright pixels containing the ball:

$$
\mathcal{L}_{\text{weighted}} = \sum_i w_i\left(o_{t,i} - \hat{o}_{t,i}\right)^2
$$

where $w_i$ is larger for pixels that are more important. Reweighting the loss encourages the latent representation and decoder to preserve the ball as well as the paddle.

This illustrates a general lesson: reconstruction objectives should reflect the information that matters for the task, not merely the number of pixels.

## 5. Memory Component

Even a perfect compressed representation of the current frame does not reveal the ball's velocity. The second component therefore maintains a memory vector that persists across time steps.

The notation used throughout this lecture is:

- $V_{\phi,\psi}$: the complete vision module.
- $E_\phi$: the encoder, with parameters $\phi$.
- $D_\psi$: the decoder, with parameters $\psi$.
- $m_t$: the memory state at time $t$; this is a state vector, not a separate function $M$.
- $R_\theta$: the recurrent state-transition function, with parameters $\theta$.
- $P_\omega$: the latent prediction head, with parameters $\omega$.

Let $m_t$ be the memory at time $t$. The memory network updates it using the previous memory, the current visual code, and the current action:

$$
m_{t+1} = R_\theta(m_t, z_t, a_t)
$$

In the architecture described in the lecture, the memory has 128 components. It is a vector rather than a literal visual record of every previous frame.

The memory can learn to encode information such as:

- The current position of the ball.
- The direction and approximate velocity of the ball.
- The paddle's recent movement.
- Relevant events from earlier frames, such as a bounce.

The memory starts empty at the beginning of an episode. As the model processes each frame, code, and action, it updates the same memory vector. Later memory states contain information accumulated from the episode history.

## 6. Prediction Component

The updated memory is used to predict the code at the next time step. A prediction network maps the current memory to the next latent representation:

$$
\hat{z}_{t+1} = P_\omega(m_{t+1})
$$

The complete transition process is therefore:

$$
z_t = E_\phi(o_t)
$$

$$
m_{t+1} = R_\theta(m_t, z_t, a_t)
$$

$$
\hat{z}_{t+1} = P_\omega(m_{t+1})
$$

During training, the predicted code is compared with the actual code extracted from the next frame:

$$
\mathcal{L}_{\text{prediction}} = \left\|\hat{z}_{t+1} - z_{t+1}\right\|^2
$$

The vision, memory, and prediction components are trained so that the prediction loss becomes smaller.

The purpose of updating the memory is not merely to store information. The memory is updated because it is needed to predict the next observation.

## 7. Training Through an Episode

For each time step in a training episode:

1. Encode the current frame into $z_t$.
2. Combine $z_t$, $a_t$, and the previous memory $m_t$.
3. Update the memory to produce $m_{t+1}$.
4. Predict the next code $\hat{z}_{t+1}$.
5. Encode the actual next frame to obtain $z_{t+1}$.
6. Compute the difference between the predicted and actual codes.
7. Backpropagate the loss and update the network parameters.

The training objective can be written as:

$$
\mathcal{L} = \sum_t \left\|P_\omega\!\left(R_\theta(m_t, E_\phi(o_t), a_t)\right) - E_\phi(o_{t+1})\right\|^2
$$

At early time steps, the memory contains little information, so prediction error may be high. As the model processes more of an episode, the memory becomes more informative and predictions can improve.

## 8. Recurrent Memory Versus Attention

The memory network is recurrent. It carries one hidden vector forward and updates it at every time step:

$$
m_0 \rightarrow m_1 \rightarrow m_2 \rightarrow \cdots \rightarrow m_t
$$

This is different from an attention-based architecture:

- **Recurrent model:** compresses the past into a persistent hidden state.
- **Attention model:** keeps a collection of past representations and can directly look back at selected earlier frames.

Both approaches can address partial observability, and each has tradeoffs. The architecture in this lecture uses recurrent memory. Other world-model simulators, such as Iris, use transformer-style attention to access past frames more directly.

## 9. Playing with the Trained Simulator

After training, the original game engine and training episodes are no longer required for inference. A person can play the learned simulator using arrow-key actions.

The process is:

1. Start with an initial frame and an empty or initialized memory.
2. Encode the current frame.
3. Provide an action such as left, right, or stay.
4. Update the memory using the code and action.
5. Predict the next code.
6. Feed the predicted code back into the system as the next input.
7. Repeat for subsequent actions.

During training, the model receives real frames at every time step. During autonomous simulation, predicted codes are fed back into the model instead. This creates a closed loop:

$$
\text{action}_t \rightarrow \text{memory update} \rightarrow \hat{z}_{t+1}
\rightarrow \text{next prediction}
$$

The decoder converts predicted codes into frames only so a human can see the simulated game. The decoder is not needed for the core latent-space transition.

## 10. Training Mode and Simulation Mode

The distinction between training and simulation is important.

### Training Mode

- Real frames are available.
- The current frame is encoded into a code.
- The model predicts the next code.
- The prediction is compared with the code from the real next frame.
- The loss updates the model.

### Simulation Mode

- The game engine is turned off.
- A player or agent supplies actions.
- The model updates its memory and predicts the next code.
- The predicted code becomes the input for the next step.
- The decoder renders a predicted frame for visualization.

This is why the trained network can simulate a complete episode without access to the original physics engine.

## 11. Limitations and Practical Details

The predicted frames may not be perfect. The ball can appear dim or slightly elongated, but the overall motion and game behavior can still match the original environment.

Important practical considerations include:

- Every latent dimension should contribute meaningfully rather than allowing some dimensions to become unused.
- The world model must be learnable from the available data.
- The training objective must give enough importance to small but meaningful objects.
- Prediction errors can accumulate when the model repeatedly feeds its own predictions back into itself.

The final quality of imagined gameplay depends on the quality of the learned world model. A poor simulator produces unrealistic dreams, and a planner trained in those dreams may fail when transferred to the real environment.

## 12. World Models and Learning in Imagination

In this lecture, a human supplies actions, so the simulator can be interactively controlled. The larger purpose is to replace the human with a planner.

Once the environment has been modeled, an agent can practice inside it thousands of times:

- The planner proposes an action.
- The simulator predicts the consequence.
- The planner evaluates the imagined result.
- The policy is updated based on the imagined experience.

This is called **learning in imagination** or **learning in dreams**. A robot could learn a task such as folding clothes in a simulator before attempting it in the physical world. The real environment is used only after the policy has become sufficiently capable in simulation.

## 13. Connection to the Agent-Environment Interface

The agent-environment loop now has a learned environment:

- **Agent:** a human or, later, a planner that chooses actions.
- **Environment:** the learned world model.
- **Input to the environment:** the selected action.
- **Output from the environment:** the predicted next observation or latent code.

The loop is:

$$
o_t, a_t \rightarrow \text{world model} \rightarrow \hat{o}_{t+1}
$$

The world model is therefore a simulator that predicts how the environment evolves, rather than a planner that decides which action is best.

## 14. Key Takeaways

1. The first model built in this series is a simulator for Mini Pong.
2. The simulator learns environment dynamics instead of using handwritten physics rules.
3. Random trajectories are sufficient for learning the game dynamics.
4. A VAE compresses each frame into a small latent code.
5. Weighted reconstruction loss prevents small but important objects such as the ball from being ignored.
6. A recurrent memory vector stores information from previous time steps, including hidden information such as velocity.
7. A prediction network uses the updated memory to predict the next latent code.
8. Training compares the predicted next code with the code extracted from the real next frame.
9. After training, predicted codes can be fed back into the model to simulate a complete episode with the engine turned off.
10. The simulator is the environment component; a planner can later use it for learning in imagination.
11. This architecture establishes the foundation for the Dreamer lineage and later world-model simulators.
