# World Models From Scratch - Lecture 12

## V-JEPA 2 and V-JEPA 2-AC

Lecture 12 explains how V-JEPA 2 scales video representation learning and how it is adapted into an action-conditioned world model for robot planning.

## 1. Scaling V-JEPA

V-JEPA 2 scales the original V-JEPA along four dimensions:

1. **Data:** about 22 million video samples instead of about 2 million.
2. **Model:** roughly one billion parameters instead of about 250 million.
3. **Resolution:** training progresses from 224 to 448 pixels.
4. **Training duration and frame count:** longer training with clips containing up to 64 frames.

The architecture remains a JEPA-style masked latent predictor. It uses 3D rotary position embeddings so the model can represent both spatial and temporal locations.

The goal is to learn general visual and physical structure from video: motion, temporal coherence, appearance, and basic physics. Unlike action-conditioned models trained only on task data, V-JEPA 2 can learn from large-scale videos without action labels.

## 2. V-JEPA 2-AC

V-JEPA 2 is converted into **V-JEPA 2-AC** by freezing the pretrained encoder and fine-tuning a predictor with robot data. Actions and proprioceptive measurements are supplied as the latent conditioning variable. The predictor then estimates the next visual representation conditioned on the current state and action.

The training losses include:

- **Teacher-forcing loss:** compare each predicted next representation with the true next representation.
- **Rollout loss:** feed predictions back into the predictor and compare a multi-step imagined rollout with the true sequence.

Rollout loss matters because deployment is autoregressive. During real planning, later predictions depend on earlier predictions rather than on ground-truth frames.

## 3. Planning in Representation Space

V-JEPA 2-AC uses model predictive control, similar to DINO-WM:

1. Encode the current observation.
2. Sample candidate action sequences.
3. Predict the resulting future representations.
4. Compare imagined states with a goal representation.
5. Execute the best action sequence and replan.

The system was fine-tuned on about 62 hours of robot data and demonstrated tasks such as reaching and pick-and-place. The pretrained video model supplies broad visual and physical knowledge, while the action-conditioned predictor learns how the robot's actions change its state.

## 4. Relation to DINO-WM

Both systems predict in latent space and use MPC. DINO-WM uses a separately pretrained, frozen DINOv2 encoder and trains a dynamics transformer. V-JEPA 2 is trained end-to-end as a masked predictive video model first, then its encoder is reused for action conditioning. JEPA's architecture also makes it natural to replace target-location conditioning with action and proprioceptive conditioning.

## 5. Limitations

V-JEPA 2-AC still has important limitations:

- MPC is slow because it must simulate many candidate trajectories.
- The architecture does not yet provide a strong hierarchy for reasoning at multiple time scales.
- Language conditioning is not included.
- Action-conditioned fine-tuning still requires robot interaction data.

The main result is that a world model pretrained on large-scale video can be adapted to robot control efficiently: video provides general physical knowledge, and a smaller action-conditioned training set teaches the model how a particular robot acts.
