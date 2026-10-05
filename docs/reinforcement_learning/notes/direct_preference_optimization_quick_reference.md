**The problem with RLHF:**

You want to align a language model to human preferences. You collect preference data — pairs of (chosen, rejected) responses. Standard approach: train a reward model on this data, then run PPO to optimize the language model against the reward model. Three stages, two models, unstable, expensive.

**Why you can't just do gradient descent directly:**

Language generation is discrete — you sample tokens. The reward sits at the end of the full sequence. Gradient cannot flow back through discrete sampling steps. So you need RL.

**DPO's insight:**

The optimal policy under the RLHF objective has a closed form:

π*(y|x) = (1/Z(x)) · πref(y|x) · exp((1/β) · r(x,y))

Rearrange this to solve for r:

r(x,y) = β · log π*(y|x)/πref(y|x) + β · log Z(x)

The reward is now expressed in terms of policy ratios. Plug this into the Bradley-Terry preference model, $P(y_1 > y_2 | x) = \frac{e^{r(x,y_1)}}{e^{r(x,y_1)}+e^{r(x,y_2)}}$. Z(x) appears in both chosen and rejected terms — cancels out. Gone forever.

Result: a loss function purely in terms of policy log probabilities:

Loss = −log σ( β · [log π(y_w|x)/πref(y_w|x) − log π(y_l|x)/πref(y_l|x)] )

**Why this works without RL:**

You already have the chosen and rejected sequences in your dataset. You're not sampling — you're just evaluating log probabilities of fixed sequences. That's fully differentiable. Backprop works. No RL needed.

**What the loss actually does:**

Increases the probability of chosen responses relative to the reference model. Decreases probability of rejected responses relative to the reference model. The sigmoid provides a dynamic weight — learns harder from examples the model currently gets wrong, stops learning once it gets them right. Prevents mode collapse.

**Why it's better than PPO:**

PPO has to approximate an intractable normalization term Z(x) — high variance gradients, instability. DPO absorbs Z(x) analytically. It never appears. Stable training, no reward model, no RL, one stage.