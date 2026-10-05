## **How to decide what algorithm to choose in the Model Alignment algorithm:**

> (This is something new bro)

### 1. What data do one have?
- If we have preference data like A is better than B, then DPO or RLHF.
- If one can write a verifiable reward function (something like passes test, correct/incorrect, reaches target; etc), then Reinforce, GRPO, PPO; etc.
- If none, go back to SFT.

### 2. Compute budget (how much compute?)
- PPO is expensive, since requires, Reference Model, Policy Model, Value Model, Reward Model. 4x compute compared to using single model. (Bruh)
- GRPO doesnt require Value model. Reference made in run time, reward required. Almost 2x.
- DPO has no RL loop, same like SFT, 1x.

### 3. What are we applying this for?
- Safety/helpfullness/tone -> DPO or RLHF using human annotator
- Reasoning/Math/Code correctness -> GRPO or PPO with rule based reward (it should be verified)
- Following instructions -> SFT is enough

### 4. How stable it is supposed to be?
- PPO is unstable because of things like Reward hacking by the model. KL Divergence can blow up. Value function can be unstable.
- GRPO is much more stable, but do require proper stable reward function.
- DPO is the most stable, least likely to make issues.
