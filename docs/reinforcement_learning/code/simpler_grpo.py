"""
GRPO — how one training step works.

THREE MODELS
    X  live model      trains, gradients ON
    Y  snapshot        frozen copy of X from the last batch. Generates the text.
    Z  reference       frozen copy of X from step 0. Never changes, ever.
    At step 0 all three are identical.

ONE STEP, IN ORDER

1. Generate.
   Y writes 8 completions for a prompt (temperature > 0, so they differ).
   While sampling, save the log-prob of each chosen token -> log_pi_old [8, 500].
   Y did the generating, so those saved numbers belong to Y.

2. Score.
   Reward function grades each completion -> 8 scalars, e.g. [1,0,1,1,0,0,1,0]

3. Advantages.
   A_i = (r_i - mean(r)) / std(r)          -> [8]
   Above the group average = positive. Below = negative.
   One number per completion, shared by all of its tokens.

4. Score the same text with the other two models.
   The tokens are FINISHED now. Nobody generates again. We just ask
   X and Z: "what probability would you have given these exact tokens?"
   One batched forward pass each -- the transformer returns a prediction
   for every position at once, so 500 tokens cost 1 pass, not 500.
       X, grads ON   -> log_pi   [8, 500]
       Z, no_grad    -> log_ref  [8, 500]
   This X pass is the ONLY source of gradients in the whole algorithm.
   Everything else -- advantages, log_pi_old, log_ref -- is a constant.

5. Two leashes, per token.
       ratio = exp(log_pi - log_pi_old)      how far X moved from Y (last batch)
       KL    = exp(d) - d - 1,  d = log_ref - log_pi
                                             how far X moved from Z (since step 0)
   KL is >= 0 always, and is computed only on the chosen token --
   never over the full vocab.

6. Objective, per token.
       J = min(ratio*A_i, clip(ratio, 0.8, 1.2)*A_i) - beta*KL

   A_i        the only term carrying reward signal. Push good tokens up,
              bad tokens down.
   clip       don't move far from Y. min() takes the pessimistic branch,
              so it kills the incentive to keep overshooting, but never
              blocks the correction back.
   -beta*KL   don't drift far from Z. Both leashes only ever restrict;
              neither one rewards change.

7. Loss.
       loss = -mean(J)     over all tokens, all completions
   J is an objective (maximize). Loss is its negative (minimize).
   Big KL -> objective down, loss UP. Sign confusion lives here.

8. Update.
       loss.backward(); optimizer.step()
   Gradient on each token's log-prob is basically -A_i, so this is
   ordinary cross-entropy SFT with each token weighted by its
   completion's advantage. That's the entire mechanism.

9. Next batch.
       Y <- X          after finishing all gradient steps on this batch
       Z               untouched, forever

SHAPES
    log_pi, log_pi_old, log_ref, ratio, KL, J   [8, 500]   per token
    advantages                                  [8]        per completion, broadcast

NOTES
    - Step 0: X == Y, so ratio == 1 and clip does nothing.
      X == Z, so KL == 0. First updates are pure reward-chasing.
    - Mid-training: X and Y stay close (a few tiny steps apart), both
      drift steadily away from Z. So the clip rarely fires; the KL
      penalty grows and pushes back harder as training goes on.
    - If you take only 1 gradient step per batch, ratio is always
      exactly 1 and you can skip computing log_pi_old entirely.
    - With LoRA you don't need a separate Z -- disable the adapters
      and X becomes Z.
"""