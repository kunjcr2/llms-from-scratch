### He mentioned some of the points to make the training faster. And by faster, he **MEANS** it.

- **The biggest exchange of data and loss of time happens between travel of data and not calculations.**

1. Parameter sharing between Embed and UnEmbed layers.
2. Model weights initialized by having normal STD and Variance. We also do something with NUmber of residual connections \* some penalty. (I dont know why but we do it. For more info, check [video](https://youtu.be/l8pRSuU81PU))
3. Using `torch.cuda.synchronize()` which will wait for GPU to get done with the task before moving ahead. More like async programming. It reduces GPU overload.
4. Using GPUs. Using `Mixed Precision Training`. It means we use `float16` or `bfloat16` instead of default `float32` reducing a large amount of memory and making it faster. Even `TFfloat32` is significantly fast. We use-

   ```python
   from torch.cuda.amp import autocast

   with torch.autocast(device="cuda", dtype=torch.bfloat16):
       logits = model(input)
       loss = loss_fn(logits, targets)

   # We just keep model and loss function in here, not .step or .backward, as it would create issues.
   # We do those in classic FP32 or default.
   ```

5. Now comes the MOST POWERFUL TOOL I HAVE HEARD SO FAR - `torch.compile()`. We do-

   ```python
   model = torch.compile(model) # This is SO OVERPOWERED
   ```

   It fuses the kernals and reduces the Python overload. Before the interpreter converted each line of code and ran it which was pretty messed up, but now with the already compiled model, runs pretty fast as the GPU now knows what code is and it reduces HBM to SM travel. It reduces from about 7-8 travels to just 1-2.

   Makes it 57% faster.

6. Flash Attention. It makes training MUCH faster. Instead of using 4 lines for-

   1. Q@K.T
   2. Masking with -inf
   3. softmax
   4. attn_scores @ V

   Into one single line -

   ```python
   attn_weights = F.scaled_dot_product_attention(q,k,v,is_casual=True)
   ```

   Makes it 27% faster.

7. He also says `nice/ugly numbers`. He mentions a GPU is usally made with bunch of kernals or cores and stuff in multiple of 2. So try getting rid of ugly numbers like 25, 17. 50257; etc. It makes it 4% faster.

8. Gradient clipping is the way to stop the gradient from exploding. We set the clip at lets say 1, so whenever gradient tries to go above, it stops at 1.

- Now the thing is, doing it to one is shit and having 1M graidents in a batch to be 1 is kind of stupid. So instead of gradient, we clip the gradient normalization by `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. The normalization of gradients MUST NOT GO ABOVE 1.

9. We are using Fused Adam Optimzer with some weight decay, making it even a bit faster.

10. We are doing `Gradient Accumulation`. We stop gradient from flowing backward before we are done with a large number of batches. We start with lets say 16 batches and 1024 tokens each. But we wanna do more size of batch - about lets say 0.5M tokens. So we take 0.5M, divide it by 16\*1024 and we get the accum_number. We do accum_numbers of forward pass before doing a backward pass.
