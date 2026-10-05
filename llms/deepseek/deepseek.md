> # How was DeepSeek built ?

---

## What is DeepSeek ?

1. Lec-1 was basically what is Deepseek and what we will be looking at in the course.
2. Lec-2 is about why is it special with MoE, MTP, MLA, Quantization, RoPE and all that. We also talked about what Reinforcement learning. Instead of human based labeled data, they used RL which basically taught the model to do complex reasoning. RL is basically rule based reward system.

---

## Starting now with DeepSeek

3.  Lec 3, 4, 5, 6, 7 and 8 are just Multi Head Attention Mechanism.
4.  `Key-Value cache` or `KV Caching` (Popular term). Lets say we have a next token generation task.

    ```
    a. AB -> Inference -> C

    b. ABC -> Inference -> D

    c. ABCD -> Inference -> E
    ```

    - So here what happens is that, we have to calculate K, Q, V, attn scores and attn weights, in transformer block for AB, thrice; ABC, twice. Which is basically waste of computation.
    - Also, as we know, to predict last token, we need the context vector of the last token ONLY. Lets assume that last token embeddings to be X.

      ```
      X*Wq=Q, X*Wk=K, X*Wv=V
      ```

      - But now the the thing is We are getting K of last token. What about others. Thats where we get the remaining K and V matrices from cache which we already have.

      - So we take that cached K and V and use it to get the attn scores, attn weights and ahead with not having any sort of redundant calculations.

      > - We basically need Context vector of last token, which can be recieved with attention weight of last token and Value matrix. We get Value vector of last token from XxWv and all we do is add other Value matrix parts from cache. Same while getting attn weights, we multiply Query vector with transpose of Key matrix. Where we get Key's final vector from XxWk and the rest from cache.

      - But issue with that was, it takes too much of space.

      ```
        size of KV cache = l*b*n*h*s*2*2
        where,
        l is number of transformer blocks
        b is batch size
        n is numebr of heads
        h is head dimension
        s is context length
        2 is one for Key and one for Value
        2 is float16 bit number = 2 byte
      ```

      and this leads to Deepseek being 400GBs at storing KV cache in memory and hence, this method is sh\*t.

5.  We use `MultiQuery attention`, where we make all the attention heads of keys an values, SAME.

    - In `Normal attention`, we have multiple attention heads for a single Key or Value matrix, and all of them are different but in `MultiQuery attention` make all of them same so that we ONLY have to save that single head which can be concated and used as an entire Key Value matrix.
    - Plus point of this is, we reduce 400 GB of KV Caching, to 3 GB by not saving all the attention heads.
    - Minus point is we loose performance as multiple heads means multiple perspective but now only 1.
    - Here, KV Cache formula is reduced to `l*b*h*s*2*2` as we dont care about number of heads anymore.

6.  In `Grouped Query Attention`, it is somewhere between `MHA` and `MQA`. `MHA` have all the K and V heads as distinct matrices, leading to KV Caching memory issue. While in `MQA`, they ALL are same, leading to loss in accuracy.

    - what we do here in `GQA`, is we make small groups of heads, for example out of 64 heads, we make 8 groups of 8 heads. Now in 8 groups, each group have distinct K-V matrices but within each group, all the heads have common K-V matrices.

    ```
    [1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4] - 4 groups with 4 heads each
    ```

    Above each 1, 2, 3, 4 are K-V amtrices which are same within the group but distinct for each group.

    - Our KV Cache formula is now `l*b*g*h*s*2*2` where g is just number of groups.
    - This leads to memory requirement of GPT-3 which is,

    ```
    MHA - 4.5 GBs
    MQA - 48 MBs
    GQA - 384 MBs (which is not much but does its job)
    ```

    - Better performance than `MQA`, but as not good as `MHA`.

> ## And that is where deepseek comes in which solved the problem using `Multihead Latent Attention`, where they got K-V memory requirement to low and Performance at High.

[Look at this video for everything TILL MLA attention](https://youtu.be/2TT384U4vQg)

### 7. **`Multihead Latent Attention`**

- Alright. So what we did here is,
  1. We take the input embeddings `X`. We do `X*Wq` to get `Q` vector.
  2. We do `X*Wdkv`, to get a Latent matrix `Ckv` which is to be cached, where `Wdkv` is a down projection of input into cached latent matrix.
  3. We do `Ckv*Wuk` to get `K` vector, where `Wuk` is up projection of Key.
  4. We do `Ckv*Wuv` to get `V` vector, where `Wuv` is up projection of Value matrix.
  5. Then getting attn scores, attn weights and finally context vector.

```
context_vector
    = attn_weights * V
    = Q*K.T*V
    = (X*Wq)*(Ckv*Wuk)*(Ckv*Wuv)
    = (X*Wq)*(X*Wdkv*Wuk).T*(X*Wdkv*Wuv)
    = (X*Wq*Wuk.T) * (X*Wdkv).T * (X*Wdkv) * Wuv
and that's exactly where we get 'Absorbed Query'.
```

- Basically instead of caching both, Keys and values, we cache a common matrix from where we can have both Keys and Values vector.
- PLus point is, the formula now becomes,

```
Size of cache = l*b*nl*s*2
    where,
    l is transformer blocks
    b is batch numbers
    nl is dimension of cached matrix
    s is context length
    2 is 2 bytes = 16 bits
```

- Basically, deepseek becomes 6GBs rather than those big numbers. Now, the memory issue is solved. Coming to perfomance. All the `Wuk` and `Wuv` are there already, FIXED, but cached `Ckv` comes in everytime frmo cache and is being updated everytime leading to distinct heads and hence better performance.
- For visual purposes, `./assets/MultiHeadLatentAttention.png`

8. We have started the `positional encoding`. Starting with simple `integer encoding`, but it had so many problems like variance not being in center and such so we dont talk about that. Then comes `binary encodings` where lower bits changes faster and upper ones, slower and they are just 1s and 0s and LLM's backpropogation get a tough challange of doing those jumps and hence we come across `Sinosuedal Encodings`, which is `binary encoding` but continuos.
   - In the next lecture we talk about Sinosuedal encodings. It is mentioned in `Attention is all you need` and there are formulas in it. We basically get the encoding for `p+k` position from `p`th encoding of any index, by just rotating the vector in (for example) 2D space.
   - In the last lecture of encodings, we talked about `Rotatory encodings`. Here we basically use the encodings inside the multi headed latent attention block and especially query and keys matrices instead of in the data preprocessing pipeline as while adding a potional encoding to token embedding makes it loose the magnitude of the vector. so to conserve the magnitude, we change the angle of the vector.
   - We breakdown (for example) context length of 1024 and embedding dimension of 4 token, into 2 sets of 2 numbers. if embedding of a word "hello" is [x,y,z,w], we make it xy and zw and change the angle of that vector and replace older vector with newer one where magnitude is not changed but the vector is based on the encodings.
   - We change the angles based on the paper of "attention is all you need" and it is basically sin and cos functions so the positions which are near, the semantics changes faster and hence near words can have better input embeddings and furthur positions doesnt have bigger changes and hence they do take information till far whcih is something very cool.
   - MAGNITUDE DOESNT CHANGE, VECTOR ANGLE DOES.

> ## 9. This is where the heart of deepseek lies. `MHLA + RoPE`.

- We basically have to apply `RoPE` to the query in `MLA` but when we do so directly, we loose absorbed query and hence leading to calculating Keys matrix EVERYTIME which is a lots of computation as before, we were absorbing `W_uk` and hence `K.T` was coming from cache only but now we have to calculate it.
- To get rid of it, we work i around by dividing query and keys into two parts and what do we do with two parts ? Go and look at `./assets/MLA+RoPE.png` as well as `Page 6-8 in ./deepseekV3.pdf`.

10. `Mixture of Experts` or `MoE` is now the second founding of deepseek team. They insteead of having a dense layer in FFNN, broke it down into smaller experts of much smaller dimension. What happens is, if a token comes in and it is a verb (for example), then only the expert who takes care of verbs, activates and the token procceds ahead. Instead of entire NN to run, there are just 1-2 experts which runs and this reduces efficiency cost and increases speed of training and inference.

- Read more about it from the deepseek pdf.
- We aplly something that we call as sparsity. Which is the method of breaking down the dense layer neural network into trainable experts.
- usually we hve to take something we call it as `Sparsity deciscion` or `Load balancing` which means the deciscion of WHAT experts to choose and move ahead with (for ex. 2 out of 64) and the process of doing so; respectively.
- We have another question of, lets say we choose 2 experts, expert A and expert B, then how much attention to be given to what expert is called `Routing`. There is a `Routing matrix (embed_dim, num_experts)`, and we do

```
inputs(n_tokens, emb_dim) * Routing matrix(emb_dim, n_experts) = Experts selector matrix(n_tokens, n_experts)
```

For ex. We are choosing 2 out of 3 experts and there 4 tokens. the ES matrix is as:

```
[1,2,3],
[5,2,4],
[9,0,4],
[8,2,0]
```

- We put -inf to remaining lower values by keeping 2 max values as it is.

```
[-inf,2,3],
[5,-inf,4],
[9,-inf,4],
[8,2,-inf]
```

- Hit it with softmax

```
[0,0.36,0.64],
[0.55,0,0.45],
[0.78,0,0.22],
[0.9,0.1,0]
```

- This means that expert1, expert2 and expert3 should be given respective attention based on each the token. And at the end we just `multiply the weight factors` and `adding the expert's output`.

11. The issue is now that, if some experts are used more, then there is an issue of imbalance in experts where some experts are overly used and some underly. For that we have `Auxiliary loss`. It is basically added to training loss to penalize imbalance expert selection, pushing the routing function towards a uniform distribution.

    - We start with calculating `Expert importance`, which is
      sum of each column in ES weight matrix for [E1, E2, ... En]; respectively.
    - Loss would be high if there is high coefficient of variation between expert importance (that we calculated).

    ```
    Coefficient Variation(CV) = Standard Deviation / Mean
    ```

    And this is where we get auxiliary loss,

    ```
    Auxiliary Loss = lambda*(CV)^2
    ```

    And this will be added to LLM training. When the params are being trained, means, the expert importance is coming together, means SD is reducing and mean is getting centered, leading to lower CV and hence lower Auxiliary loss.

    - Now, another thing, `Load balancing`, as the issue is, expert importance is not same as equal token importance. We want to have each expert, having balanced tokens routed, so like we dont want something where an expert have 1 token with high confidence routed but other expert have 4 tokens with lower confidence.
    - First we need to have expert probablity(Pi) of it being selected based on its importance.

    ```
    P1 is probability that Router will select E1. (E1 Importance / sum importance)
    P2 is probability that Router will select E2. (E2 Importance / sum importance)
    P3 is probability that Router will select E3. (E3 Importance / sum importance)
    And so on...
    ```

    And also, another term `Fraction of tokens routed`,

    ```
    f1 is fraction of tokens routed to E1. (Tokens routed to E1 / n_tokens)
    f2 is fraction of tokens routed to E2. (Tokens routed to E2 / n_tokens)
    f3 is fraction of tokens routed to E3. (Tokens routed to E3 / n_tokens)
    And so on...
    ```

    And finally we get,

    ```
    Load balance Loss = Scaling factor * n_experts * sum(fi*pi)
    ```

    - Basically, now experts with more importance to handle proportionally more tokens while the experts with lower importance to handle less tokens which basically descreases mismatch. This reduces overall imbalance.
    - To avoid the experts shutting off, we can use `capacity factor`, We have smoething we call as expert capacity,

    ```
    Expert Capacity = Tokens per batch/n_experts * Capacity factor
        where,
        Tokens per batch = batch_size * context length * top_k
            where,
            top_k is number of experts chosen for each token.
    ```

    Means how many maximum tokens can be routed to an expert.

12. There are `3 innovation` that Deepseek themselves did,

    - The first one being `Auxililary loss free load balancing`. The issue with scaling factor in load balancing is that, if its low, its negligible and we wwill have issues in load balancing across the experts while its high than we will have issues with training loss being very high elading to inefficient backpropogation. And hence they got rid of loss.
      - First we find average token load per expert (total experts used or total tokens routed / num_experts)
      - We see from that if the expert is 'Underloaded' or 'Overloaded'. And from there we calculate `load violation`.
      - And now we have `bias=0` for each expert and it is updated like,
      ```
      bi = bi + u * sign(load violation error)
      where,
          u is predefined constant
          sign(load violation error) is just +/- for load violation.
          bi is bias for expoert i
      ```
      - And finally we add biases (or subtract, depends), to `Expert Selector Matrix`.
      ```
      Underloaded -> add bias -> increase Pi of being chosen
      Overloaded -> reduce bias -> decreaase Pi of being chosen
      ```
    - The Second innovation in `Shared experts`. We had issues of `Knowledge hybridity` and `Knowledge Redundancy` (Read those from DeepseekMoE paper), and that stopped us from having Specialized experts, which they wanted.
      - To get rid of second issue of knowledge redundancy, they had 2 sorts of experts. First is `Shared experts` and second is `Routed Experts`. Here, shared are ALWAYS ACTIVATED, for EACH TOKEN. And on the other hand, routed ones are selected through experts selctors matrix. This way we have redundant stuff on shared experts and then routed experts can do specilized tasks. And then the outputs are added together to have MoE output.
    - And finally, the last innovation is `Fine grained Expert segmentation`, we revoke the issue of `Knowledge hybridity`. We basically convert hidden layer of 4096 neurons in 4 parts with 1024 neurons but now we have 64 experts with 64 neurons each. SIMPLE. So that every expoert can become super specialized experts and can focus on single thing.

13. And finally the code. Go to `./Codes/deepseek_moe.ipynb`

# Code challange: COMPLETE ENTIRE DEEPSEEK ARCHITECTURE

## Do it in `./Codes/deepeseek_complete.ipynb`

# Multi-Token predicition in a seprate file - `Multi-token prediction`.
