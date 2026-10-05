<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

## Do We Skip Transformer Blocks Entirely in Mixture-of-Depths (MoD)?

No, the Mixture-of-Depths (MoD) technique does not skip **all** transformer blocks for any given token, but rather allows for tokens to **selectively bypass individual blocks** within the network.

### How Skipping Works in MoD

- **Per-block Routing:** At each layer (or block), a routing mechanism assigns a score to every token, determining its relevance for further processing. Only the top-scoring (top-k) tokens at each layer are routed through the full computations (self-attention and MLP). The other tokens are sent through a residual (skip) connection, effectively **bypassing that specific block’s expensive computations** while carrying forward their current representations[^1][^2][^3].
- **Residual Path:** Skipped tokens are not processed by the active computations of that layer; instead, they pass directly to the next block through the residual connection, unchanged. This saves computational resources and accelerates inference/training.
- **Dynamic but Not Absolute:** The skipping is dynamic and adaptive per token and per layer—some tokens may process through most layers, others may skip many. However, no token needs to traverse all blocks by default, and no block is skipped for all tokens. This enables the model to maintain capacity for complex tokens while saving compute on easier ones[^1][^4][^5].


### Extreme Cases

- **All Tokens Routed:** If every token is chosen by the router at every block, MoD reduces to a standard transformer.
- **All Tokens Skipped:** If the router skips every token at every block, the model would skip all learning capacity and fail. In practice, this never happens as it's not useful—the optimal balance is somewhere in between[^1][^3].


### Empirical Findings

- In aggressive settings, MoD can skip a significant portion of blocks for most tokens: some reported configurations process only 12.5% of tokens through certain blocks, meaning 87.5% of tokens entirely bypass those blocks at each corresponding depth[^4][^2].
- This selective mechanism reduces FLOPs while maintaining or even improving model performance compared to full, dense transformers[^1][^2].


### Summary Table

| Aspect | Mixture-of-Depths (MoD) Behavior |
| :-- | :-- |
| Skip All Blocks? | No: Skipping is dynamic and per-token[^1][^3] |
| Skip Some Blocks? | Yes: Many tokens skip many blocks[^4][^2] |
| How is skipping chosen? | By learned, dynamic router at each block[^1][^5] |
| Always skip same blocks? | No: Pathways are adaptive for each token[^5] |
| Any residual learning? | Yes: Tokens that skip continue via residuals |

### Conclusion

The MoD approach introduces *dynamic, token-level skipping* of transformer blocks, but never skips all blocks for all tokens. Instead, it adaptively routes tokens through the layers necessary for effective learning and inference, optimizing efficiency without sacrificing performance[^1][^4][^2][^5][^3].

<div style="text-align: center">⁂</div>

[^1]: https://arxiv.org/html/2404.02258v1

[^2]: https://the-decoder.com/googles-mixture-of-depths-uses-computing-power-more-efficiently-by-prioritizing-key-tokens/

[^3]: https://ajithp.com/2024/04/07/mixture-of-depths-the-innovative-solution-for-efficient-and-high-performing-transformer-models/

[^4]: https://inwaves.io/mixture-of-depths/

[^5]: https://graphcore-research.github.io/mixture-of-depths/

[^6]: https://www.reddit.com/r/MachineLearning/comments/1bvrduw/r_deepmind_mixtureofdepths_dynamically_allocating/

[^7]: https://www.clioapp.ai/research/mixture-of-depths

[^8]: https://arxiv.org/html/2506.21103

[^9]: https://openreview.net/forum?id=jIAKjjEmWi

[^10]: https://www.linkedin.com/pulse/topic-18-what-mixture-of-depths-theturingpost-ioj4f

[^11]: https://arxiv.org/abs/2404.02258

[^12]: https://www.turingpost.com/p/mod

[^13]: https://news.ycombinator.com/item?id=39960717

