# World Models From Scratch - Lecture 5

## Iris: A Discrete Transformer World Model

Lecture 5 introduces **Iris**, a world model for the CoinRun game. Iris keeps the same simulator objective as the previous lectures:

$$
(o_{\leq t}, a_{\leq t}) \longmapsto \hat{o}_{t+1},
$$

but replaces the recurrent, continuous-latent RSSM with two ideas familiar from language modeling:

- represent each image using a short sequence of **discrete tokens**; and
- predict future tokens with a causal **Transformer**.

The result is an action-conditioned model that can generate sharp imagined game frames. The lecture focuses on the simulator component; training an agent inside the imagined environment is a later planner problem.

## 1. From Recurrent Memory to Attention

The earlier world models use recurrent networks. At every time step, an RNN writes its entire history into one fixed-size hidden vector. This is efficient, but earlier details can fade as more inputs overwrite that vector. In long imagined rollouts, this loss of history contributes to drift and blurry frames.

A Transformer retains tokens from the past and decides which ones matter for the present prediction. At a given position, the model forms a **query**; previous positions provide **keys** and **values**. Query--key similarity produces attention weights, and the resulting weighted combination of values is used to predict the next token.

$$
\operatorname{Attention}(Q,K,V) =
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

In language, this lets a model retrieve a relevant word even when it appeared many tokens ago. In a world model, it can look back through recent frames and actions rather than requiring all relevant information to survive in one repeatedly overwritten memory vector.

### Causal Sequence Modeling

The Transformer is **causal**: while predicting token $x_i$, it may attend only to tokens $x_{<i}$. Its learning objective is next-token prediction,

$$
p(x_{1:N}) = \prod_{i=1}^{N} p(x_i \mid x_{<i}).
$$

This is the same factorization used by an autoregressive language model. Iris makes it applicable to images by first converting images into discrete symbols.

## 2. Continuous and Discrete Latent Spaces

The RSSM uses continuous latent variables. Continuous spaces interpolate smoothly: an intermediate point between two representations also has a meaning. This is useful for representation learning, but it is awkward when the next state has several distinct plausible outcomes.

For example, if a match may be cancelled because of rain, snow, or a strike, a single continuous regression can average those alternatives. In image prediction, averaging visually distinct futures appears as blur.

A discrete latent space instead predicts a categorical distribution over a finite vocabulary:

$$
p(z_t \mid \text{history}) \in \Delta^{K-1}.
$$

The model can assign probability to several possibilities and then sample or select one concrete token. It does not need to output an average between mutually exclusive visual states. This is why discrete prediction can preserve sharper images while still representing uncertainty.

## 3. Tokenizing an Image with a VQ-VAE

CoinRun frames are arrays of thousands of pixel values, not pre-existing words. Iris learns a visual vocabulary with a **vector-quantized variational autoencoder (VQ-VAE)**.

1. An encoder maps a frame $o_t$ to a spatial grid of continuous feature vectors.
2. Each feature vector is replaced by its nearest entry in a learned **codebook**.
3. The selected codebook indices form the discrete visual-token sequence.
4. A decoder reconstructs the original frame from the quantized vectors.

For the lecture setup, the encoder produces a $4 \times 4$ grid. Therefore one frame becomes 16 token IDs. The codebook has 512 learned entries, so each token is an integer in a vocabulary of size 512:

$$
o_t \xrightarrow{E} z_{t,1:16}, \qquad z_{t,j} \in \{1,\ldots,512\}.
$$

The codebook is not a hand-written collection of image patches. Its vectors are learned jointly with the encoder and decoder so that combinations of 16 entries can reconstruct the game frames.

### Vector Quantization

Let $e_j$ be an encoder output for one grid cell and let $c_k$ be codebook entry $k$. Quantization chooses the nearest entry:

$$
z_j = \arg\min_k \lVert e_j - c_k \rVert_2^2,
\qquad \tilde e_j = c_{z_j}.
$$

The decoder receives $\tilde e_{1:16}$ rather than the original continuous features. Training contains a reconstruction term that makes the decoded frame resemble the input, plus codebook/commitment terms that bring encoder features and their selected codebook entries together. Thus the forced discretization remains informative rather than arbitrarily throwing image information away.

## 4. Iris Dynamics Model

Once every frame is a sequence of 16 symbols, world modeling becomes a token-prediction problem. A causal Transformer receives the discrete tokens from recent frames together with the associated action tokens, and outputs a distribution over the 512 possible next visual tokens.

With a history of eight frames, the visual context contains

$$
8 \times 16 = 128
$$

image tokens. Including eight action tokens yields 136 input tokens in the lecture example. The Transformer generates the next frame's 16 tokens autoregressively, one at a time. The VQ-VAE decoder then turns those token IDs back into a predicted game image.

At a high level:

$$
\text{frame} \rightarrow \text{16 visual tokens}
\rightarrow \text{Transformer conditioned on actions}
\rightarrow \text{next 16 tokens}
\rightarrow \text{decoded frame}.
$$

Actions are essential: a world model is not merely a video predictor. It must predict how the environment changes in response to the agent's choices, such as moving right or jumping in CoinRun.

## 5. Iris Components

Iris has three conceptual components:

1. **Tokenizer (VQ-VAE):** converts a frame to 16 discrete codebook indices and decodes indices to a frame.
2. **Dynamics model (causal Transformer):** predicts the next visual token distribution from past visual tokens and actions.
3. **Agent/planner:** can be trained in the imagined environment, but is outside this lecture's simulator-focused scope.

During a rollout, generated visual tokens become part of the context for the next prediction. The model can therefore simulate an action-conditioned CoinRun trajectory after an initial observed context.

## 6. Position in the World-Model Design Space

Two design choices distinguish the models discussed so far:

| Sequence model | Latent space | Example |
| --- | --- | --- |
| Recurrent network | Continuous | RSSM / Dreamer-style model in Lecture 4 |
| Recurrent network | Discrete | DreamerV2-style design |
| Attention / Transformer | Discrete | Iris |
| Attention / Transformer | Continuous | Later Transformer-based world models |

Iris is the opposite corner from the Lecture 4 RSSM: it uses attention instead of recurrence and a discrete codebook instead of a continuous stochastic latent. These choices are not a claim that one family is always superior; they make different trade-offs in memory, scalability, uncertainty modeling, and image fidelity.

## 7. Why Iris Can Produce Sharp Rollouts

- The VQ-VAE compresses each frame into a compact, learned discrete vocabulary.
- The Transformer can retrieve useful evidence from the token history through attention rather than relying only on one recurrent summary.
- A categorical prediction represents alternative next visual elements as probabilities over tokens instead of averaging them in pixel or continuous-latent space.
- Decoding a concrete sampled/selected token sequence produces a concrete image, helping avoid the averaged appearance associated with blurry predictions.

Discrete tokens do not eliminate all world-model error: poor tokenization, limited context, or inaccurate action-conditioned dynamics can still cause rollout mistakes. They provide a particularly natural interface between visual environments and autoregressive Transformer modeling.

## 8. Key Takeaways

1. Iris formulates action-conditioned visual simulation as autoregressive sequence modeling.
2. Attention preserves access to past tokens and can focus on relevant history, unlike a recurrent model that continually rewrites one memory vector.
3. A VQ-VAE converts a CoinRun frame into 16 discrete tokens selected from a learned codebook of 512 entries.
4. Discrete categorical prediction can represent multiple plausible futures without producing their visual average.
5. The causal Transformer consumes past frame tokens and actions, then predicts the next frame's tokens one at a time.
6. Decoding predicted tokens produces the next imagined observation; this completes the world-model simulator loop.
