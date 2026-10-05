# Mixture of Experts (MoE) Architectures

## Overview
Mixture of Experts (MoE) architectures are a key technique powering many advanced AI models. They enable dramatic scaling of model parameters without slowing down training or inference speed by routing each token to only a sparse set of expert networks.

## The Role of Feedforward Networks (FFN)
In a standard Transformer block, attention mechanisms capture contextual dependencies, while FFNs run independently for each token to retrieve factual information embedded within its weights.
- **Normalization (RMSNorm):** Before the FFN, a normalization layer scales the token vector to unit norm with a learnable scaling factor per dimension to prevent instability.
- **Up-projection ($W_{up}$):** A linear layer maps the input vector to a higher-dimensional space (typically 4x larger). Each learned direction represents a semantic concept or question posed to the token.
- **Activation (e.g., ReLU):** Removes irrelevant semantic matches (negative values).
- **Down-projection ($W_{down}$):** Maps the activated vector back to the original dimension. The column vectors encode factual information corresponding to the concepts selected by the activation. This output is added to the original embedding via a residual connection.

Increasing the hidden dimension of the FFN captures more knowledge but increases computation and memory. MoE solves this by recognizing that for any given token, many hidden units are irrelevant. 

## Sparse Mixture of Experts
MoE partitions a large FFN into several smaller, specialized **expert networks**. For each token, only a sparse subset of experts is activated. This creates a highly expressive network without incuring additional computational costs at test time.

### Router Design
A router predicts which experts to activate for a given token.
1. The router computes a score (logit) for each expert $i$ using learned weight vectors $g_i$. The logit is the dot product:
   $$ h_i(x) = g_i \cdot x $$
2. A softmax function normalizes these logits into probabilities $p_i(x)$:
   $$ p_i(x) = \frac{\exp(h_i(x))}{\sum_{j=1}^N \exp(h_j(x))} $$
3. The router selects the top-$k$ experts (e.g., top 2) with the highest probabilities.
4. The final output is a weighted sum of the chosen experts' outputs, using the routing probabilities as weights:
   $$ y = \sum_{i \in \text{Top-}k} p_i(x) \cdot \text{Expert}_i(x) $$

### Standard Architecture & Expert Variations
- **Standard MoE (Top-2 Sparse MoE):** The most widely adopted and "proper" implementation found almost everywhere (such as Mixtral 8x7B) uses a moderate number of experts (e.g., 8) and routes each token to the Top-2 experts. This strikes a highly reliable balance between expressiveness, performance, and computational overhead.
- **Fine-grained Experts:** Instead of a few large experts, recent advanced models (e.g., DeepSeek V3, Qwen 3) use many smaller experts (e.g., 256), activating a larger subset (e.g., 8). This allows finer granularity.
- **Shared Experts:** A dedicated expert that is always active for every token. It learns broad, common knowledge, allowing routed specialists to focus on nuanced patterns. (Effectiveness is debated; some models show improvements, others do not).

## Token-to-Expert Assignment & Imbalance
Routing is dynamic, which can lead to load imbalances across devices and experts.

### Token Overflow and Dropless MoE
- **Expert Capacity:** $ \text{Expert Capacity} = \frac{\text{Total Tokens in Batch}}{\text{Number of Experts}} \times \text{Capacity Factor} $
- **Token Overflow:** When an expert exceeds its token capacity, overflowing tokens are dropped (computation skipped, passed unchanged via residual connection).
- **Tradeoff:** Increasing the capacity factor reduces overflow but wastes computation/memory on padding empty slots.
- **Dropless MoE:** Avoids this tradeoff by representing expert computation as block-diagonal matrix multiplications. With variable block sizes (depending on the number of tokens routed to each expert), block sparse matrix multiplication kernels process all tokens without dropping or padding.

## Load Balancing
> **Intuition: The "Rich Get Richer" Problem**
> Imagine you manage 8 employees (experts). On day one, they are all equally inexperienced. You assign the first few tasks at random to Employee 1 and 2. Because they practice, they get slightly better. The next round, you naturally assign more tasks to Employee 1 and 2 because they are now your best workers. Fast forward a year, and Employees 1 and 2 are overworked, while Employees 3-8 have done nothing and are completely useless (so-called "dead experts"). Load balancing forces you to distribute work evenly so everyone learns.

Since experts are initialized randomly, unbalanced routing early in training creates a feedback loop: frequently used experts learn quickly and are preferred more, while unused experts become "dead".

### Load Balancing with Auxiliary Loss
To ensure tokens are distributed evenly:
1. **Noisy Top-K Gating:** Adding tunable Gaussian noise to logits encourages diverse routing.
2. **Load Balancing Loss:** Soft constraints are added to the loss function.
   - **Importance:** (Router Probs) Sum of router probabilities for an expert over a batch. A balanced system has a low coefficient of variation (CV) for importance.
     - *Coefficient of Variation (CV):* Quantifies variability by calculating the ratio of the standard deviation ($\sigma$) to the mean ($\mu$), expressed as $CV = \frac{\sigma}{\mu}$. A lower CV indicates expert importance is distributed more evenly. When importance is perfectly uniform, the CV becomes zero.
   - **Load:** (ACTUAL Token count probs) The actual number of tokens assigned to an expert. (Note: CV for importance can be zero even if actual token loads are unbalanced, so load must be measured as well). Since this raw count is non-differentiable, a smooth estimator (probability of selection) is used.
   - The simplified auxiliary load balancing loss forces both the average router probability per expert ($\pi_i$) and the fraction of tokens per expert ($f_i$) to approach $1/N$. It is computed as:
     $$ L_{balance} = \alpha \cdot N \sum_{i=1}^N f_i \cdot \pi_i $$
     where $\alpha$ is a tunable hyperparameter and $N$ is the number of experts. Multiplying by $N$ ensures the minimum possible value of the loss is 1, so the regularization strength doesn't shrink when scaling up the number of experts.

### Auxiliary-Loss-Free Load Balancing (DeepSeek V3 approach)
> **Intuition: The "Line Wait" or "Handicap" Approach**
> Using a traditional Load Balancing Loss is like a boss yelling at you for assigning too much work to Employee 1—you appease the boss, but your company's actual performance drops because you're fighting two competing goals (doing a good job vs. distributing work). Auxiliary-Loss-Free routing solves this organically: If the line for Expert 1 is too long, we artificially add a "wait time" (reduce its routing score). Now, Expert 3's shorter line looks more appealing, and work naturally distributes *without* needing a conflicting penalty in our main loss function.

Auxiliary losses force a tradeoff between model performance and load balance. DeepSeek V3 avoids this:
- Instead of softmax, it uses a sigmoid function on router logits.
- A dynamic, small bias term is added to each expert's logits during top-$k$ routing calculation.
- If an expert is overloaded, its bias is decreased. If under-loaded, its bias is increased.
- The top-$k$ selection uses these bias-adjusted scores, but the final output combination uses the original probabilities, maintaining both load balance and model performance.

## Stabilizing Training with Router Z-Loss
The router uses a softmax function, which is mathematically **shift-invariant**. Adding a constant $c$ to all logits doesn't change the final output probabilities, because the constant factors out and cancels:
$$ \frac{\exp(h_i + c)}{\sum \exp(h_j + c)} = \frac{\exp(c)\exp(h_i)}{\exp(c)\sum \exp(h_j)} = p_i $$

> **Intuition: The "Volume Knob" Problem**
> Softmax only cares about *relative* differences between scores. Imagine adjusting the bass and treble on a stereo—the ratio between them is what matters. If you want the bass to be louder than the treble, you can achieve that by turning the bass slightly up, *or* by turning everything up to absolute maximum volume. While the mathematical *ratio* is identical, blasting the speakers at max volume eventually blows out your physical hardware. In neural networks, "blowing out the hardware" means your underlying logit numbers grow infinitely until they hit a `NaN` software crash from exceeding memory limits (Float16 overflow).

However, this mathematical property creates a massive training pitfall:
- Because the loss depends only on the relative differences in $p_i$, the underlying logits $h_i(x)$ can grow infinitely large without penalty.
- In low-precision architectures (like Float16/Bfloat16), infinitely growing logits quickly overflow the memory limits and cause `NaN` errors, collapsing the training.
- **Safe Softmax** (subtracting the max logit before exponentiating) prevents immediate overflow during calculation, but it doesn't stop the actual logit parameters from continuing to drift upwards forever.
- **Router Z-Loss:** Solves the root cause by explicitly penalizing a large denominator (log-normalizer). It calculates the logarithm of the denominator and penalizes its square. This acts as a gravity well, forcing logits to stay numerically small and centered around zero:
  $$ L_{Z} = c \cdot \log^2 \left( \sum_{i=1}^N \exp(h_i(x)) \right) $$
- Implementing Router Z-Loss prevents gradient explosions and guarantees stable training natively, overcoming the sudden loss spikes common in unregularized MoE training.

## Conclusion
Modern State-of-the-Art MoE training involves three main components:
1. Standard cross-entropy loss for next-token prediction
2. Load Balancing Loss (or auxiliary-loss-free bias adjustment) to prevent dead experts
3. Router Z-Loss to keep training numerically stable

## Related Note

If you are comparing ways to add capacity without scaling the full dense stack, also read [Per-Layer Embeddings (PLE)](./per_layer_embeddings.md). PLE is not sparse expert routing; it is a cheaper token-specific residual path that gives each transformer block its own small embedding slice.

## PyTorch Implementation (with Z-Loss & Load Balancing)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProperMoERouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2, alpha=1e-2, z_loss_coeff=1e-4):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.z_loss_coeff = z_loss_coeff

    def forward(self, x):
        logits = self.router(x)
        
        # 1. Router Z-Loss: c * mean(logsumexp(logits)^2)
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = self.z_loss_coeff * torch.mean(log_z ** 2)
        
        # 2. Routing Probabilities 
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # 3. Load Balancing Loss
        # pi_i (Importance): average router prob per expert
        pi_i = probs.mean(dim=(0, 1)) 
        
        # f_i (Load): fraction of tokens assigned to each expert
        mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        f_i = mask.sum(dim=-2).mean(dim=(0, 1))
        
        # L_balance = alpha * N * sum(f_i * pi_i)
        balance_loss = self.alpha * self.num_experts * torch.sum(f_i * pi_i)
        
        # Normalize top-k probs for output
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices, z_loss, balance_loss
```
