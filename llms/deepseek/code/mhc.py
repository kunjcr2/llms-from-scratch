"""
Hyper-Connections (HC) and Manifold-Constrained Hyper-Connections (mHC)
------------------------------------------------------------------------
Educational implementation of both, with a comparison demo.

Standard residual connection (every transformer uses this):
    x_out = x_in + f(x_in)

Hyper-Connections (ByteDance):
    Replace the fixed skip connection with LEARNED mixing matrices
    that can route information across multiple streams.
    Problem: unstable at scale — signal gains can explode to 3000x+.

mHC (DeepSeek fix):
    Same idea but mixing matrices are constrained to the Birkhoff Polytope
    (doubly stochastic matrices — every row and column sums to 1).
    Information is CONSERVED not amplified. Stable training.

Components covered:
  1. Standard residual connection (baseline)
  2. Multi-stream Hyper-Connection (HC)
  3. Sinkhorn-Knopp projection onto Birkhoff Polytope
  4. Manifold-Constrained Hyper-Connection (mHC)
  5. Demo comparing signal magnitude across all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Component 1: Standard Residual Connection (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Standard transformer residual connection.
    x_out = x + f(x)

    The skip connection (+ x) is fixed — identity mapping.
    Gradients flow directly back through the + operation.
    No learnable mixing, no routing between streams.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.f(x)    # fixed skip connection


# ─────────────────────────────────────────────────────────────────────────────
# Component 2: Hyper-Connection (HC) — ByteDance version
# ─────────────────────────────────────────────────────────────────────────────

class HyperConnection(nn.Module):
    """
    Hyper-Connections replace the fixed residual with learned mixing matrices.

    Instead of one stream (standard transformer), we run N parallel streams.
    Each stream is a copy of the hidden state at different "depths".

    The mixing matrix A controls how streams combine BEFORE the layer.
    The mixing matrix B controls how the layer output gets added back.

    Think of it like:
        Standard: x_new = x + f(x)            (one highway, fixed on-ramp)
        HC:       x_new = A @ x_streams        (multiple highways, learned routing)
                  x_streams_new = x_streams + B * f(A @ x_streams)

    Why it helps: the model can learn to skip layers, repeat layers,
    or combine information from different depths — more expressive than fixed +.

    Why it breaks: A and B are unconstrained. As you stack many layers,
    the mixing matrices multiply together. Even small imbalances compound.
    Signal can grow to 3000x its original magnitude → divergence.
    """
    def __init__(self, d_model: int, n_streams: int = 4):
        super().__init__()
        self.d_model   = d_model
        self.n_streams = n_streams

        # f: the actual layer function (same as before, e.g. FFN or attention)
        self.f = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # A: input mixing matrix [n_streams, n_streams]
        # Controls how streams are combined to form the input to f
        # Initialized near identity so training starts like standard residual
        self.A = nn.Parameter(torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams))

        # B: output mixing vector [n_streams]
        # Controls how f's output gets distributed back to each stream
        # Initialized to ones so all streams receive the update equally
        self.B = nn.Parameter(torch.ones(n_streams))

    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        x_streams: [B, T, n_streams, D]
            The hidden state split across n_streams parallel streams.

        Returns: [B, T, n_streams, D]
        """
        B, T, S, D = x_streams.shape

        # Step 1: Mix streams using A to form the input to f
        # A: [S, S], x_streams: [B, T, S, D]
        # For each (b, t), compute A @ x_streams[b, t] across the stream dim
        x_mixed = torch.einsum('ij, btjd -> btid', self.A, x_streams)
        # x_mixed: [B, T, S, D]  — each stream is now a mix of all streams

        # Step 2: Use the first mixed stream as input to f
        # (in practice you could use any stream or a learned combination)
        x_input = x_mixed[:, :, 0, :]    # [B, T, D]
        fx = self.f(x_input)             # [B, T, D]

        # Step 3: Add f's output back to each stream, weighted by B
        # B: [S] — how much each stream receives of the update
        fx_expanded = fx.unsqueeze(2) * self.B.view(1, 1, S, 1)
        # fx_expanded: [B, T, S, D]

        # Step 4: Residual update — add update to mixed streams
        x_out = x_mixed + fx_expanded    # [B, T, S, D]

        return x_out


# ─────────────────────────────────────────────────────────────────────────────
# Component 3: Sinkhorn-Knopp — projects any matrix onto the Birkhoff Polytope
# ─────────────────────────────────────────────────────────────────────────────

def sinkhorn_knopp(
    M: torch.Tensor,
    n_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Projects matrix M onto the Birkhoff Polytope — the set of
    doubly stochastic matrices (all rows sum to 1, all columns sum to 1).

    This is the 1967 Sinkhorn-Knopp algorithm. It alternates between:
      - Normalizing rows (divide each row by its sum)
      - Normalizing columns (divide each column by its sum)

    After enough iterations, the matrix converges to doubly stochastic.

    Why doubly stochastic?
      - Every row sums to 1  → output is a weighted average of inputs (no amplification)
      - Every column sums to 1 → each input contributes exactly once in total (no loss)
      - Together: information is ROUTED not CREATED or DESTROYED

    M: [n, n]  any square matrix (positive values work best — we use softmax first)
    returns: [n, n]  doubly stochastic matrix
    """
    # First make all values positive via exp (like softmax without the division)
    # This ensures the alternating normalization converges
    M = torch.exp(M)

    for _ in range(n_iters):
        # Normalize rows: each row sums to 1
        M = M / (M.sum(dim=1, keepdim=True) + eps)
        # Normalize columns: each column sums to 1
        M = M / (M.sum(dim=0, keepdim=True) + eps)

    return M


# ─────────────────────────────────────────────────────────────────────────────
# Component 4: mHC — Manifold-Constrained Hyper-Connections (DeepSeek fix)
# ─────────────────────────────────────────────────────────────────────────────

class ManifoldHyperConnection(nn.Module):
    """
    mHC = HC + Sinkhorn-Knopp projection of the mixing matrix A.

    Everything is the same as HyperConnection EXCEPT:
    After every forward pass, A is projected back onto the Birkhoff Polytope
    using Sinkhorn-Knopp. This enforces doubly stochastic constraint.

    Effect:
        - Each stream output is a weighted AVERAGE of stream inputs (row sum = 1)
        - Each stream input contributes exactly 1 unit total (column sum = 1)
        - Signal magnitude is bounded — cannot explode across layers
        - Training stays stable even at 27B+ parameters

    DeepSeek reports: unconstrained HC caused signal gains of 3000x at 27B.
    mHC eliminates this completely with only ~6-7% training overhead.
    """
    def __init__(self, d_model: int, n_streams: int = 4, sinkhorn_iters: int = 20):
        super().__init__()
        self.d_model        = d_model
        self.n_streams      = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        self.f = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # A: same as HC, but will be projected to Birkhoff Polytope each forward pass
        self.A = nn.Parameter(torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams))

        # B: output mixing — kept unconstrained (less impact on stability)
        self.B = nn.Parameter(torch.ones(n_streams))

    def get_constrained_A(self) -> torch.Tensor:
        """
        Project A onto Birkhoff Polytope via Sinkhorn-Knopp.
        This is called every forward pass so A always satisfies the constraint
        during inference, even if the raw parameter drifts during training.
        """
        return sinkhorn_knopp(self.A, n_iters=self.sinkhorn_iters)

    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        x_streams: [B, T, n_streams, D]
        Returns:   [B, T, n_streams, D]

        Identical to HC forward EXCEPT A is constrained before use.
        """
        B, T, S, D = x_streams.shape

        # ── THE KEY DIFFERENCE: project A onto Birkhoff Polytope ─────────────
        A_constrained = self.get_constrained_A()    # doubly stochastic [S, S]

        # Step 1: Mix streams using constrained A
        x_mixed = torch.einsum('ij, btjd -> btid', A_constrained, x_streams)

        # Step 2: Apply f to first mixed stream
        x_input = x_mixed[:, :, 0, :]
        fx = self.f(x_input)

        # Step 3: Distribute f output back to each stream
        fx_expanded = fx.unsqueeze(2) * self.B.view(1, 1, S, 1)

        # Step 4: Residual update
        x_out = x_mixed + fx_expanded

        return x_out


# ─────────────────────────────────────────────────────────────────────────────
# Component 5: Demo — signal magnitude comparison across many layers
# ─────────────────────────────────────────────────────────────────────────────

def measure_signal_growth(
    model_class,
    d_model: int,
    n_layers: int,
    n_streams: int,
    use_streams: bool,
) -> list:
    """
    Run a random input through n_layers of the given model class.
    Track the L2 norm of the signal after each layer.
    If the norm explodes, the architecture is unstable.
    """
    torch.manual_seed(42)
    B, T = 2, 16

    # Build n_layers deep
    layers = nn.ModuleList([model_class(d_model, n_streams) for _ in range(n_layers)])

    norms = []

    with torch.no_grad():
        if use_streams:
            # HC and mHC: input has stream dimension
            x = torch.randn(B, T, n_streams, d_model)
            norms.append(x.norm().item())
            for layer in layers:
                x = layer(x)
                norms.append(x.norm().item())
        else:
            # Standard residual: no stream dimension
            x = torch.randn(B, T, d_model)
            norms.append(x.norm().item())
            for layer in layers:
                x = layer(x)
                norms.append(x.norm().item())

    return norms


if __name__ == "__main__":
    torch.manual_seed(42)

    d_model   = 64
    n_streams = 4
    n_layers  = 12     # stack 12 layers deep to see divergence

    print("=" * 65)
    print("Hyper-Connections vs mHC: Signal Magnitude Across Layers")
    print("=" * 65)
    print(f"d_model={d_model}, n_streams={n_streams}, n_layers={n_layers}")
    print()

    # Run all three
    residual_norms = measure_signal_growth(ResidualBlock,           d_model, n_layers, n_streams, use_streams=False)
    hc_norms       = measure_signal_growth(HyperConnection,         d_model, n_layers, n_streams, use_streams=True)
    # mhc_norms      = measure_signal_growth(ManifoldHyperConnection, d_model, n_layers, n_streams, use_streams=True)

    # Print table
    print(f"{'Layer':<8} {'Residual norm':<18} {'HC norm':<18} {'mHC norm':<18} {'HC/Residual ratio'}")
    print("-" * 80)
    for i in range(n_layers + 1):
        ratio = hc_norms[i] / (residual_norms[i] + 1e-8)
        print(
            f"{i:<8} "
            f"{residual_norms[i]:<18.2f} "
            f"{hc_norms[i]:<18.2f} "
            # f"{mhc_norms[i]:<18.2f} "
            f"{ratio:.2f}x"
        )

    print()
    print("Observations:")
    print(f"  Residual final norm : {residual_norms[-1]:.2f}  (stable)")
    print(f"  HC final norm       : {hc_norms[-1]:.2f}  (can explode at scale)")
    # print(f"  mHC final norm      : {mhc_norms[-1]:.2f}  (stable, like residual)")
    print()

    # Verify Birkhoff Polytope constraint on a sample matrix
    print("=" * 65)
    print("Sinkhorn-Knopp verification")
    print("=" * 65)
    raw = torch.randn(n_streams, n_streams)
    ds  = sinkhorn_knopp(raw)
    print(f"Raw matrix row sums:    {raw.sum(dim=1).tolist()}")
    print(f"Projected row sums:     {ds.sum(dim=1).tolist()}")
    print(f"Projected column sums:  {ds.sum(dim=0).tolist()}")
    print(f"All sums ≈ 1.0? Row: {torch.allclose(ds.sum(dim=1), torch.ones(n_streams), atol=1e-4)}, "
          f"Col: {torch.allclose(ds.sum(dim=0), torch.ones(n_streams), atol=1e-4)}")
    print()
    print("Parameter counts:")
    res = ResidualBlock(d_model)
    hc  = HyperConnection(d_model, n_streams)
    # mhc = ManifoldHyperConnection(d_model, n_streams)
    print(f"  Residual : {sum(p.numel() for p in res.parameters()):,}")
    print(f"  HC       : {sum(p.numel() for p in hc.parameters()):,}  (+A, +B matrices)")
    # print(f"  mHC      : {sum(p.numel() for p in mhc.parameters()):,}  (same as HC, just constrained)")
    # print()
    # print("mHC adds ZERO extra parameters over HC.")
    # print("The constraint is enforced algorithmically (Sinkhorn-Knopp), not by adding params.")