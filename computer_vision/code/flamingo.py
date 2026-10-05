"""
flamingo_educational.py

A simplified, educational Flamingo-style multimodal model.
Goal: understand HOW vision tokens flow into a frozen LLM via gated cross-attention.
This is NOT production code. Every design choice is made for readability.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPVisionModel

# ──────────────────────────────────────────────
# STEP 1: PERCEIVER RESAMPLER
# Problem: CLIP gives us a variable number of patch tokens (e.g. 196 for ViT-B/32).
# We want a fixed, small set of visual tokens to feed into the LLM.
# Solution: Learn a fixed set of "query" vectors that cross-attend to the patches.
# Result: No matter the image size, we always get `num_latents` visual tokens out.
# ──────────────────────────────────────────────

class PerceiverResampler(nn.Module):
    def __init__(self, dim=768, num_latents=64):
        super().__init__()
        # These are the learned query vectors — they "ask questions" of the image patches
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        # One cross-attention layer: latents query the image patches
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, image_features):
        # image_features: (B, num_patches, dim) — raw CLIP patch embeddings
        B = image_features.shape[0]
        # Expand the learned latents for the whole batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, num_latents, dim)
        # Latents ask questions (query), image patches provide answers (key, value)
        out, _ = self.cross_attn(query=latents, key=image_features, value=image_features)
        return self.norm(out)  # (B, num_latents, dim)


# ──────────────────────────────────────────────
# STEP 2: GATED CROSS-ATTENTION BLOCK
# Problem: We need to inject visual context into the LLM without destroying its
#          pre-trained text knowledge.
# Solution: A cross-attention layer gated by tanh(alpha).
#           At init, alpha=0 → tanh(0)=0 → the gate is closed → LLM is unchanged.
#           During training, alpha slowly opens and lets vision in.
# ──────────────────────────────────────────────

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        # The gate. Starts at 0 → output is identity → safe initialization.
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_hidden, vision_tokens):
        # text_hidden:   (B, seq_len, dim) — current LLM hidden states
        # vision_tokens: (B, num_latents, dim) — output of PerceiverResampler

        # Text tokens ask: "what in the image is relevant to me?"
        attn_out, _ = self.cross_attn(
            query=self.norm(text_hidden),
            key=vision_tokens,
            value=vision_tokens,
        )
        # Gate controls how much visual info leaks in. tanh keeps it in [-1, 1].
        gate = torch.tanh(self.alpha)
        return text_hidden + gate * attn_out  # residual add


# ──────────────────────────────────────────────
# STEP 3: THE FLAMINGO MODEL
# Wires everything together.
# Frozen: vision encoder, LLM weights
# Trainable: perceiver, gated cross-attention blocks
# ──────────────────────────────────────────────

class SimpleFlamingo(nn.Module):
    def __init__(self, cross_attn_freq=4):
        """
        cross_attn_freq: inject a gated cross-attn block every N LLM layers.
        4 means layers 0, 4, 8 get visual context. Reduce for more vision influence.
        """
        super().__init__()

        # --- Frozen vision encoder ---
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # --- Trainable perceiver ---
        self.perceiver = PerceiverResampler(dim=768, num_latents=64)

        # --- Frozen LLM (GPT-2 small for education) ---
        self.llm = AutoModelForCausalLM.from_pretrained("gpt2")
        for p in self.llm.parameters():
            p.requires_grad = False

        # --- Trainable gated cross-attention blocks ---
        # One block per injection point (every cross_attn_freq layers)
        n_layers = self.llm.config.n_layer  # 12 for GPT-2
        self.cross_attn_freq = cross_attn_freq
        num_xattn_blocks = n_layers // cross_attn_freq
        
        self.gated_xattn_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(dim=768) for _ in range(num_xattn_blocks)
        ])

    def forward(self, pixel_values, input_ids):
        """
        pixel_values: (B, 3, H, W) — preprocessed image tensor from CLIP processor
        input_ids:    (B, seq_len) — tokenized text
        Returns:      logits (B, seq_len, vocab_size) for computing loss
        """

        # ── Vision pathway ──────────────────────────────────────────────
        # 1. Frozen CLIP extracts patch features
        with torch.no_grad():
            image_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
            # image_features: (B, 197, 768)  [1 CLS + 196 patches for ViT-B/32]

        # 2. Perceiver compresses to fixed-size visual tokens
        vision_tokens = self.perceiver(image_features)
        # vision_tokens: (B, 64, 768)

        # ── Language pathway ─────────────────────────────────────────────
        # 3. Embed text tokens
        hidden = self.llm.transformer.wte(input_ids)                            # (B, seq, 768)
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = hidden + self.llm.transformer.wpe(pos_ids)                     # add positional emb

        # 4. Walk through LLM layers, injecting vision at every cross_attn_freq layers
        xattn_idx = 0
        for layer_idx, llm_layer in enumerate(self.llm.transformer.h):

            # Inject vision context before this layer?
            if layer_idx % self.cross_attn_freq == 0 and xattn_idx < len(self.gated_xattn_blocks):
                hidden = self.gated_xattn_blocks[xattn_idx](hidden, vision_tokens)
                xattn_idx += 1

            # Frozen LLM self-attention + FFN
            hidden = llm_layer(hidden)[0]

        # 5. Final norm + project to vocab
        hidden = self.llm.transformer.ln_f(hidden)
        logits = self.llm.lm_head(hidden)  # (B, seq, vocab_size)
        return logits


# ──────────────────────────────────────────────
# STEP 4: TRAINING LOOP
# Standard autoregressive language modeling loss.
# We predict the NEXT token at each position — classic causal LM objective.
# The image is the "context"; the text is what we train the model to generate.
# ──────────────────────────────────────────────

def train(model, dataloader, num_epochs=3, lr=1e-4):
    """
    Minimal training loop.

    Dataloader should yield batches of:
        pixel_values: (B, 3, 224, 224)
        input_ids:    (B, seq_len)  — e.g. "<image> A photo of a cat sitting on a mat"

    Loss: cross-entropy on next-token prediction.
    Only the Perceiver + GatedCrossAttentionBlocks get gradients.
    """
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"]  # (B, 3, 224, 224)
            input_ids    = batch["input_ids"]      # (B, seq_len)

            # Forward pass
            logits = model(pixel_values, input_ids)
            # logits: (B, seq_len, vocab_size)

            # Autoregressive loss:
            # At position t, predict token t+1.
            # So we shift: input is [0..T-1], targets are [1..T]
            shift_logits = logits[:, :-1, :].contiguous()   # (B, seq-1, vocab)
            shift_labels = input_ids[:, 1:].contiguous()    # (B, seq-1)

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # (B*(seq-1), vocab)
                shift_labels.view(-1),                          # (B*(seq-1),)
            )

            # Backward pass — only trainable params get updated
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"── Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

    print("Training done.")


# ──────────────────────────────────────────────
# QUICK SMOKE TEST (no real data needed)
# Verifies shapes are correct end-to-end.
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Building model...")
    model = SimpleFlamingo(cross_attn_freq=4)

    # Dummy batch — replace with a real CLIP-preprocessed image + tokenized caption
    B, seq_len = 2, 20
    dummy_pixels   = torch.randn(B, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 50257, (B, seq_len))

    print("Running forward pass...")
    logits = model(dummy_pixels, dummy_input_ids)
    print(f"Logits shape: {logits.shape}")  # expect (2, 20, 50257)

    # Dummy training loop smoke test (1 step)
    print("\nSmoke-testing training loop...")
    dummy_dataloader = [{"pixel_values": dummy_pixels, "input_ids": dummy_input_ids}]
    train(model, dummy_dataloader, num_epochs=1, lr=1e-4)