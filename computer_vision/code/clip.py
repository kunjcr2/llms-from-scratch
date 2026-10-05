import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- 1. Architecture Components ---

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=256, img_size=224):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, E, H/P, W/P) -> (B, N, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, emb_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x[:, 0]  # Return [CLS] token representation

class TextTransformer(nn.Module):
    def __init__(self, vocab_size=10000, max_seq_len=77, emb_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.token_embed(x)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.transformer(x)
        # Using the last token representation (e.g., EOS token)
        return x[:, -1]

class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training (CLIP) Model.
    Connects Vision and Text Encoders into a shared embedding space.
    """
    def __init__(self, emb_dim=256, projection_dim=128):
        super().__init__()
        self.vision_encoder = VisionTransformer(emb_dim=emb_dim)
        self.text_encoder = TextTransformer(emb_dim=emb_dim)
        
        self.visual_projection = nn.Linear(emb_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(emb_dim, projection_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07), requires_grad=True)
        
    def forward(self, image, text):
        image_features = self.vision_encoder(image)
        text_features = self.text_encoder(text)
        
        image_embeddings = self.visual_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        # L2 normalize alignments
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        return image_embeddings, text_embeddings


# --- 2. Loss Function ---
def clip_loss(image_embeddings, text_embeddings, logit_scale):
    """
    Computes symmetric InfoNCE pseudo-cross-entropy loss.
    """
    logit_scale = logit_scale.exp().clamp(max=100)

    # calculating the similarity of all images and texts and multiplying with the temprature
    logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
    logits_per_text = logits_per_image.T
    
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size, device=image_embeddings.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2


# --- 3. Training Pipeline ---
class DummyCLIPDataset(Dataset):
    def __init__(self, size=128, max_seq_len=77, vocab_size=10000):
        self.size = size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        text = torch.randint(0, self.vocab_size, (self.max_seq_len,))
        return image, text

def run_training_pipeline(epochs=5, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing CLIP Training on {device}...")
    
    model = CLIP(emb_dim=256, projection_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    dataset = DummyCLIPDataset(size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, texts in dataloader:
            images, texts = images.to(device), texts.to(device)
            
            optimizer.zero_grad()
            image_embeds, text_embeds = model(images, texts)
            
            loss = clip_loss(image_embeds, text_embeds, model.logit_scale)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Contrastive Loss: {avg_loss:.4f}")

    print("Training Complete.")

if __name__ == "__main__":
    run_training_pipeline(1, 16)
