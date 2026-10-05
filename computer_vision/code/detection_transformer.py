"""
Detection Transformer (DETR): From-Scratch Implementation with Pretrained Weights
==================================================================================
An educational implementation of DETR for object detection, built from scratch
but loaded with Facebook's pretrained weights for immediate inference.

Key Components:
- ResNet-50 Backbone: Extracts 2048-channel feature maps from input images
- 1x1 Conv Projection: Projects 2048 channels -> 256 (hidden_dim)
- Transformer Encoder: 6 layers of self-attention over flattened image features
- Transformer Decoder: 6 layers with self-attention among object queries
  and cross-attention between queries and encoder output
- Prediction Heads: Separate MLPs for class logits and bounding box coordinates

Simplifications vs Original Paper:
- Position embeddings use learned row/column vectors (not sinusoidal)
- Object queries are initialized as random learned parameters (not zeros + learned PE)
- Uses nn.Transformer instead of custom encoder/decoder blocks

Reference: Carion et al., "End-to-End Object Detection with Transformers" (2020)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet50
import torchvision.transforms as T
import requests

# Disable gradient computation globally since we're only doing inference
# (no training or fine-tuning — pretrained weights are loaded directly)
torch.set_grad_enabled(False)


# ==============================================================================
# COCO Dataset Classes (91 categories used in DETR pretraining)
# ==============================================================================
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Colors for drawing bounding boxes (cycled if more boxes than colors)
BBOX_COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']


# ==============================================================================
# Detection Transformer Model
# ==============================================================================
class DETRdemo(nn.Module):
    """Detection Transformer (DETR) — simplified demo implementation.

    Architecture Overview:
        Input Image -> ResNet-50 Backbone -> Feature Map (B, 2048, H, W)
                    -> 1x1 Conv Projection -> (B, 256, H, W)
                    -> Flatten + Position Embedding -> Transformer Encoder
                    -> Object Queries -> Transformer Decoder
                    -> Class MLP (92 classes) + BBox MLP (4 coords)

    Args:
        num_classes: Number of object categories (91 COCO classes + 1 "no object")
        hidden_dim: Embedding dimension for transformer (must be 256 for pretrained weights)
        n_heads: Number of attention heads in multi-head attention
        num_encoder_layers: Number of transformer encoder blocks in series
        num_decoder_layers: Number of transformer decoder blocks in series
    """

    def __init__(
        self,
        num_classes=91,
        hidden_dim=256,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()

        # ── Part 1: ResNet-50 Backbone ──────────────────────────────────
        # We use ResNet-50 purely as a feature extractor (no classification head).
        # The architecture is defined here; weights are random until we load
        # pretrained state_dict later. ResNet-50's final conv layer outputs
        # 2048-channel feature maps with reduced spatial dimensions.
        self.backbone = resnet50()
        del self.backbone.fc  # Remove classification head — not needed for detection

        # Project ResNet's 2048-channel feature map to the transformer's
        # hidden_dim (256) using a 1x1 convolution (acts as a linear projection
        # per spatial position)
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # ── Part 2: Transformer (Encoder + Decoder) ─────────────────────
        # nn.Transformer bundles both encoder and decoder together.
        # - Encoder: self-attention over image feature tokens
        # - Decoder: self-attention among object queries + cross-attention
        #            between object queries and encoder output
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        # ── Part 3: Prediction Heads (MLPs) ─────────────────────────────
        # Classification head: projects each decoder output to class logits
        # num_classes + 1 because DETR includes a "no object" (∅) class
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)

        # Bounding box head: projects each decoder output to 4 normalized
        # coordinates (center_x, center_y, width, height), all in [0, 1]
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # ── Part 4: Position Embeddings ─────────────────────────────────
        # Object query position embeddings (100 learnable vectors):
        # In the original DETR, object queries start as zeros and get
        # learnable position embeddings added. Since 0 + PE = PE, we just
        # use the PE directly as the object queries.
        # 100 = maximum number of objects we can detect per image.
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # Row and column position embeddings for image features:
        # Instead of sinusoidal PE (used in original DETR), we use learned
        # row/column embeddings. Each is hidden_dim//2 so that when
        # concatenated (row || col), total dimension = hidden_dim.
        # 50 is a safe upper bound for the feature map spatial dimensions
        # (ResNet-50 outputs ~25x42 for 800px input — both < 50).
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs: torch.Tensor):
        """Forward pass through the full DETR pipeline.

        Args:
            inputs: (B, 3, H_img, W_img) — batch of RGB images

        Returns:
            pred_logits: (B, 100, num_classes+1) — class predictions per query
            pred_boxes:  (B, 100, 4) — bbox predictions (cx, cy, w, h) in [0,1]
        """
        # ── Forward pass through ResNet backbone ────────────────────────
        # We pass through individual ResNet layers (not the full model)
        # because the pretrained DETR uses a slightly trimmed ResNet
        # (fewer layers than vanilla ResNet-50).
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # Spatial downsampling via max pooling

        # Four residual blocks — each progressively extracts higher-level features
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # x shape: (B, 2048, H_feat, W_feat) — e.g., (1, 2048, 25, 42)

        # Project from 2048 channels to hidden_dim (256)
        h = self.conv(x)
        # h shape: (B, 256, H_feat, W_feat)

        # ── Construct position embeddings for image features ────────────
        # Extract spatial dimensions of the feature map
        H, W = h.shape[-2:]

        # Build 2D position embedding by combining row and column embeddings:
        #   col_embed[:W] -> (W, hidden_dim//2), unsqueeze(0) -> (1, W, D//2)
        #       .repeat(H, 1, 1) -> (H, W, D//2): same col PE for every row
        #   row_embed[:H] -> (H, hidden_dim//2), unsqueeze(1) -> (H, 1, D//2)
        #       .repeat(1, W, 1) -> (H, W, D//2): same row PE for every col
        #   Concatenate along last dim -> (H, W, hidden_dim)
        #   Flatten H*W -> (H*W, hidden_dim), unsqueeze(1) -> (H*W, 1, hidden_dim)
        #
        # This encodes 2D spatial structure: two positions sharing a row get
        # the same row embedding; two positions sharing a column get the
        # same column embedding.
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        # pos shape: (H*W, 1, hidden_dim)

        # ── Prepare inputs for the transformer ──────────────────────────
        # Flatten the feature map from (B, 256, H, W) to (H*W, B, 256)
        # PyTorch's nn.Transformer expects (seq_len, batch, embed_dim) by default
        # (unlike batch-first convention). That's why we permute to (2, 0, 1):
        #   flatten(2): (B, 256, H*W) — merge spatial dims
        #   permute(2, 0, 1): (H*W, B, 256) — seq_len first for transformer
        # Add position embedding to give spatial awareness
        h = pos + h.flatten(2).permute(2, 0, 1)
        # h shape: (H*W, B, hidden_dim) — encoder input

        # ── Run the transformer ─────────────────────────────────────────
        # First argument = encoder input (image features + position PE)
        # Second argument = decoder input (object queries)
        #
        # Object queries (query_pos) are learned parameters of shape (100, 256).
        # unsqueeze(1) adds batch dim -> (100, 1, 256) for broadcasting.
        # The transformer decoder:
        #   1. Self-attention among the 100 object queries
        #   2. Cross-attention: queries attend to encoder output (image features)
        # Output: (100, B, 256) — one context vector per object query
        h = self.transformer(
            h,
            self.query_pos.unsqueeze(1)
        ).transpose(0, 1)
        # transpose(0, 1): (100, B, 256) -> (B, 100, 256) — batch first again

        # ── Prediction heads ────────────────────────────────────────────
        # Each of the 100 context vectors is independently projected to:
        #   - Class logits (92 dims: 91 COCO classes + "no object")
        #   - BBox coordinates (4 dims: cx, cy, w, h)
        # Sigmoid on bbox ensures all coordinates are in [0, 1]
        return {
            'pred_logits': self.linear_class(h),
            'pred_boxes': self.linear_bbox(h).sigmoid(),
        }


# ==============================================================================
# Loading Pretrained Weights from Facebook
# ==============================================================================
def load_pretrained_detr() -> DETRdemo:
    """Download and load Facebook's pretrained DETR weights into our model.

    The pretrained model was trained on COCO dataset.
    Weights URL: https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth

    Returns:
        DETRdemo model with pretrained weights in eval mode
    """
    detr = DETRdemo(num_classes=91, hidden_dim=256, n_heads=8,
                    num_encoder_layers=6, num_decoder_layers=6)

    # Download pretrained weights from Facebook's servers
    state_dict = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
        map_location='cpu',
        check_hash=True,
    )

    # Load weights into our model architecture
    detr.load_state_dict(state_dict)

    # Set to evaluation mode (disables dropout, batchnorm uses running stats)
    detr.eval()

    return detr


# ==============================================================================
# Image Preprocessing & BBox Utilities
# ==============================================================================

# Standard ImageNet normalization — required because the ResNet backbone
# in DETR was pretrained on ImageNet with these statistics
transform = T.Compose([
    T.Resize(800),              # Resize shorter side to 800px
    T.ToTensor(),               # Convert PIL Image to tensor (0-1 range)
    T.Normalize(                # Normalize with ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from center format to corner format.

    DETR predicts boxes as (center_x, center_y, width, height).
    For drawing rectangles, we need (x_min, y_min, x_max, y_max).

    Args:
        x: (N, 4) tensor of boxes in (cx, cy, w, h) format

    Returns:
        (N, 4) tensor of boxes in (x_min, y_min, x_max, y_max) format
    """
    x_c, y_c, w, h = x.unbind(1)
    b = [
        x_c - 0.5 * w,  # x_min = center_x - half_width
        y_c - 0.5 * h,  # y_min = center_y - half_height
        x_c + 0.5 * w,  # x_max = center_x + half_width
        y_c + 0.5 * h,  # y_max = center_y + half_height
    ]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox: torch.Tensor, size: tuple) -> torch.Tensor:
    """Scale normalized bounding box coordinates to actual image dimensions.

    DETR outputs boxes in [0, 1] normalized coordinates. This function
    scales them to pixel coordinates matching the original image size.

    Args:
        out_bbox: (N, 4) normalized boxes in (cx, cy, w, h) format
        size: (width, height) of the original image

    Returns:
        (N, 4) boxes in pixel coordinates (x_min, y_min, x_max, y_max)
    """
    img_w, img_h = size

    # Convert from center to corner format first
    b = box_cxcywh_to_xyxy(out_bbox)

    # Scale each coordinate by the corresponding image dimension
    # x coords scaled by width, y coords scaled by height
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    return b


# ==============================================================================
# Inference Functions
# ==============================================================================
def detect(image: Image.Image, model: DETRdemo, transform: T.Compose,
           confidence_threshold: float = 0.7):
    """Run DETR inference on a single image.

    Args:
        image: PIL Image to run detection on
        model: Pretrained DETRdemo model in eval mode
        transform: Image preprocessing transforms
        confidence_threshold: Only keep predictions above this probability

    Returns:
        probas: (K, 91) class probabilities for K high-confidence detections
        bboxes_scaled: (K, 4) pixel-coordinate bounding boxes
    """
    # Preprocess: resize, to tensor, normalize, add batch dimension
    img = transform(image).unsqueeze(0)  # (1, 3, H, W)

    # Forward pass through DETR
    outputs = model(img)

    # Extract class probabilities (exclude "no object" class at index -1)
    # outputs['pred_logits'] shape: (1, 100, 92)
    # After softmax + slicing: (100, 91) — probability for each real class
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

    # Keep only detections where the max class probability exceeds threshold
    keep = probas.max(-1).values > confidence_threshold

    # Scale bounding boxes from normalized to pixel coordinates
    bboxes_scaled = rescale_bboxes(
        outputs['pred_boxes'][0, keep],  # Only kept boxes
        image.size                        # (width, height) of original image
    )

    return probas[keep], bboxes_scaled


# ==============================================================================
# Visualization
# ==============================================================================
def plot_results(pil_img: Image.Image, prob: torch.Tensor, boxes: torch.Tensor):
    """Overlay detected bounding boxes and labels on the image.

    Draws a colored rectangle for each detection with the predicted class
    name and confidence score as a label.

    Args:
        pil_img: Original PIL Image
        prob: (K, 91) class probabilities for K detections
        boxes: (K, 4) bounding boxes in pixel coordinates (x1, y1, x2, y2)
    """
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    # Cycle through colors if there are more boxes than available colors
    colors = BBOX_COLORS * 100

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        # Draw bounding box rectangle
        ax.add_patch(plt.Rectangle(
            (xmin, ymin),               # Bottom-left corner
            xmax - xmin,                # Width
            ymax - ymin,                # Height
            fill=False,                 # No fill — outline only
            color=c,
            linewidth=3,
        ))

        # Get predicted class name and confidence
        cl = p.argmax()
        text = f'{COCO_CLASSES[cl]}: {p[cl]:0.2f}'

        # Draw label above the bounding box
        ax.text(
            xmin, ymin,
            text,
            fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5),
        )

    plt.axis('off')
    plt.show()


# ==============================================================================
# Main: Download Image, Run Detection, Plot Results
# ==============================================================================
if __name__ == '__main__':
    # Load pretrained DETR model
    print("Loading pretrained DETR model...")
    detr_model = load_pretrained_detr()
    print("Model loaded successfully!\n")
    print(detr_model)
    print()

    # Load a sample image from the web
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1200px-New_york_times_square-terabass.jpg'
    print(f"Downloading test image from:\n  {url}\n")
    im = Image.open(requests.get(url, stream=True).raw)

    # Run detection
    print("Running inference...")
    scores, boxes = detect(im, detr_model, transform, confidence_threshold=0.7)
    print(f"Detected {len(scores)} objects\n")

    # Plot results with bounding boxes and labels
    plot_results(im, scores, boxes)
