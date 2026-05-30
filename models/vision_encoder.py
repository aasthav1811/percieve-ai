"""
models/vision_encoder.py
------------------------
ResNet-50 vision encoder fine-tuned on FER2013 for facial emotion recognition.
Produces a 512-d embedding + 7-class emotion logits.

Emotions: Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pathlib import Path

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_EMOTIONS = len(EMOTION_LABELS)
EMBED_DIM = 512


class FacialEmotionEncoder(nn.Module):
    """
    ResNet-50 backbone (ImageNet pretrained) with a custom head for:
      - 512-d emotion embedding (for cross-modal fusion)
      - 7-class emotion classifier
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        # Load ResNet-50 backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Remove the final FC layer → keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # out: (B, 2048, 1, 1)

        # Projection to 512-d embedding
        self.embed_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # Emotion classification head
        self.classifier = nn.Linear(EMBED_DIM, NUM_EMOTIONS)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) face crops, normalised ImageNet stats.
        Returns:
            {
              'embedding': (B, 512)   — fused feature vector,
              'logits':    (B, 7)     — raw class logits,
              'probs':     (B, 7)     — softmax probabilities,
            }
        """
        features = self.backbone(x)           # (B, 2048, 1, 1)
        embedding = self.embed_proj(features) # (B, 512)
        logits = self.classifier(embedding)   # (B, 7)

        return {
            "embedding": embedding,
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
        }

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["embedding"]


# ── Standard image transforms ──────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── FER2013 Dataset ─────────────────────────────────────────────────────────

class FER2013Dataset(torch.utils.data.Dataset):
    """
    Reads FER2013 from the raw CSV (data/fer2013/fer2013.csv).
    Columns: emotion, pixels, Usage
    """

    def __init__(self, csv_path: str, split: str = "Training", transform=None):
        import pandas as pd
        import numpy as np
        from PIL import Image

        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split].reset_index(drop=True)
        self.labels = df["emotion"].values.astype(int)
        self.pixels = [
            np.array(row.split(), dtype=np.uint8).reshape(48, 48)
            for row in df["pixels"]
        ]
        self.transform = transform or EVAL_TRANSFORM
        self.Image = Image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.Image.fromarray(self.pixels[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FacialEmotionEncoder(pretrained=True)
    model.eval()
    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print("Vision Encoder — forward pass:")
    print(f"  embedding shape : {out['embedding'].shape}")   # (4, 512)
    print(f"  logits shape    : {out['logits'].shape}")      # (4, 7)
    print(f"  probs sum       : {out['probs'].sum(dim=-1)}")  # all ≈ 1.0
    print("✓ Vision encoder OK")
