"""
models/text_encoder.py
----------------------
RoBERTa-base text encoder for sentiment + emotion classification.
Produces a 768-d [CLS] embedding + sentiment score + 7-class emotion logits.

Pre-trained on:
  - SST-2 sentiment (via HuggingFace SentimentIntensityAnalyzer)
  - GoEmotions 27-class → mapped to 7 core emotions
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizerFast

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_EMOTIONS = len(EMOTION_LABELS)
TEXT_EMBED_DIM = 768
MODEL_NAME = "roberta-base"

# Map GoEmotions 28 labels → our 7 core emotions
GOEMOTIONS_TO_CORE = {
    # Angry
    "anger": "Angry", "annoyance": "Angry", "disapproval": "Angry",
    # Disgust
    "disgust": "Disgust", "embarrassment": "Disgust",
    # Fear
    "fear": "Fear", "nervousness": "Fear",
    # Happy
    "joy": "Happy", "amusement": "Happy", "excitement": "Happy",
    "gratitude": "Happy", "love": "Happy", "optimism": "Happy",
    "pride": "Happy", "relief": "Happy",
    # Sad
    "sadness": "Sad", "grief": "Sad", "disappointment": "Sad",
    "remorse": "Sad",
    # Surprise
    "surprise": "Surprise", "realization": "Surprise",
    "confusion": "Surprise",
    # Neutral
    "neutral": "Neutral", "approval": "Neutral", "caring": "Neutral",
    "curiosity": "Neutral", "desire": "Neutral", "admiration": "Neutral",
}


class TextEmotionEncoder(nn.Module):
    """
    RoBERTa-base with two task heads:
      1. Emotion classifier (7-class)
      2. Sentiment regressor (0 = negative, 1 = positive)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        freeze_base_layers: int = 8,   # freeze bottom N of 12 transformer layers
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

        # Freeze lower layers to preserve pre-trained features
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < freeze_base_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        # Emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_EMOTIONS),
        )

        # Sentiment head (scalar regression)
        self.sentiment_head = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # → [0, 1]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, seq_len) — tokenized text
            attention_mask: (B, seq_len) — padding mask
        Returns:
            {
              'embedding':  (B, 768)  — [CLS] token representation,
              'logits':     (B, 7)    — emotion logits,
              'probs':      (B, 7)    — emotion probabilities,
              'sentiment':  (B, 1)    — sentiment score [0, 1],
            }
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token: (B, 768)

        logits = self.emotion_head(cls_embedding)           # (B, 7)
        sentiment = self.sentiment_head(cls_embedding)      # (B, 1)

        return {
            "embedding": cls_embedding,
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "sentiment": sentiment,
        }

    def encode_text(self, text: str | list[str], device: str = "cpu", max_length: int = 128):
        """Tokenize and run a forward pass. Convenience wrapper for inference."""
        if isinstance(text, str):
            text = [text]
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return self.forward(enc["input_ids"], enc["attention_mask"])

    def get_embedding(self, text: str | list[str], device: str = "cpu") -> torch.Tensor:
        """Returns only the [CLS] embedding — used by the fusion layer."""
        return self.encode_text(text, device)["embedding"]


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading RoBERTa-base (first run downloads ~500MB)...")
    model = TextEmotionEncoder()
    model.eval()

    samples = [
        "This is exactly what I needed. Very impressed!",
        "I've been waiting 45 minutes. Completely unacceptable.",
        "I'm not sure if my account is secure.",
        "I would like to update my shipping address.",
    ]

    results = model.encode_text(samples)
    print("\nText Encoder — forward pass:")
    print(f"  embedding shape : {results['embedding'].shape}")  # (4, 768)
    print(f"  logits shape    : {results['logits'].shape}")     # (4, 7)
    print(f"  sentiments      : {results['sentiment'].squeeze().tolist()}")
    print(f"\nPredicted emotions:")
    for text, probs in zip(samples, results["probs"]):
        top = EMOTION_LABELS[probs.argmax().item()]
        conf = probs.max().item()
        print(f"  [{top:>8s} {conf:.0%}] {text[:60]}")
    print("\n✓ Text encoder OK")
