"""
inference.py
------------
End-to-end inference pipeline using pretrained HuggingFace models.

Vision : motheecreator/vit-Facial-Expression-Recognition  (ViT, FER2013, 7-class)
Text   : SamLowe/roberta-base-go_emotions                 (RoBERTa, GoEmotions→7-class)
Fusion : Rule-based weighted attention (no training needed)

Setup (one-time):
    python download_pretrained.py

Run:
    python inference.py --demo
    python inference.py --image face.jpg --text "I feel frustrated"
"""

from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

HF_CACHE   = Path("models/hf_pretrained")
VISION_DIR = HF_CACHE / "vision"
TEXT_DIR   = HF_CACHE / "text"

VISION_REPO = "motheecreator/vit-Facial-Expression-Recognition"
TEXT_REPO   = "SamLowe/roberta-base-go_emotions"

# ── Emotion label mapping ─────────────────────────────────────────────────────
# ViT model outputs these 7 labels directly
VIT_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# GoEmotions 28-class → our 7 core emotions
GOEMOTIONS_MAP = {
    "admiration": "Happy",   "amusement": "Happy",    "approval": "Happy",
    "caring": "Happy",       "excitement": "Happy",   "gratitude": "Happy",
    "joy": "Happy",          "love": "Happy",         "optimism": "Happy",
    "pride": "Happy",        "relief": "Happy",
    "anger": "Angry",        "annoyance": "Angry",    "disapproval": "Angry",
    "disgust": "Disgust",    "embarrassment": "Disgust",
    "fear": "Fear",          "nervousness": "Fear",
    "sadness": "Sad",        "grief": "Sad",          "disappointment": "Sad",
    "remorse": "Sad",
    "surprise": "Surprise",  "realization": "Surprise", "confusion": "Surprise",
    "curiosity": "Neutral",  "desire": "Neutral",     "neutral": "Neutral",
}

CORE_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Valence / arousal / risk per emotion
VALENCE  = {"Happy": 0.85, "Angry": -0.80, "Fear": -0.45, "Sad": -0.65,
            "Surprise": 0.15, "Disgust": -0.70, "Neutral": 0.02}
AROUSAL  = {"Happy": 0.55, "Angry": 0.90,  "Fear": 0.70,  "Sad": 0.30,
            "Surprise": 0.75, "Disgust": 0.60, "Neutral": 0.22}
RISK_MAP = {"Happy": "low", "Angry": "high", "Fear": "medium", "Sad": "medium",
            "Surprise": "low", "Disgust": "medium", "Neutral": "low"}

RISK_RECOMMENDATIONS = {
    "low":    "✓ Stable session. Maintain current engagement tone.",
    "medium": "⚠ Monitor closely. Apply calm reassurance protocol. Follow-up within 24h.",
    "high":   "🚨 Immediate escalation. Senior agent + compensation. Churn risk elevated.",
}


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class EmotionResult:
    facial_emotion: str
    facial_probs:   dict[str, float]
    text_emotion:   str
    text_probs:     dict[str, float]
    sentiment_score: float
    fused_emotion:  str
    fused_probs:    dict[str, float]
    confidence:     float
    valence:        float
    arousal:        float
    risk_level:     str
    modal_weights:  dict
    recommendation: str
    inference_ms:   float


# ── Pipeline ──────────────────────────────────────────────────────────────────

class EmotionInferencePipeline:
    """
    Uses real pretrained HuggingFace models.
    First run downloads models (~900 MB). Subsequent runs load from disk instantly.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        # HuggingFace pipelines want "cpu"/"cuda", not "mps" for pipeline()
        self._hf_device = 0 if device == "cuda" else -1
        self._load_models()

    def _load_models(self):
        print(f"[init] Loading pretrained HuggingFace models on {self.device}...")

        # ── Vision: ViT fine-tuned on FER2013 ──
        vision_source = str(VISION_DIR) if VISION_DIR.exists() else VISION_REPO
        print(f"  ↓ Vision model: {vision_source}")
        try:
            self._vis_processor = AutoImageProcessor.from_pretrained(vision_source)
        except Exception:
            self._vis_processor = AutoFeatureExtractor.from_pretrained(vision_source)
        self._vis_model = AutoModelForImageClassification.from_pretrained(vision_source)
        self._vis_model.eval().to(self.device)
        print("  ✓ Vision encoder ready (ViT — FER2013 fine-tuned)")

        # ── Text: RoBERTa fine-tuned on GoEmotions ──
        text_source = str(TEXT_DIR) if TEXT_DIR.exists() else TEXT_REPO
        print(f"  ↓ Text model: {text_source}")
        self._txt_tokenizer = AutoTokenizer.from_pretrained(text_source)
        self._txt_model = AutoModelForSequenceClassification.from_pretrained(text_source)
        self._txt_model.eval().to(self.device)
        # Map label ids → names
        self._txt_id2label = self._txt_model.config.id2label
        print("  ✓ Text encoder ready (RoBERTa — GoEmotions fine-tuned)")

        print("[init] Pipeline ready.\n")

    # ── Vision inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_vision(self, image: Image.Image) -> tuple[str, dict[str, float], torch.Tensor]:
        """Returns (top_emotion, probs_dict, embedding_tensor)."""
        inputs = self._vis_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._vis_model(**inputs, output_hidden_states=True)
        logits  = outputs.logits                    # (1, 7)
        probs   = F.softmax(logits, dim=-1)[0]      # (7,)

        # Build prob dict using model's own label map
        id2label = self._vis_model.config.id2label
        raw_probs = {id2label[i]: probs[i].item() for i in range(len(probs))}

        # Normalise label names to our standard set
        norm_probs = {e: 0.0 for e in CORE_EMOTIONS}
        for label, prob in raw_probs.items():
            # Handle slight naming differences e.g. "Surprise" vs "Surprised"
            for core in CORE_EMOTIONS:
                if core.lower() in label.lower() or label.lower() in core.lower():
                    norm_probs[core] += prob
                    break

        # Re-normalise
        total = sum(norm_probs.values()) or 1.0
        norm_probs = {k: v / total for k, v in norm_probs.items()}

        top_emotion = max(norm_probs, key=norm_probs.get)

        # Use last hidden state CLS token as embedding for fusion
        # ViT: last_hidden_state[:, 0, :] or pooler_output
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            embedding = outputs.hidden_states[-1][:, 0, :]  # (1, hidden_dim)
        else:
            embedding = logits  # fallback

        return top_emotion, norm_probs, embedding

    # ── Text inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_text(self, text: str) -> tuple[str, dict[str, float], float]:
        """Returns (top_core_emotion, core_probs_dict, sentiment_score)."""
        enc = self._txt_tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128, padding=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self._txt_model(**enc)
        probs_28 = F.softmax(outputs.logits, dim=-1)[0]  # (28,)

        # Aggregate GoEmotions 28 classes → 7 core emotions
        core_probs = {e: 0.0 for e in CORE_EMOTIONS}
        for idx, prob in enumerate(probs_28):
            go_label = self._txt_id2label[idx]          # e.g. "anger", "joy"
            core     = GOEMOTIONS_MAP.get(go_label, "Neutral")
            core_probs[core] += prob.item()

        # Re-normalise
        total = sum(core_probs.values()) or 1.0
        core_probs = {k: v / total for k, v in core_probs.items()}

        top_emotion = max(core_probs, key=core_probs.get)

        # Sentiment score: sum of positive-emotion probabilities
        positive_emotions = {"Happy", "Surprise"}
        sentiment = sum(core_probs[e] for e in positive_emotions)
        sentiment = min(1.0, max(0.0, sentiment))

        return top_emotion, core_probs, sentiment

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(
        self,
        vis_emotion: str, vis_probs: dict,
        txt_emotion: str, txt_probs: dict,
        sentiment:   float,
    ) -> tuple[str, dict, float, float, float, str, dict]:
        """
        Weighted attention fusion — no training required.
        Visual weight 55%, Text weight 45% (text is slightly less reliable
        without full fine-tuning on paired data).
        """
        VIS_W, TXT_W = 0.55, 0.45

        fused = {}
        for e in CORE_EMOTIONS:
            fused[e] = vis_probs.get(e, 0.0) * VIS_W + txt_probs.get(e, 0.0) * TXT_W

        # Re-normalise
        total = sum(fused.values()) or 1.0
        fused = {k: v / total for k, v in fused.items()}

        top_emotion = max(fused, key=fused.get)
        confidence  = fused[top_emotion]

        # Valence: weighted between emotion-derived and sentiment-derived
        emo_valence = VALENCE[top_emotion]
        sent_valence = (sentiment - 0.5) * 2          # map [0,1] → [-1,+1]
        valence = emo_valence * 0.7 + sent_valence * 0.3
        valence = float(max(-1.0, min(1.0, valence)))

        arousal = AROUSAL[top_emotion]
        risk    = RISK_MAP[top_emotion]

        modal_weights = {"visual": VIS_W, "text": TXT_W}
        return top_emotion, fused, confidence, valence, arousal, risk, modal_weights

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, image: Image.Image | np.ndarray, text: str) -> EmotionResult:
        """Full multimodal inference. Returns EmotionResult."""
        t0 = time.perf_counter()

        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1] if image.ndim == 3 else image)
        image = image.convert("RGB")

        # Run encoders
        vis_emotion, vis_probs, _vis_embed = self._run_vision(image)
        txt_emotion, txt_probs, sentiment  = self._run_text(text)

        # Fuse
        fused_emo, fused_probs, conf, valence, arousal, risk, modal_w = self._fuse(
            vis_emotion, vis_probs,
            txt_emotion, txt_probs,
            sentiment,
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        return EmotionResult(
            facial_emotion  = vis_emotion,
            facial_probs    = vis_probs,
            text_emotion    = txt_emotion,
            text_probs      = txt_probs,
            sentiment_score = sentiment,
            fused_emotion   = fused_emo,
            fused_probs     = fused_probs,
            confidence      = conf,
            valence         = valence,
            arousal         = arousal,
            risk_level      = risk,
            modal_weights   = modal_w,
            recommendation  = RISK_RECOMMENDATIONS[risk],
            inference_ms    = inference_ms,
        )


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(r: EmotionResult):
    print("=" * 55)
    print(f"  Fused Emotion : {r.fused_emotion}  (conf={r.confidence:.1%})")
    print(f"  Valence       : {r.valence:+.3f}  |  Arousal: {r.arousal:.3f}")
    print(f"  Risk Level    : {r.risk_level.upper()}")
    print(f"  Inference     : {r.inference_ms:.0f}ms")
    print("-" * 55)
    print(f"  Facial → {r.facial_emotion}")
    print(f"  Text   → {r.text_emotion}  (sentiment: {r.sentiment_score:.2f})")
    print(f"  Modal weights: visual={r.modal_weights['visual']:.0%}  text={r.modal_weights['text']:.0%}")
    print("-" * 55)
    print(f"  {r.recommendation}")
    print("=" * 55)


# ── Demo scenarios ────────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    {
        "name": "Happy Customer",
        "text": "This is exactly what I needed. The response was fast and the solution worked perfectly. Very impressed!",
        "image_color": (200, 230, 200),
    },
    {
        "name": "Frustrated User",
        "text": "This is ridiculous. I've been waiting for 45 minutes and nobody has resolved my issue. Completely unacceptable.",
        "image_color": (230, 200, 200),
    },
    {
        "name": "Anxious Caller",
        "text": "I'm not sure if my account is secure. I saw some transactions I didn't recognise and I'm worried.",
        "image_color": (230, 220, 190),
    },
    {
        "name": "Neutral Session",
        "text": "I would like to update my shipping address for order number 4829.",
        "image_color": (200, 210, 220),
    },
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="emit_ai — Multimodal Emotion Inference")
    p.add_argument("--image",  help="Path to face image")
    p.add_argument("--text",   help="Text input")
    p.add_argument("--demo",   action="store_true", help="Run all demo scenarios")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    pipe  = EmotionInferencePipeline(device=args.device)

    if args.demo:
        print("\n── Demo Scenarios ──\n")
        for s in DEMO_SCENARIOS:
            print(f"\n[{s['name']}]")
            img    = Image.new("RGB", (224, 224), color=s["image_color"])
            result = pipe.run(image=img, text=s["text"])
            print(f"  Text: {s['text'][:70]}…")
            print_result(result)
    elif args.image and args.text:
        img    = Image.open(args.image).convert("RGB")
        result = pipe.run(image=img, text=args.text)
        print_result(result)
    else:
        print("Usage:")
        print("  python inference.py --demo")
        print("  python inference.py --image face.jpg --text 'I feel frustrated'")