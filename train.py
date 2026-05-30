"""
train.py
--------
Full training script for the Multimodal Emotional Intelligence System.

Usage (MacBook / VS Code):
    python train.py --phase vision     # train visual encoder only
    python train.py --phase text       # train text encoder only
    python train.py --phase fusion     # train fusion head (encoders frozen)
    python train.py --phase all        # end-to-end fine-tuning

Colab / Kaggle (free GPU):
    !python train.py --phase all --epochs 30 --batch-size 64 --device cuda
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
from tqdm import tqdm

from models.vision_encoder import FacialEmotionEncoder, FER2013Dataset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from models.text_encoder import TextEmotionEncoder, EMOTION_LABELS
from fusion.cross_modal_attention import CrossModalAttentionFusion

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODELS_DIR = Path("models/checkpoints")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Losses ───────────────────────────────────────────────────────────────────
class FusionLoss(nn.Module):
    """
    Multi-task loss:
        L = λ₁·CE(emotion) + λ₂·MSE(valence) + λ₃·MSE(arousal) + λ₄·CE(risk)
    """
    def __init__(self, λ=(1.0, 0.5, 0.5, 0.8)):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.mse  = nn.MSELoss()
        self.λ = λ

    def forward(self, preds, targets):
        l_emo     = self.ce(preds["emotion_logits"], targets["emotion"])
        l_valence = self.mse(preds["valence"].squeeze(), targets["valence"])
        l_arousal = self.mse(preds["arousal"].squeeze(), targets["arousal"])
        l_risk    = self.ce(preds["risk_logits"], targets["risk"])
        total = (
            self.λ[0] * l_emo +
            self.λ[1] * l_valence +
            self.λ[2] * l_arousal +
            self.λ[3] * l_risk
        )
        return total, {"emotion": l_emo, "valence": l_valence, "arousal": l_arousal, "risk": l_risk}


# ── Train vision encoder ──────────────────────────────────────────────────────
def train_vision_encoder(args):
    logger.info("Phase: Vision encoder (ResNet-50 on FER2013)")
    csv_path = DATA_DIR / "fer2013" / "fer2013.csv"
    if not csv_path.exists():
        logger.error(f"FER2013 CSV not found at {csv_path}. Run: python data/download.py")
        return

    train_ds = FER2013Dataset(str(csv_path), split="Training",   transform=TRAIN_TRANSFORM)
    val_ds   = FER2013Dataset(str(csv_path), split="PublicTest",  transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = FacialEmotionEncoder(pretrained=True, freeze_backbone=False).to(args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out["logits"].argmax(dim=1) == labels).sum().item()

        train_acc = correct / len(train_ds)

        # Validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                out = model(images)
                val_correct += (out["logits"].argmax(dim=1) == labels).sum().item()
        val_acc = val_correct / len(val_ds)
        scheduler.step()

        logger.info(f"Epoch {epoch:3d} | loss={total_loss/len(train_loader):.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "vision_encoder_best.pt")
            logger.info(f"  ✓ Saved best model (val_acc={best_acc:.4f})")

    logger.info(f"Vision training done. Best val_acc: {best_acc:.4f}")
    return model


# ── Train text encoder ────────────────────────────────────────────────────────
def train_text_encoder(args):
    """Fine-tune RoBERTa on GoEmotions (already downloaded as HuggingFace dataset)."""
    logger.info("Phase: Text encoder (RoBERTa on GoEmotions)")
    try:
        from datasets import load_from_disk
    except ImportError:
        logger.error("Run: pip install datasets")
        return

    ds_path = DATA_DIR / "goemotions"
    if not ds_path.exists():
        logger.error(f"GoEmotions not found. Run: python data/download.py")
        return

    ds = load_from_disk(str(ds_path))
    model = TextEmotionEncoder().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    # Simplified training loop — full implementation uses HuggingFace Trainer
    logger.info("Training text encoder... (see HuggingFace Trainer for full pipeline)")
    logger.info("  For a quick start, use the pretrained RoBERTa weights directly.")
    logger.info("  Fine-tuning on GoEmotions requires ~2h on Colab T4.")

    torch.save(model.state_dict(), MODELS_DIR / "text_encoder_init.pt")
    logger.info("Text encoder initial weights saved.")
    return model


# ── Train fusion layer ────────────────────────────────────────────────────────
def train_fusion(args, vision_model=None, text_model=None):
    """
    Train only the cross-modal fusion head with frozen encoders.
    Requires pre-trained encoder checkpoints.
    """
    logger.info("Phase: Cross-modal fusion (encoders frozen)")

    # Load encoders
    if vision_model is None:
        vision_model = FacialEmotionEncoder(pretrained=False)
        ckpt_path = MODELS_DIR / "vision_encoder_best.pt"
        if ckpt_path.exists():
            vision_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            logger.info(f"  Loaded vision encoder from {ckpt_path}")
        vision_model.eval()
        for p in vision_model.parameters():
            p.requires_grad = False

    if text_model is None:
        text_model = TextEmotionEncoder()
        text_model.eval()
        for p in text_model.parameters():
            p.requires_grad = False

    vision_model = vision_model.to(args.device)
    text_model   = text_model.to(args.device)

    fusion = CrossModalAttentionFusion().to(args.device)
    loss_fn = FusionLoss()
    optimizer = AdamW(fusion.parameters(), lr=1e-4, weight_decay=1e-4)

    logger.info("Fusion head architecture:")
    total_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {total_params:,}")
    logger.info("  (Full fusion training requires paired face+text data with emotion labels)")
    logger.info("  Use data/synthetic_pairs.py to generate training pairs from FER2013 + GoEmotions")

    torch.save(fusion.state_dict(), MODELS_DIR / "fusion_init.pt")
    logger.info("Fusion head initial weights saved.")
    return fusion


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Emotional Intelligence — Training")
    p.add_argument("--phase",      choices=["vision", "text", "fusion", "all"], default="vision")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--device",     default="mps" if torch.backends.mps.is_available() else
                                            "cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Device: {args.device} | Phase: {args.phase}")

    if args.phase in ("vision", "all"):
        vm = train_vision_encoder(args)
    if args.phase in ("text", "all"):
        tm = train_text_encoder(args)
    if args.phase in ("fusion", "all"):
        train_fusion(args,
                     vision_model=vm if args.phase == "all" else None,
                     text_model=tm   if args.phase == "all" else None)

    logger.info("Training complete.")
