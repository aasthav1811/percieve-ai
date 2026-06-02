---
title: Perceive AI
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Multimodal Emotional Intelligence System

Fuses facial expression analysis (ViT) with NLP sentiment (RoBERTa) via cross-modal attention to predict emotions, valence, arousal, and risk level in real time.

## Stack
- Vision: ViT fine-tuned on FER2013
- Text: RoBERTa fine-tuned on GoEmotions  
- Fusion: Cross-modal attention
- API: FastAPI

## API Endpoints
- `GET /health` — health check
- `POST /analyze` — image + text → full emotion analysis
- `POST /analyze/text` — text only analysis
- `GET /docs` — interactive API documentation
