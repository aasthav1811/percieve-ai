"""
api.py
------
FastAPI inference server for the Multimodal Emotional Intelligence System.

Endpoints:
  POST /analyze          — image + text → full emotion analysis
  POST /analyze/text     — text only (no image required)
  GET  /health           — health check
  GET  /emotions         — list supported emotion labels

Run locally (MacBook):
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Deploy on Render (free tier):
  1. Push to GitHub
  2. New Web Service → connect repo → start command: uvicorn api:app --host 0.0.0.0 --port $PORT
  3. Set env var: PYTHON_VERSION=3.11.9
"""

from __future__ import annotations
import io
import os
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from inference import EmotionInferencePipeline, EmotionResult

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Emotional Intelligence API",
    description="Fuses facial expression analysis and NLP sentiment into unified emotion outputs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://emit-ai.vercel.app",
        "*",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load pipeline on first request (avoids cold-start memory issues on free tier)
_pipeline: Optional[EmotionInferencePipeline] = None

def get_pipeline() -> EmotionInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EmotionInferencePipeline()
    return _pipeline

@app.on_event("startup")
async def startup_event():
    global _pipeline
    print("[startup] Loading models into memory...")
    _pipeline = EmotionInferencePipeline()
    print("[startup] Models ready.")




# ── Response models ───────────────────────────────────────────────────────────

class ModalWeights(BaseModel):
    visual: float
    text: float

class EmotionAnalysisResponse(BaseModel):
    facial_emotion: str
    facial_probs: dict[str, float]
    text_emotion: str
    text_probs: dict[str, float]
    sentiment_score: float
    fused_emotion: str
    fused_probs: dict[str, float]
    confidence: float
    valence: float
    arousal: float
    risk_level: str
    modal_weights: ModalWeights
    recommendation: str
    inference_ms: float

class TextOnlyResponse(BaseModel):
    emotion: str
    probs: dict[str, float]
    sentiment_score: float
    inference_ms: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def result_to_response(r: EmotionResult) -> EmotionAnalysisResponse:
    return EmotionAnalysisResponse(
        facial_emotion=r.facial_emotion,
        facial_probs=r.facial_probs,
        text_emotion=r.text_emotion,
        text_probs=r.text_probs,
        sentiment_score=r.sentiment_score,
        fused_emotion=r.fused_emotion,
        fused_probs=r.fused_probs,
        confidence=r.confidence,
        valence=r.valence,
        arousal=r.arousal,
        risk_level=r.risk_level,
        modal_weights=ModalWeights(**r.modal_weights),
        recommendation=r.recommendation,
        inference_ms=r.inference_ms,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
        "timestamp": time.time(),
    }


@app.get("/emotions")
def list_emotions():
    from fusion.cross_modal_attention import EMOTION_LABELS, RISK_LABELS
    return {
        "emotions": EMOTION_LABELS,
        "risk_levels": RISK_LABELS,
    }


@app.post("/analyze", response_model=EmotionAnalysisResponse)
async def analyze(
    image: UploadFile = File(..., description="Face image (JPEG/PNG)"),
    text:  str        = Form(..., description="Transcribed speech or chat message"),
):
    """
    Full multimodal emotion analysis.
    Accepts a face image + text, returns fused emotion prediction.
    """
    # Validate image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File must be an image (JPEG/PNG).")

    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Text field cannot be empty.")

    pipeline = get_pipeline()
    result = pipeline.run(image=pil_image, text=text.strip())
    return result_to_response(result)


@app.post("/analyze/text", response_model=TextOnlyResponse)
async def analyze_text(text: str = Form(..., description="Text to analyze")):
    """
    Text-only emotion and sentiment analysis (no image required).
    Faster — useful for chat or voice transcript pipelines.
    """
    if not text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    t0 = time.perf_counter()
    pipeline = get_pipeline()
    import torch
    with torch.no_grad():
        result = pipeline.text.encode_text(text.strip(), device=pipeline.device)

    from fusion.cross_modal_attention import EMOTION_LABELS
    probs = result["probs"][0].cpu().numpy()
    emotion = EMOTION_LABELS[probs.argmax()]

    return TextOnlyResponse(
        emotion=emotion,
        probs={k: float(v) for k, v in zip(EMOTION_LABELS, probs)},
        sentiment_score=float(result["sentiment"][0, 0]),
        inference_ms=(time.perf_counter() - t0) * 1000,
    )


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
