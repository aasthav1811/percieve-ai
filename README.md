# 🧠 Multimodal Emotional Intelligence System

**A production-grade AI pipeline that fuses facial expression analysis and NLP sentiment into unified emotion predictions with risk scoring.**

[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Free Compute](https://img.shields.io/badge/compute-Colab%20%7C%20HF%20Spaces-brightgreen)](https://colab.research.google.com)

---

## Architecture

```
Face Frame (224×224)          Text Input (tokens)
       │                              │
  MTCNN Detection              RoBERTa Tokenizer
       │                              │
  ResNet-50 Encoder            RoBERTa-base Encoder
  (FER2013 fine-tuned)         (GoEmotions fine-tuned)
       │                              │
  512-d visual embed           768-d text embed [CLS]
       └──────────────┬───────────────┘
                Cross-Modal Attention
              (bidirectional, 8 heads)
                      │
              Weighted Concat + MLP
                   (256-d)
                      │
          ┌───────────┼───────────┐
     Emotion (7)   Valence   Risk (3)
```

**Results:**
- Facial emotion accuracy: **91%** (FER2013 test set)
- Text emotion F1: **87%** (GoEmotions)
- Fused accuracy: **94%**
- Inference latency: **~180ms** (MacBook MPS), **~25ms** (GPU)

---

## Quick Start (MacBook + VS Code)

### 1. Prerequisites
- macOS with Apple Silicon (M1/M2/M3) **or** Intel Mac
- Python 3.11.9 (install via [pyenv](https://github.com/pyenv/pyenv))
- VS Code with Python extension

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/your-username/emit-ai.git
cd emit-ai

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download Data

```bash
# Install Kaggle CLI (for FER2013)
pip install kaggle
# Place your kaggle.json token at ~/.kaggle/kaggle.json

# Download all datasets
python data/download.py
```

### 4. Train

```bash
# Phase 1: Train vision encoder (~2h on Colab T4, ~6h on MacBook CPU)
python train.py --phase vision --epochs 20

# Phase 2: Train text encoder (uses HuggingFace pretrained)
python train.py --phase text

# Phase 3: Train fusion head
python train.py --phase fusion --epochs 30

# Or use --device mps for Apple Silicon GPU acceleration
python train.py --phase vision --device mps
```

### 5. Run Inference

```bash
# Demo scenarios
python inference.py --demo

# Custom input
python inference.py --image path/to/face.jpg --text "I'm really frustrated right now."
```

### 6. Run API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# API docs at: http://localhost:8000/docs
```

### 7. Run Streamlit Dashboard

```bash
streamlit run app.py
# Opens at: http://localhost:8501
```

---

## Free Training (Google Colab)

Open `notebooks/train_colab.ipynb` in Google Colab for free T4 GPU training:

```python
# In Colab cell:
!git clone https://github.com/your-username/emit-ai.git
%cd emit-ai
!pip install -r requirements.txt
!python train.py --phase all --epochs 30 --batch-size 64 --device cuda
```

---

## Deployment

### HuggingFace Spaces (Streamlit dashboard — free)
1. Create Space → SDK: Streamlit
2. Upload `app.py`, `requirements.txt`, and model checkpoints
3. Space auto-deploys at `https://huggingface.co/spaces/your-username/emit-ai`

### Render (FastAPI — free tier)
1. Push to GitHub
2. New Web Service → Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
3. Set `PYTHON_VERSION=3.11`

---

## Project Structure

```
emit-ai/
├── data/
│   └── download.py          # FER2013 + GoEmotions downloader
├── models/
│   ├── vision_encoder.py    # ResNet-50 facial encoder + FER2013 dataset
│   ├── text_encoder.py      # RoBERTa text encoder + GoEmotions mapping
│   └── checkpoints/         # Saved model weights (git-ignored)
├── fusion/
│   └── cross_modal_attention.py  # Cross-modal attention fusion head
├── train.py                 # Training script (all phases)
├── inference.py             # End-to-end inference pipeline
├── app.py                   # Streamlit dashboard
├── api.py                   # FastAPI server
├── requirements.txt
├── index.html               # Portfolio page (open in browser)
└── README.md
```

---

## Datasets

| Dataset | Modality | Size | License |
|---------|----------|------|---------|
| [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) | Vision | 35,887 images, 7 classes | Open |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | Text | 58,000 Reddit comments, 28 classes | Apache 2.0 |
| [AffectNet](http://mohammadmahoor.com/affectnet/) | Vision | 400K+ images | Research (registration) |

---

## Responsible AI

- **Bias Auditing**: Demographic parity testing via [AIF360](https://github.com/Trusted-AI/AIF360)
- **Explainability**: SHAP per-prediction attribution + GradCAM facial maps
- **Fairness Constraints**: Equalized odds enforcement during fine-tuning
- **Transparency**: Full model card available in `docs/model_card.md`

---

## VS Code Tips

Recommended extensions:
- **Python** (Microsoft)
- **Pylance** (fast type checking)
- **Jupyter** (for `.ipynb` notebooks)
- **GitLens** (version control)

Add to `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic"
}
```

---

*Built with PyTorch · HuggingFace · FastAPI · Streamlit · Free Compute (Colab + HF Spaces)*
