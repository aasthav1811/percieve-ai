#!/usr/bin/env python3
"""
data/download.py
----------------
Downloads FER2013 (facial) and GoEmotions (text) datasets.
Run from project root:  python data/download.py
Free datasets — no login required.
"""

import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent

def download_fer2013():
    """
    FER2013 via Kaggle CLI (free).
    Requires: pip install kaggle, and ~/.kaggle/kaggle.json token.
    Falls back to instructions if Kaggle CLI not available.
    """
    print("\n[FER2013] Checking for Kaggle CLI...")
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("[FER2013] Downloading via Kaggle CLI...")
            subprocess.run([
                "kaggle", "competitions", "download",
                "-c", "challenges-in-representation-learning-facial-expression-recognition-challenge",
                "-p", str(DATA_DIR / "fer2013")
            ], check=True)
            # Unzip
            fer_zip = DATA_DIR / "fer2013" / "challenges-in-representation-learning-facial-expression-recognition-challenge.zip"
            if fer_zip.exists():
                with zipfile.ZipFile(fer_zip, "r") as z:
                    z.extractall(DATA_DIR / "fer2013")
                print("[FER2013] ✓ Downloaded and extracted to data/fer2013/")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[FER2013] Kaggle CLI not found. Manual download steps:")
        print("  1. Go to: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
        print("  2. Download fer2013.csv")
        print("  3. Place at: data/fer2013/fer2013.csv")
        (DATA_DIR / "fer2013").mkdir(exist_ok=True)

def download_goemotions():
    """GoEmotions (Google) — 27-class emotion labels on Reddit comments."""
    print("\n[GoEmotions] Downloading from HuggingFace datasets...")
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "simplified", trust_remote_code=True)
        save_path = DATA_DIR / "goemotions"
        save_path.mkdir(exist_ok=True)
        ds.save_to_disk(str(save_path))
        print(f"[GoEmotions] ✓ Saved to {save_path}")
        print(f"  Train: {len(ds['train'])} examples")
        print(f"  Test:  {len(ds['test'])} examples")
    except Exception as e:
        print(f"[GoEmotions] Error: {e}")
        print("  Run:  pip install datasets  then retry.")

def download_affectnet_sample():
    """
    AffectNet is gated. We use AffectNet-HQ subset on HuggingFace as fallback.
    Full AffectNet: http://mohammadmahoor.com/affectnet/
    """
    print("\n[AffectNet] Note: Full AffectNet requires registration at mohammadmahoor.com")
    print("  For development, using a public 8-class subset via HuggingFace.")
    try:
        from datasets import load_dataset
        ds = load_dataset("Panlizhi/AffectNet-8", split="train[:2000]", trust_remote_code=True)
        save_path = DATA_DIR / "affectnet_sample"
        save_path.mkdir(exist_ok=True)
        ds.save_to_disk(str(save_path))
        print(f"[AffectNet] ✓ Sample (2000 images) saved to {save_path}")
    except Exception as e:
        print(f"[AffectNet] Could not download sample: {e}")

if __name__ == "__main__":
    print("=" * 55)
    print(" Multimodal Emotional Intelligence — Data Setup")
    print("=" * 55)
    (DATA_DIR / "fer2013").mkdir(exist_ok=True)
    download_fer2013()
    download_goemotions()
    download_affectnet_sample()
    print("\n✓ Data setup complete. Run train.py to begin training.")
