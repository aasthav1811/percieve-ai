"""
download_pretrained.py
----------------------
Downloads community-trained models from HuggingFace Hub.
No training required — models are already fine-tuned on FER2013 / GoEmotions.

Run once:
    python download_pretrained.py

Models downloaded:
  Vision : motheecreator/vit-Facial-Expression-Recognition  (ViT on FER2013, 7-class)
  Text   : SamLowe/roberta-base-go_emotions                 (RoBERTa on GoEmotions, 28-class)
  
Total download size: ~900 MB
"""

from pathlib import Path
from huggingface_hub import snapshot_download

HF_CACHE = Path("models/hf_pretrained")
HF_CACHE.mkdir(parents=True, exist_ok=True)

VISION_MODEL = "motheecreator/vit-Facial-Expression-Recognition"
TEXT_MODEL   = "SamLowe/roberta-base-go_emotions"

def download(repo_id, subfolder):
    dest = HF_CACHE / subfolder
    if dest.exists() and any(dest.iterdir()):
        print(f"  ✓ Already downloaded: {subfolder}")
        return dest
    print(f"  ↓ Downloading {repo_id}  →  {dest}")
    snapshot_download(repo_id=repo_id, local_dir=str(dest))
    print(f"  ✓ Saved to {dest}")
    return dest

if __name__ == "__main__":
    print("=" * 55)
    print(" perceive_ai — Pretrained Model Download")
    print("=" * 55)
    print("\n[1/2] Vision encoder (ViT — FER2013 fine-tuned)...")
    download(VISION_MODEL, "vision")
    print("\n[2/2] Text encoder (RoBERTa — GoEmotions fine-tuned)...")
    download(TEXT_MODEL, "text")
    print("\n✓ All models ready. Run: uvicorn api:app --port 8000 --reload")