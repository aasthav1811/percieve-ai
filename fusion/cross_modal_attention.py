"""
fusion/cross_modal_attention.py
--------------------------------
Cross-modal attention fusion layer.

Architecture:
  visual_embed (512-d) ──┐
                          ├──► cross-attn → concat → MLP → 256-d ──► heads
  text_embed   (768-d) ──┘

Two lightweight multi-head cross-attention blocks let each modality
attend to the other, then weighted concatenation + a 3-layer MLP
projects to a shared 256-d emotion space.

Output heads:
  - emotion classifier (7-class)
  - valence regressor  (scalar, [-1, +1])
  - arousal regressor  (scalar, [0, 1])
  - risk classifier    (low / medium / high)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

VISUAL_DIM  = 512
TEXT_DIM    = 768
FUSION_DIM  = 256
NUM_HEADS   = 8
NUM_EMOTIONS = 7
NUM_RISK_LEVELS = 3  # 0=low, 1=medium, 2=high

EMOTION_LABELS   = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
RISK_LABELS      = ["low", "medium", "high"]


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-modal attention + MLP fusion head.

    Step 1 — Project both embeddings to a shared d_model
    Step 2 — Visual attends to text (visual queries, text keys/values)
    Step 3 — Text attends to visual (text queries, visual keys/values)
    Step 4 — Learnable weighted concat → 3-layer MLP
    Step 5 — Task-specific heads
    """

    def __init__(self, d_model: int = 256, num_heads: int = NUM_HEADS, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # ── 1. Input projections ──
        self.vis_proj  = nn.Linear(VISUAL_DIM, d_model)
        self.text_proj = nn.Linear(TEXT_DIM,   d_model)

        # ── 2 & 3. Cross-modal attention ──
        # visual queries ← text keys/values
        self.vis_to_text_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # text queries ← visual keys/values
        self.text_to_vis_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_vis  = nn.LayerNorm(d_model)
        self.norm_text = nn.LayerNorm(d_model)

        # ── 4. Learnable modality weights ──
        # α = σ(w_vis) / (σ(w_vis) + σ(w_text))  — soft modality gating
        self.modal_gate = nn.Linear(d_model * 2, 2)

        # ── 4b. MLP projector ──
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FUSION_DIM, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
        )

        # ── 5. Task heads ──
        self.emotion_head = nn.Linear(FUSION_DIM, NUM_EMOTIONS)
        self.valence_head = nn.Sequential(nn.Linear(FUSION_DIM, 1), nn.Tanh())   # [-1, +1]
        self.arousal_head = nn.Sequential(nn.Linear(FUSION_DIM, 1), nn.Sigmoid())# [0, 1]
        self.risk_head    = nn.Linear(FUSION_DIM, NUM_RISK_LEVELS)

    def forward(
        self,
        visual_embed: torch.Tensor,   # (B, 512)
        text_embed:   torch.Tensor,   # (B, 768)
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
          fusion_embed  : (B, 256)
          emotion_logits: (B, 7)
          emotion_probs : (B, 7)
          valence       : (B, 1)   — [-1=negative, +1=positive]
          arousal       : (B, 1)   — [0=calm, 1=activated]
          risk_logits   : (B, 3)
          risk_probs    : (B, 3)
          modal_weights : (B, 2)   — [visual_w, text_w], sum to 1
        """
        # 1. Project to shared d_model
        v = self.vis_proj(visual_embed).unsqueeze(1)   # (B, 1, d)
        t = self.text_proj(text_embed).unsqueeze(1)    # (B, 1, d)

        # 2. Cross attention (residual)
        v_attn, _ = self.vis_to_text_attn(query=v, key=t, value=t)
        v = self.norm_vis(v + v_attn)                  # (B, 1, d)

        t_attn, _ = self.text_to_vis_attn(query=t, key=v, value=v)
        t = self.norm_text(t + t_attn)                 # (B, 1, d)

        v = v.squeeze(1)   # (B, d)
        t = t.squeeze(1)   # (B, d)

        # 4. Modality gating
        concat = torch.cat([v, t], dim=-1)             # (B, 2d)
        gates  = F.softmax(self.modal_gate(concat), dim=-1)  # (B, 2)
        v_w, t_w = gates[:, :1], gates[:, 1:]         # each (B, 1)
        weighted = torch.cat([v * v_w, t * t_w], dim=-1)     # (B, 2d)

        # 4b. MLP projector
        fusion_embed = self.mlp(weighted)              # (B, 256)

        # 5. Heads
        emotion_logits = self.emotion_head(fusion_embed)
        valence        = self.valence_head(fusion_embed)
        arousal        = self.arousal_head(fusion_embed)
        risk_logits    = self.risk_head(fusion_embed)

        return {
            "fusion_embed":   fusion_embed,
            "emotion_logits": emotion_logits,
            "emotion_probs":  F.softmax(emotion_logits, dim=-1),
            "valence":        valence,
            "arousal":        arousal,
            "risk_logits":    risk_logits,
            "risk_probs":     F.softmax(risk_logits, dim=-1),
            "modal_weights":  gates,              # (B, 2): [vis_w, text_w]
        }


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CrossModalAttentionFusion()
    model.eval()

    B = 4
    vis  = torch.randn(B, VISUAL_DIM)
    text = torch.randn(B, TEXT_DIM)

    with torch.no_grad():
        out = model(vis, text)

    print("Fusion Layer — forward pass:")
    for k, v in out.items():
        print(f"  {k:20s}: {tuple(v.shape)}")

    print(f"\nModal weights (vis | text):")
    for i, w in enumerate(out["modal_weights"]):
        print(f"  Sample {i}: visual={w[0]:.3f}  text={w[1]:.3f}")

    print("\n✓ Cross-modal fusion OK")
