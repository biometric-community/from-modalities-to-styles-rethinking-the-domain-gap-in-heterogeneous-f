"""Contrastive loss for paired VIS(HR) / target(LR) embeddings (paper Sec. III)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_pairs(emb_s: torch.Tensor, emb_t: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """Positive: same-index HR–LR; negatives: shuffled LR.

    Uses Euclidean distance as in classic contrastive formulation.
    """
    d_pos = (emb_s - emb_t).pow(2).sum(dim=1).sqrt()
    # negatives via roll
    emb_neg = torch.roll(emb_t, shifts=1, dims=0)
    d_neg = (emb_s - emb_neg).pow(2).sum(dim=1).sqrt()
    loss_pos = d_pos.pow(2)
    loss_neg = F.relu(margin - d_neg).pow(2)
    return 0.5 * (loss_pos + loss_neg).mean()
