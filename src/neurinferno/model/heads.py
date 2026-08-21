"""
Prediction heads.

FieldTypeHead  — per-byte classification into 17 taxonomy classes.
BoundaryHead   — per-gap binary prediction using pairwise position features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurinferno.data_generation.label_format import FIELD_TYPES

N_FIELD_TYPES = len(FIELD_TYPES)  # 17


class FieldTypeHead(nn.Module):
    """
    Per-byte field-type classifier (original, h-only).

    Input  : h  (B, N, L, D)
    Output : logits  (B, N, L, 17)
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, N_FIELD_TYPES),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (B, N, L, 17)


class RelationalFieldTypeHead(nn.Module):
    """
    Relational per-byte field-type classifier.

    Gives the type head the same evidence as BoundaryHead: position
    representation h, per-byte LM entropy, and cross-message variance.
    A bidirectional GRU enforces field run-contiguity without a learned
    transition prior (safer than CRF under strict LOPO).

    Input:
        h         (B, N, L, D)
        entropy   (B, N, L)       — ByteLM conditional entropy
        cross_var (B, N, L, D)    — cross-message variance from encoder
        pad_mask  (B, N, L) bool  — True where PAD
    Output:
        logits    (B, N, L, 17)
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        feat_dim = 2 * d_model + 1  # h + cross_var + entropy scalar
        self.proj = nn.Linear(feat_dim, d_model)
        # BiGRU: each direction has d_model//2 hidden → concat = d_model out
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(d_model, N_FIELD_TYPES)

    def forward(
        self,
        h: torch.Tensor,  # (B, N, L, D)
        entropy: torch.Tensor,  # (B, N, L)
        cross_var: torch.Tensor,  # (B, N, L, D)
        pad_mask: torch.Tensor,  # (B, N, L) bool  True=PAD
    ) -> torch.Tensor:  # (B, N, L, 17)
        B, N, L, D = h.shape
        feat = torch.cat([h, entropy.unsqueeze(-1), cross_var], dim=-1)  # (B,N,L,2D+1)
        feat = F.gelu(self.proj(feat))  # (B,N,L,D)
        # Zero padded positions so the backward GRU doesn't start from PAD content
        feat = feat.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        gru_out, _ = self.gru(feat.view(B * N, L, D))  # (B*N, L, D)
        return self.head(gru_out.view(B, N, L, D))  # (B, N, L, 17)


class BoundaryHead(nn.Module):
    """
    Per-gap boundary predictor.

    Feature vector per gap (between position i and i+1):
      h_left    (D)   — encoder repr at position i
      h_right   (D)   — encoder repr at position i+1
      diff      (D)   — h_left - h_right
      prod      (D)   — h_left * h_right
      ent_right (1)   — LM entropy at i+1  (zeroed if disable_entropy=True)
      var_left  (D)   — cross-msg variance at i
      var_right (D)   — cross-msg variance at i+1
    Total: 6D + 1

    Input  : h (B,N,L,D), entropy (B,N,L), cross_var (B,N,L,D)
    Output : logits (B, N, L-1, 1)
    """

    def __init__(self, d_model: int = 128, disable_entropy: bool = False) -> None:
        super().__init__()
        self.disable_entropy = disable_entropy
        feat_dim = 6 * d_model + 1
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def boundary_features(
        self,
        h: torch.Tensor,  # (B, N, L, D)
        entropy: torch.Tensor,  # (B, N, L)
        cross_var: torch.Tensor,  # (B, N, L, D)
    ) -> torch.Tensor:  # (B, N, L-1, 6D+1)
        h_left = h[..., :-1, :]
        h_right = h[..., 1:, :]
        diff = h_left - h_right
        prod = h_left * h_right

        if self.disable_entropy:
            ent_right = torch.zeros(
                *h.shape[:-2], h.shape[-2] - 1, 1, device=h.device, dtype=h.dtype
            )
        else:
            ent_right = entropy[..., 1:].unsqueeze(-1)

        var_left = cross_var[..., :-1, :]
        var_right = cross_var[..., 1:, :]

        return torch.cat(
            [h_left, h_right, diff, prod, ent_right, var_left, var_right],
            dim=-1,
        )  # (B, N, L-1, 6D+1)

    def forward(
        self,
        h: torch.Tensor,  # (B, N, L, D)
        entropy: torch.Tensor,  # (B, N, L)
        cross_var: torch.Tensor,  # (B, N, L, D)
    ) -> torch.Tensor:  # (B, N, L-1, 1)
        feat = self.boundary_features(h, entropy, cross_var)
        logits = self.net(feat)
        return logits


class BIOTaggingHead(nn.Module):
    """
    Ablation baseline: per-byte BIO tagger.

    Predicts B (beginning) or I (inside) for each byte.
    A boundary between gap i exists when byte i+1 is predicted as B.

    Input  : h (B, N, L, D)
    Output : bio_logits (B, N, L, 2)  — [I_score, B_score]
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (B, N, L, 2)

    @staticmethod
    def bio_to_boundary_logits(bio_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert per-byte BIO logits to per-gap boundary logits.

        A gap i has a boundary when byte i+1 has label B (index 1).
        Return shape: (B, N, L-1, 1).
        """
        # B-score minus I-score for bytes 1..L-1 gives gap boundary score
        b_score = bio_logits[..., 1:, 1]  # (B, N, L-1)  B-class logit
        i_score = bio_logits[..., 1:, 0]  # (B, N, L-1)  I-class logit
        return (b_score - i_score).unsqueeze(-1)  # (B, N, L-1, 1)
