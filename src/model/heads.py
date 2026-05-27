"""
Prediction heads.

FieldTypeHead  — per-byte classification into 17 taxonomy classes.
BoundaryHead   — per-gap binary prediction using pairwise position features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data_generation.label_format import FIELD_TYPES

N_FIELD_TYPES = len(FIELD_TYPES)  # 17


class FieldTypeHead(nn.Module):
    """
    Per-byte field-type classifier.

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


class BoundaryHead(nn.Module):
    """
    Per-gap boundary predictor.

    Constructs pairwise features for each gap between adjacent bytes using the
    encoder representation h, the ByteLM conditional entropy, and the
    cross-message variance from the encoder.

    Feature vector per gap (between position i and i+1):
      h_left    (D)   — encoder repr at position i
      h_right   (D)   — encoder repr at position i+1
      diff      (D)   — h_left - h_right
      prod      (D)   — h_left * h_right
      ent_right (1)   — LM entropy at position i+1
      var_left  (D)   — cross-msg variance at position i
      var_right (D)   — cross-msg variance at position i+1
    Total: 6D + 1

    Input  : h (B,N,L,D), entropy (B,N,L), cross_var (B,N,L,D)
    Output : logits (B, N, L-1, 1)  — raw boundary score (apply sigmoid for prob)
    """

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        feat_dim = 6 * d_model + 1
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def boundary_features(
        h:          torch.Tensor,  # (B, N, L, D)
        entropy:    torch.Tensor,  # (B, N, L)
        cross_var:  torch.Tensor,  # (B, N, L, D)
    ) -> torch.Tensor:             # (B, N, L-1, 6D+1)
        h_left  = h[..., :-1, :]            # (B, N, L-1, D)
        h_right = h[..., 1:,  :]
        diff    = h_left - h_right          # (B, N, L-1, D)
        prod    = h_left * h_right          # (B, N, L-1, D)

        # Entropy at the right position of each gap  (B, N, L-1, 1)
        ent_right = entropy[..., 1:].unsqueeze(-1)

        var_left  = cross_var[..., :-1, :]  # (B, N, L-1, D)
        var_right = cross_var[..., 1:,  :]  # (B, N, L-1, D)

        return torch.cat(
            [h_left, h_right, diff, prod, ent_right, var_left, var_right],
            dim=-1,
        )  # (B, N, L-1, 6D+1)

    def forward(
        self,
        h:         torch.Tensor,  # (B, N, L, D)
        entropy:   torch.Tensor,  # (B, N, L)
        cross_var: torch.Tensor,  # (B, N, L, D)
    ) -> torch.Tensor:            # (B, N, L-1, 1)
        feat   = self.boundary_features(h, entropy, cross_var)
        logits = self.net(feat)
        return logits
