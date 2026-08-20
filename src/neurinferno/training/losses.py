"""
Loss functions for binary protocol field inference.

L_total = L_boundary + 0.5 * L_type + 0.2 * L_consistency
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurinferno.data_generation.label_format import FIELD_TYPE_IDS

_UNK = FIELD_TYPE_IDS["UNKNOWN"]


# ── Focal loss helper ────────────────────────────────────────────────────────

def focal_bce_loss(
    logits:  torch.Tensor,   # (N,)
    targets: torch.Tensor,   # (N,) float 0/1
    gamma:   float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary focal loss: FL = -(1-p_t)^gamma * log(p_t)."""
    bce    = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight)
    p_t    = torch.exp(-bce)                       # ≈ sigmoid prob of correct class
    focal  = ((1.0 - p_t) ** gamma) * bce
    return focal.mean()


# ── Per-batch positive weight ────────────────────────────────────────────────

def _pos_weight(targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """num_neg / num_pos, clamped to [1, 100]."""
    n_pos = targets.sum().clamp(min=eps)
    n_neg = (1.0 - targets).sum().clamp(min=eps)
    return (n_neg / n_pos).clamp(1.0, 100.0)


# ── Individual loss components ───────────────────────────────────────────────

def boundary_loss(
    logits:   torch.Tensor,  # (B, N, L-1, 1) or (M,)
    targets:  torch.Tensor,  # (B, N, L-1) or (M,) float
    gap_mask: torch.Tensor,  # (B, N, L-1) bool — True where valid
    use_focal: bool = True,
    gamma:     float = 2.0,
) -> torch.Tensor:
    """BCE (focal) over valid gaps only, with dynamic positive weighting."""
    logits_flat  = logits.squeeze(-1)[gap_mask]   # (M,)
    targets_flat = targets[gap_mask]               # (M,)

    if logits_flat.numel() == 0:
        return logits.sum() * 0.0  # zero, keeps grad graph alive

    pw = _pos_weight(targets_flat).to(logits.device)
    if use_focal:
        return focal_bce_loss(logits_flat, targets_flat, gamma=gamma,
                              pos_weight=pw.unsqueeze(0))
    return F.binary_cross_entropy_with_logits(
        logits_flat, targets_flat,
        pos_weight=pw,
    )


def type_loss(
    logits:    torch.Tensor,  # (B, N, L, 17)
    targets:   torch.Tensor,  # (B, N, L) long
    byte_mask: torch.Tensor,  # (B, N, L) bool — True where real (not PAD)
    use_focal: bool  = False,
    gamma:     float = 2.0,
) -> torch.Tensor:
    """Cross-entropy (or focal CE) over real bytes; ignore UNKNOWN (index 0)."""
    valid = byte_mask & (targets != _UNK)
    if valid.sum() == 0:
        return logits.sum() * 0.0

    logits_flat  = logits[valid]    # (M, 17)
    targets_flat = targets[valid]   # (M,)

    if use_focal:
        log_p = F.log_softmax(logits_flat, dim=-1)
        nll   = F.nll_loss(log_p, targets_flat, reduction='none')  # (M,)
        p_t   = torch.exp(-nll)
        return ((1.0 - p_t) ** gamma * nll).mean()

    return F.cross_entropy(logits_flat, targets_flat)


def consistency_loss(
    boundary_probs: torch.Tensor,  # (B, N, L-1)
    ft_labels:      torch.Tensor,  # (B, N, L) long
    gap_mask:       torch.Tensor,  # (B, N, L-1) bool
) -> torch.Tensor:
    """
    MSE between predicted boundary probability and the type-disagreement signal.

    type_disagree[i] = 1  iff  ft_labels[..., i] != ft_labels[..., i+1]
    """
    type_disagree = (ft_labels[..., :-1] != ft_labels[..., 1:]).float()
    if gap_mask.sum() == 0:
        return boundary_probs.sum() * 0.0

    return F.mse_loss(boundary_probs[gap_mask], type_disagree[gap_mask])


# ── Combined loss ────────────────────────────────────────────────────────────

@dataclass
class LossWeights:
    boundary:    float = 1.0
    type_:       float = 0.0
    consistency: float = 0.0


@dataclass
class LossBreakdown:
    total:       torch.Tensor
    boundary:    torch.Tensor
    type_:       torch.Tensor
    consistency: torch.Tensor

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}loss_total":       self.total.item(),
            f"{prefix}loss_boundary":    self.boundary.item(),
            f"{prefix}loss_type":        self.type_.item(),
            f"{prefix}loss_consistency": self.consistency.item(),
        }


def compute_total_loss(
    boundary_logits:     torch.Tensor,  # (B, N, L-1, 1)
    field_logits:        torch.Tensor,  # (B, N, L, 17)
    bg_labels:           torch.Tensor,  # (B, N, L-1)
    ft_labels:           torch.Tensor,  # (B, N, L)
    pad_mask:            torch.Tensor,  # (B, N, L)  True=PAD
    weights:             LossWeights | None = None,
    use_focal:           bool = True,   # focal BCE for boundary loss
    use_focal_type:      bool = False,  # focal CE  for field-type loss
    use_consistency_loss: bool = True,
) -> LossBreakdown:
    """Compute and return all loss components."""
    if weights is None:
        weights = LossWeights()

    byte_mask = ~pad_mask
    gap_mask  = byte_mask[..., :-1] & byte_mask[..., 1:]

    l_bd = boundary_loss(boundary_logits, bg_labels, gap_mask, use_focal=use_focal)
    l_ty = type_loss(field_logits, ft_labels, byte_mask, use_focal=use_focal_type)

    if use_consistency_loss:
        l_co = consistency_loss(
            boundary_logits.squeeze(-1).sigmoid(), ft_labels, gap_mask)
    else:
        l_co = boundary_logits.sum() * 0.0  # zero, keeps graph alive

    l_total = (weights.boundary    * l_bd
             + weights.type_       * l_ty
             + weights.consistency * l_co)

    return LossBreakdown(total=l_total, boundary=l_bd, type_=l_ty, consistency=l_co)
