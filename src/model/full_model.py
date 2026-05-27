"""
FullModel: wires ByteEncoder + ByteLM + FieldTypeHead + BoundaryHead.

Forward signature:
    x : (B, N, L) long  — byte indices (PAD_IDX for padding)

Returns a ModelOutput dataclass with:
    boundary_logits  : (B, N, L-1, 1) — raw boundary scores
    field_logits     : (B, N, L,   17) — per-byte field-type logits
    entropy          : (B, N, L)       — ByteLM conditional entropy (for viz)
    padding_mask     : (B, N, L)       — True where x == PAD_IDX
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.encoder import ByteEncoder, PAD_IDX
from src.model.byte_lm import ByteLM
from src.model.heads import FieldTypeHead, BoundaryHead


@dataclass
class ModelOutput:
    boundary_logits: torch.Tensor   # (B, N, L-1, 1)
    field_logits:    torch.Tensor   # (B, N, L, 17)
    entropy:         torch.Tensor   # (B, N, L)
    padding_mask:    torch.Tensor   # (B, N, L) bool


class FullModel(nn.Module):
    """
    End-to-end binary protocol field-inference model.

    Parameters
    ----------
    d_model   : encoder hidden dimension (default 128)
    n_heads   : attention heads (default 4)
    d_ff      : FFN width (default 512)
    n_layers  : encoder transformer layers (default 4)
    max_len   : max byte-sequence length (default 512)
    lm_frozen : if True, freeze ByteLM weights during training
    """

    def __init__(
        self,
        d_model:   int   = 128,
        n_heads:   int   = 4,
        d_ff:      int   = 512,
        n_layers:  int   = 4,
        max_len:   int   = 512,
        dropout:   float = 0.1,
        lm_frozen: bool  = True,
    ) -> None:
        super().__init__()

        self.encoder = ByteEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_len=max_len, dropout=dropout,
        )
        self.byte_lm = ByteLM(max_len=max_len, dropout=dropout)

        self.field_type_head = FieldTypeHead(d_model=d_model)
        self.boundary_head   = BoundaryHead(d_model=d_model)

        if lm_frozen:
            self._freeze_lm()

    def _freeze_lm(self) -> None:
        for p in self.byte_lm.parameters():
            p.requires_grad = False

    def unfreeze_lm(self) -> None:
        for p in self.byte_lm.parameters():
            p.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Parameters
        ----------
        x : (B, N, L) long  — byte indices, PAD_IDX where padded

        Returns
        -------
        ModelOutput with boundary_logits, field_logits, entropy, padding_mask
        """
        B, N, L = x.shape
        padding_mask = (x == PAD_IDX)  # (B, N, L)

        # — Encoder —
        h, cross_var = self.encoder(x, padding_mask)   # (B,N,L,D), (B,N,L,D)

        # — ByteLM entropy (no grad if frozen) —
        entropy = self.byte_lm.compute_entropy_batched(x)  # (B, N, L)

        # — Prediction heads —
        field_logits = self.field_type_head(h)                           # (B,N,L,17)
        boundary_logits = self.boundary_head(h, entropy, cross_var)      # (B,N,L-1,1)

        return ModelOutput(
            boundary_logits=boundary_logits,
            field_logits=field_logits,
            entropy=entropy,
            padding_mask=padding_mask,
        )

    # ── Inference helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return hard predictions.

        Returns
        -------
        field_preds    : (B, N, L)   long — argmax over 17 classes
        boundary_preds : (B, N, L-1) bool — sigmoid(logit) > 0.5
        """
        out = self.forward(x)
        field_preds    = out.field_logits.argmax(dim=-1)
        boundary_preds = (out.boundary_logits.squeeze(-1).sigmoid() > 0.5)
        return field_preds, boundary_preds

    # ── Param count utility ──────────────────────────────────────────────────

    def param_count(self) -> dict[str, int]:
        """Return parameter counts broken down by submodule."""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "encoder":         count(self.encoder),
            "byte_lm":         count(self.byte_lm),
            "field_type_head": count(self.field_type_head),
            "boundary_head":   count(self.boundary_head),
            "total":           count(self),
            "trainable":       sum(p.numel() for p in self.parameters()
                                   if p.requires_grad),
        }
