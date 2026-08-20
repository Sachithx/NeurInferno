"""
FullModel: wires ByteEncoder + ByteLM + FieldTypeHead + BoundaryHead.

Forward signature:
    x : (B, N, L) long  — byte indices (PAD_IDX for padding)

Returns a ModelOutput dataclass with:
    boundary_logits  : (B, N, L-1, 1) — raw boundary scores
    field_logits     : (B, N, L,   17) — per-byte field-type logits
    entropy          : (B, N, L)       — ByteLM conditional entropy (for viz)
    padding_mask     : (B, N, L)       — True where x == PAD_IDX

Ablation flags (passed to constructor):
    disable_cross_msg        : skip cross-message statistics injection in encoder
    disable_entropy          : zero out entropy feature in BoundaryHead
    use_bio_head             : replace BoundaryHead with BIOTaggingHead
    use_relational_type_head : use RelationalFieldTypeHead (h+entropy+cross_var+BiGRU)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from neurinferno.model.encoder import ByteEncoder, PAD_IDX
from neurinferno.model.byte_lm import ByteLM
from neurinferno.model.heads import (FieldTypeHead, RelationalFieldTypeHead,
                             BoundaryHead, BIOTaggingHead)


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
    d_model            : encoder hidden dimension (default 128)
    n_heads            : attention heads (default 4)
    d_ff               : FFN width (default 512)
    n_layers           : encoder transformer layers (default 4)
    max_len            : max byte-sequence length (default 512)
    lm_frozen          : if True, freeze ByteLM weights during training
    disable_cross_msg  : ablation — skip cross-message statistics injection
    disable_entropy    : ablation — zero out entropy in boundary features
    use_bio_head       : ablation — use BIO tagger instead of pairwise boundary head
    """

    def __init__(
        self,
        d_model:           int   = 128,
        n_heads:           int   = 4,
        d_ff:              int   = 512,
        n_layers:          int   = 4,
        max_len:           int   = 512,
        dropout:           float = 0.1,
        lm_frozen:                bool  = True,
        disable_cross_msg:        bool  = False,
        disable_entropy:          bool  = False,
        use_bio_head:             bool  = False,
        use_relational_type_head: bool  = False,
        lm_d_model:               int   = 64,
        lm_n_heads:        int   = 4,
        lm_d_ff:           int   = 256,
        lm_n_layers:       int   = 2,
    ) -> None:
        super().__init__()

        self.use_bio_head             = use_bio_head
        self.use_relational_type_head = use_relational_type_head

        self.encoder = ByteEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_len=max_len, dropout=dropout,
            disable_cross_msg=disable_cross_msg,
        )
        self.byte_lm = ByteLM(
            d_model=lm_d_model, n_heads=lm_n_heads,
            d_ff=lm_d_ff, n_layers=lm_n_layers,
            max_len=max_len, dropout=dropout,
        )

        if use_relational_type_head:
            self.field_type_head = RelationalFieldTypeHead(d_model=d_model)
        else:
            self.field_type_head = FieldTypeHead(d_model=d_model)
        if use_bio_head:
            self.boundary_head = BIOTaggingHead(d_model=d_model)
        else:
            self.boundary_head = BoundaryHead(d_model=d_model,
                                              disable_entropy=disable_entropy)

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
        if self.use_relational_type_head:
            field_logits = self.field_type_head(h, entropy, cross_var, padding_mask)
        else:
            field_logits = self.field_type_head(h)                       # (B,N,L,17)
        if self.use_bio_head:
            bio_logits      = self.boundary_head(h)                      # (B,N,L,2)
            boundary_logits = BIOTaggingHead.bio_to_boundary_logits(bio_logits)  # (B,N,L-1,1)
        else:
            boundary_logits = self.boundary_head(h, entropy, cross_var)  # (B,N,L-1,1)

        return ModelOutput(
            boundary_logits=boundary_logits,
            field_logits=field_logits,
            entropy=entropy,
            padding_mask=padding_mask,
        )

    # ── Inference helpers ────────────────────────────────────────────────────

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run encoder + ByteLM without prediction heads.

        Returns
        -------
        h            : (B, N, L, D)   — frozen encoder output
        entropy      : (B, N, L)      — ByteLM conditional entropy
        cross_var    : (B, N, L, D)   — cross-message variance
        padding_mask : (B, N, L) bool — True where PAD
        """
        B, N, L = x.shape
        padding_mask = (x == PAD_IDX)
        h, cross_var = self.encoder(x, padding_mask)
        entropy = self.byte_lm.compute_entropy_batched(x)
        return h, entropy, cross_var, padding_mask

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
