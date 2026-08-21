"""
ByteEncoder: byte-level transformer with cross-message statistics injection.

Input:  (B, N, L)  — B formats, N messages each, L bytes (padded to max_len)
Output: (B, N, L, D) per-byte representations, plus the last-layer cross-msg variance

The cross-message statistics injection is the key inductive bias:
  after each self-attention block we compute mean / max / var across the N
  messages in the same format and concatenate them to each token, then project
  back to D.  This makes per-offset correlations across messages directly
  learnable with no extra overhead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

# ── Special token indices ────────────────────────────────────────────────────
PAD_IDX = 256
BOS_IDX = 257
EOS_IDX = 258
VOCAB = 259  # 0-255 byte values + PAD + BOS + EOS


class CrossMsgTransformerLayer(nn.Module):
    """
    One transformer layer whose post-attention step injects cross-message stats.

    Sub-block order (pre-LayerNorm throughout for training stability):
      1. LN → self-attention  → residual
      2. LN → cross-msg stats projection → residual   (skipped if disable_cross_msg)
      3. LN → FFN              → residual
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        disable_cross_msg: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.disable_cross_msg = disable_cross_msg

        # — Self-attention —
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        # — Cross-message statistics injection —
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_msg_proj = nn.Linear(d_model * 4, d_model)
        self.drop2 = nn.Dropout(dropout)

        # — FFN —
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,  # (B, N, L, D)
        padding_mask: torch.Tensor | None,  # (B, N, L) bool, True=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return updated h and the cross-msg variance (for boundary features)."""
        B, N, L, D = h.shape

        # ── 1. Self-attention ──────────────────────────────────────────────
        h_norm = self.norm1(h)
        h_flat = rearrange(h_norm, "b n l d -> (b n) l d")

        attn_mask_flat: torch.Tensor | None = None
        if padding_mask is not None:
            # MultiheadAttention key_padding_mask: (B, L), True = ignore
            attn_mask_flat = rearrange(padding_mask, "b n l -> (b n) l")

        attn_out, _ = self.self_attn(
            h_flat,
            h_flat,
            h_flat,
            key_padding_mask=attn_mask_flat,
        )
        h = h + self.drop1(rearrange(attn_out, "(b n) l d -> b n l d", b=B, n=N))

        # ── 2. Cross-message statistics injection ─────────────────────────
        h_norm = self.norm2(h)

        if self.disable_cross_msg:
            # Ablation: skip cross-msg stats; pass zeros for mean/max/var slots
            zeros = torch.zeros_like(h_norm)
            combined = torch.cat([h_norm, zeros, zeros, zeros], dim=-1)
            var_m = zeros
        else:
            # Zero out padding positions before aggregating across messages
            if padding_mask is not None:
                pad_mask_4d = padding_mask.unsqueeze(-1).float()  # (B, N, L, 1)
                h_agg = h_norm * (1.0 - pad_mask_4d)  # zero padding
            else:
                h_agg = h_norm

            mean_m = h_agg.mean(dim=1, keepdim=True).expand_as(h_norm)
            max_m = h_agg.max(dim=1, keepdim=True).values.expand_as(h_norm)
            var_m = h_agg.var(dim=1, unbiased=False, keepdim=True).expand_as(h_norm)
            combined = torch.cat([h_norm, mean_m, max_m, var_m], dim=-1)  # (B,N,L,4D)

        h = h + self.drop2(self.cross_msg_proj(combined))

        # ── 3. FFN ────────────────────────────────────────────────────────
        h_norm = self.norm3(h)
        h_flat = rearrange(h_norm, "b n l d -> (b n) l d")
        ffn_out = self.ffn(h_flat)
        h = h + rearrange(ffn_out, "(b n) l d -> b n l d", b=B, n=N)

        # Return variance from this layer (used by boundary head if last layer)
        return h, var_m  # var_m: (B, N, L, D)


class ByteEncoder(nn.Module):
    """
    Byte-level encoder with cross-message attention.

    Accepts a batch of (B, N, L) byte index tensors and returns
    per-byte representations h of shape (B, N, L, D) together with
    the final-layer cross-message variance tensor for use in the
    boundary prediction head.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
        disable_cross_msg: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.disable_cross_msg = disable_cross_msg

        # Byte + special-token embedding
        self.embedding = nn.Embedding(VOCAB, d_model, padding_idx=PAD_IDX)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.embed_drop = nn.Dropout(dropout)

        # Transformer layers with cross-message injection
        self.layers = nn.ModuleList(
            [
                CrossMsgTransformerLayer(
                    d_model, n_heads, d_ff, dropout, disable_cross_msg=disable_cross_msg
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.cross_msg_proj.weight)
            for ffn_layer in layer.ffn:
                if isinstance(ffn_layer, nn.Linear):
                    nn.init.xavier_uniform_(ffn_layer.weight)

    def make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return bool mask (B, N, L), True where x == PAD_IDX."""
        return x == PAD_IDX

    def forward(
        self,
        x: torch.Tensor,  # (B, N, L) long
        padding_mask: torch.Tensor | None = None,  # (B, N, L) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h          : (B, N, L, D)  per-byte representations
        cross_var  : (B, N, L, D)  final-layer cross-message variance
        """
        B, N, L = x.shape
        assert L <= self.max_len, f"Sequence length {L} > max_len {self.max_len}"

        if padding_mask is None:
            padding_mask = self.make_padding_mask(x)

        # — Embeddings —
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        # Expand to (B*N, L) for pos lookup, then reshape
        tok_emb = self.embedding(x)  # (B, N, L, D)
        pos_emb = self.pos_embedding(pos_ids)  # (1, L, D) → broadcasts to (B,N,L,D)
        h = self.embed_drop(tok_emb + pos_emb)

        # — Transformer layers —
        cross_var = torch.zeros_like(h)
        for layer in self.layers:
            h, cross_var = layer(h, padding_mask)

        h = self.final_norm(h)

        # Zero out padding positions in the output
        if padding_mask is not None:
            h = h * (~padding_mask).unsqueeze(-1).float()

        return h, cross_var
