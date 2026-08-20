"""
ByteLM: small causal byte language model for conditional entropy estimation.

Pre-trained separately (Phase 4 Stage 1), then frozen.
Exposes compute_entropy(byte_seq) -> (L,) per-position conditional entropy
H(byte_{i+1} | bytes_{0..i}).

Architecture:
  Embedding(259, 64) + learned positional
  2 causal TransformerDecoder-style layers, dim=64, heads=4
  Linear(64, 256) next-byte logit head  (predicts byte values 0-255 only)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neurinferno.model.encoder import VOCAB, PAD_IDX


_LM_DIM    = 64
_LM_HEADS  = 4
_LM_FF     = 256
_LM_LAYERS = 2


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask: True where attention is forbidden."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class CausalTransformerLayer(nn.Module):
    """Single causal (decoder-style) transformer layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1     = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm
        xn = self.norm1(x)
        attn_out, _ = self.self_attn(
            xn, xn, xn,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class ByteLM(nn.Module):
    """
    Causal byte language model.

    compute_entropy(x) returns H(byte_{i+1} | bytes_{0..i}) for each position i.
    """

    def __init__(
        self,
        d_model:  int   = _LM_DIM,
        n_heads:  int   = _LM_HEADS,
        d_ff:     int   = _LM_FF,
        n_layers: int   = _LM_LAYERS,
        max_len:  int   = 512,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embedding     = nn.Embedding(VOCAB, d_model, padding_idx=PAD_IDX)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.embed_drop    = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        # Predicts next byte in 0-255 (not PAD/BOS/EOS)
        self.lm_head    = nn.Linear(d_model, 256)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L) long — byte indices

        Returns
        -------
        logits : (B, L, 256) — next-byte logits at each position
        """
        B, L = x.shape
        device = x.device

        pos_ids  = torch.arange(L, device=device).unsqueeze(0)
        tok_emb  = self.embedding(x)
        pos_emb  = self.pos_embedding(pos_ids)
        h        = self.embed_drop(tok_emb + pos_emb)

        causal   = _causal_mask(L, device)
        pad_mask = (x == PAD_IDX)

        for layer in self.layers:
            h = layer(h, attn_mask=causal, key_padding_mask=pad_mask)

        h      = self.final_norm(h)
        logits = self.lm_head(h)  # (B, L, 256)
        return logits

    @torch.no_grad()
    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute H(byte_{i+1} | bytes_{0..i}) at each position.

        Parameters
        ----------
        x : (B, L) long — byte indices (may include PAD at end)

        Returns
        -------
        entropy : (B, L) float — per-position conditional entropy in nats.
                  Position 0 = entropy of byte 1 given byte 0.
                  The last position has no next-byte; it is set to 0.
        """
        logits = self.forward(x)                        # (B, L, 256)
        log_probs = F.log_softmax(logits, dim=-1)       # (B, L, 256)
        probs     = log_probs.exp()                     # (B, L, 256)
        # Shannon entropy: -sum(p * log p)
        ent = -(probs * log_probs).sum(dim=-1)          # (B, L)
        # Shift: entropy[i] = H(byte[i+1] | context[0..i])
        # The logits at position i predict byte[i+1], so entropy is already
        # aligned: entropy[i] = H(next byte | bytes 0..i).
        # The very last position has no "next byte" — zero it out.
        ent[:, -1] = 0.0
        return ent

    @torch.no_grad()
    def compute_entropy_batched(
        self,
        x: torch.Tensor,         # (B, N, L) long
    ) -> torch.Tensor:           # (B, N, L) float
        """Convenience wrapper that handles the (B, N, L) → (B*N, L) fold."""
        B, N, L = x.shape
        x_flat = rearrange(x, "b n l -> (b n) l")
        ent_flat = self.compute_entropy(x_flat)         # (B*N, L)
        return rearrange(ent_flat, "(b n) l -> b n l", b=B, n=N)
