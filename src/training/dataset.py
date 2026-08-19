"""
Dataset utilities for binary protocol field inference training.

Two batch modes:
  LM pretraining  — standard batching of individual byte sequences.
  Main/LOPO       — format-grouped batching: every item in a batch
                    comes from the same format/protocol so cross-message
                    statistics are meaningful.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from src.model.encoder import PAD_IDX
from src.data_generation.label_format import FIELD_TYPE_IDS

_UNK = FIELD_TYPE_IDS["UNKNOWN"]

# ── Held-out splits ──────────────────────────────────────────────────────────
# Structural validation protocols.  "coap" is listed for compatibility with
# the paper training run; it is simply skipped when absent from data/protocols.
# With the 12-protocol set this leaves only ospf in the val split (unless ospf
# itself is the LOPO/L4PO hold-out).  Do not change this list — training
# dynamics depend on it.
VAL_TIER2_PROTOCOLS: list[str] = ["coap", "ospf"]


def split_tier2_protocols(
    tier2_dir:         Path,
    exclude_protocols: list[str],
    n_val:             int  = 6,
    seed:              int  = 42,
) -> tuple[list[str], list[str]]:
    """
    From available Tier-2 protocols (minus excluded ones), randomly assign
    n_val protocols entirely to validation and the rest to training.

    Used only when --use_dynamic_val is passed (the paper run does not).

    Returns (train_protocols, val_protocols).
    """
    excl = set(exclude_protocols)
    available = sorted(
        d.name
        for d in tier2_dir.iterdir()
        if d.is_dir() and (d / "messages.jsonl").exists() and d.name not in excl
    )
    n_val = min(n_val, len(available) - 1)   # always keep ≥1 train protocol
    shuffled = list(available)
    random.Random(seed).shuffle(shuffled)
    val_protos   = shuffled[:n_val]
    train_protos = shuffled[n_val:]
    return train_protos, val_protos

# 12 protocols used for LOPO / L4PO
LOPO_PROTOCOLS: list[str] = [
    "arp", "bgp_raw", "dhcp", "dns", "icmp", "igmp",
    "ip", "modbus", "ntp", "ospf", "tcp", "udp",
]


# ── Record types ─────────────────────────────────────────────────────────────

@dataclass
class MessageRecord:
    bytes_hex:          str
    field_type_per_byte: list[int]
    boundary_per_gap:    list[int]
    format_id:           str

    @classmethod
    def from_dict(cls, d: dict) -> "MessageRecord":
        return cls(
            bytes_hex=d["bytes_hex"],
            field_type_per_byte=d["field_type_per_byte"],
            boundary_per_gap=d["boundary_per_gap"],
            format_id=d["format_id"],
        )

    def to_tensors(self, max_len: int = 512):
        """Return (x, ft, bg) with no padding — raw length."""
        raw = bytes.fromhex(self.bytes_hex)
        n   = min(len(raw), max_len)
        x   = torch.tensor(list(raw[:n]), dtype=torch.long)
        ft  = torch.tensor(self.field_type_per_byte[:n], dtype=torch.long)
        bg  = torch.tensor(self.boundary_per_gap[:n-1] if n > 1 else [], dtype=torch.float)
        return x, ft, bg


@dataclass
class FormatGroup:
    """All messages belonging to one format."""
    format_id: str
    messages:  list[MessageRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.messages)


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path, max_per_file: int | None = None) -> list[MessageRecord]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(MessageRecord.from_dict(json.loads(line)))
            if max_per_file and len(records) >= max_per_file:
                break
    return records


def load_tier1_groups(
    tier1_dir:        Path,
    val_fraction:     float = 0.10,
    seed:             int   = 42,
) -> tuple[list[FormatGroup], list[FormatGroup]]:
    """Load Tier 1, return (train_groups, val_groups) split by format."""
    all_dirs = sorted(tier1_dir.iterdir())
    rng      = random.Random(seed)
    rng.shuffle(all_dirs)

    n_val  = max(1, int(len(all_dirs) * val_fraction))
    val_dirs   = set(d.name for d in all_dirs[:n_val])
    train_dirs = all_dirs[n_val:]

    def _make_group(d: Path) -> FormatGroup:
        msgs = _load_jsonl(d / "messages.jsonl")
        return FormatGroup(format_id=d.name, messages=msgs)

    train = [_make_group(d) for d in train_dirs]
    val   = [_make_group(Path(tier1_dir / v))
             for v in val_dirs if (tier1_dir / v / "messages.jsonl").exists()]
    return train, val


def load_tier2_groups(
    tier2_dir:          Path,
    val_protocols:      list[str] | None = None,
    exclude_protocols:  list[str] | None = None,
    test_fraction:      float = 0.20,
    train_val_fraction: float = 0.0,
) -> tuple[list[FormatGroup], list[FormatGroup]]:
    """
    Load Tier 2, returning (train_groups, val_groups) split by protocol.

    val_protocols      — these protocols go entirely to val_groups.
                         Defaults to VAL_TIER2_PROTOCOLS.
    test_fraction      — fraction of val_protocol messages reserved for
                         run_eval (never exposed here). Only applies to
                         val_protocols, not to train protocols.
    train_val_fraction — fraction of each TRAIN protocol's 200-msg chunks
                         that are moved to val_groups (e.g. 0.10 = 10%).
                         Use this with dynamic val splits so the val set
                         also contains samples from the training families.
    """
    val_protocols     = set(val_protocols or VAL_TIER2_PROTOCOLS)
    exclude_protocols = set(exclude_protocols or [])

    train_groups, val_groups = [], []
    for proto_dir in sorted(tier2_dir.iterdir()):
        name = proto_dir.name
        if name in exclude_protocols:
            continue
        path = proto_dir / "messages.jsonl"
        if not path.exists():
            continue
        msgs = _load_jsonl(path)

        if name in val_protocols:
            # Reserve last test_fraction for held-out test evaluation
            n_keep = max(1, int(len(msgs) * (1.0 - test_fraction)))
            msgs = msgs[:n_keep]
            chunk_size = 200
            for i in range(0, len(msgs), chunk_size):
                chunk = msgs[i: i + chunk_size]
                if chunk:
                    val_groups.append(FormatGroup(
                        format_id=f"{name}_{i//chunk_size:04d}", messages=chunk))
        else:
            # Training protocol: optionally split last train_val_fraction chunks to val
            chunk_size = 200
            all_chunks: list[FormatGroup] = []
            for i in range(0, len(msgs), chunk_size):
                chunk = msgs[i: i + chunk_size]
                if chunk:
                    all_chunks.append(FormatGroup(
                        format_id=f"{name}_{i//chunk_size:04d}", messages=chunk))
            if train_val_fraction > 0 and all_chunks:
                n_val_chunks = max(1, round(len(all_chunks) * train_val_fraction))
                train_groups.extend(all_chunks[:-n_val_chunks])
                val_groups.extend(all_chunks[-n_val_chunks:])
            else:
                train_groups.extend(all_chunks)

    return train_groups, val_groups


# ── Format-batched dataset ────────────────────────────────────────────────────

class FormatBatchDataset(Dataset):
    """
    Wraps a list of FormatGroups.  __getitem__ returns a FormatGroup index.
    The actual per-message tensors are produced by the collate function.
    """

    def __init__(self, groups: list[FormatGroup]) -> None:
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> FormatGroup:
        return self.groups[idx]


def collate_format_batch(
    groups:  list[FormatGroup],
    n_msgs:  int  = 64,
    max_len: int  = 512,
    seed:    int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Sample n_msgs from each group, pad to the batch's max length,
    and return stacked tensors.

    Returns dict with keys:
      x          (B, N, L) long
      ft_labels  (B, N, L) long
      bg_labels  (B, N, L-1) float
      pad_mask   (B, N, L) bool  — True where padding
    """
    rng = random.Random(seed)
    B   = len(groups)
    all_x, all_ft, all_bg = [], [], []

    for grp in groups:
        msgs = grp.messages
        sampled = rng.choices(msgs, k=n_msgs) if len(msgs) < n_msgs \
                  else rng.sample(msgs, n_msgs)
        xs, fts, bgs = [], [], []
        for m in sampled:
            x, ft, bg = m.to_tensors(max_len)
            xs.append(x); fts.append(ft); bgs.append(bg)
        all_x.append(xs); all_ft.append(fts); all_bg.append(bgs)

    # Find global max length across the batch
    L = min(max(max(x.size(0) for x in xs) for xs in all_x), max_len)

    def _pad_seq(t: torch.Tensor, length: int, pad_val: int) -> torch.Tensor:
        if t.size(0) >= length:
            return t[:length]
        return torch.cat([t, torch.full((length - t.size(0),), pad_val, dtype=t.dtype)])

    x_out  = torch.zeros(B, n_msgs, L, dtype=torch.long)
    ft_out = torch.zeros(B, n_msgs, L, dtype=torch.long)
    bg_out = torch.zeros(B, n_msgs, max(L - 1, 1), dtype=torch.float)

    for b in range(B):
        for n in range(n_msgs):
            x_out[b, n]  = _pad_seq(all_x[b][n],  L, PAD_IDX)
            ft_out[b, n] = _pad_seq(all_ft[b][n], L, _UNK)
            bg_len        = max(L - 1, 1)
            bg_out[b, n] = _pad_seq(all_bg[b][n], bg_len, 0)

    bg_out = bg_out[..., :L-1] if L > 1 else bg_out[..., :0]

    pad_mask = (x_out == PAD_IDX)
    return {
        "x":         x_out,
        "ft_labels": ft_out,
        "bg_labels": bg_out,
        "pad_mask":  pad_mask,
    }


def make_format_dataloader(
    groups:    list[FormatGroup],
    n_msgs:    int  = 64,
    max_len:   int  = 512,
    shuffle:   bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader that yields one format-batch at a time (B=1)."""
    ds = FormatBatchDataset(groups)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda gs: collate_format_batch(gs, n_msgs=n_msgs, max_len=max_len),
    )


# ── LM Dataset (for ByteLM pretraining) ─────────────────────────────────────

class ByteSequenceDataset(Dataset):
    """
    Flat dataset of byte sequences for causal LM pretraining.
    Returns a single (L,) long tensor (raw bytes, no labels).
    """

    def __init__(
        self,
        groups:  list[FormatGroup],
        max_len: int = 256,
    ) -> None:
        self.max_len = max_len
        self.seqs: list[torch.Tensor] = []
        for grp in groups:
            for msg in grp.messages:
                raw = bytes.fromhex(msg.bytes_hex)
                n   = min(len(raw), max_len)
                t   = torch.tensor(list(raw[:n]), dtype=torch.long)
                self.seqs.append(t)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.seqs[idx]


def lm_collate(seqs: list[torch.Tensor]) -> torch.Tensor:
    """Pad byte sequences to the batch's max length."""
    L = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), L), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def make_lm_dataloader(
    groups:     list[FormatGroup],
    batch_size: int  = 128,
    max_len:    int  = 256,
    shuffle:    bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = ByteSequenceDataset(groups, max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lm_collate,
    )
