"""
Ground-truth loading for LOPO evaluation.

We use Tier-2 Scapy-extracted messages as the test set because they carry
per-byte and per-gap labels.  The BI benchmark `.txt.input` files have no
explicit labels, so we do NOT use them for metric computation.  See NOTES.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LabeledMessage:
    bytes_hex:           str
    boundary_per_gap:    list[int]    # binary, len = n_bytes - 1
    field_type_per_byte: list[int]
    format_id:           str


def load_labeled_messages(
    tier2_dir: Path,
    protocol:  str,
    max_msgs:  int | None = 1000,
    seed:      int = 0,
    exclude_corrupted: bool = True,
    split: str = "test",
    test_fraction: float = 0.20,
) -> list[LabeledMessage]:
    """
    Load messages for one Tier-2 protocol.

    split="test"  — last test_fraction of lines (never seen during LOPO training)
    split="all"   — every line (use only for analysis, not for leakage-free eval)

    The split boundary matches the one used in dataset.load_tier2_groups so
    that the val messages used during LOPO training and the test messages used
    here are strictly disjoint.
    """
    path = Path(tier2_dir) / protocol / "messages.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No Tier-2 data for protocol '{protocol}': {path}")

    # Read all raw lines first so the split boundary is by total line index,
    # matching the cut in dataset.load_tier2_groups (which counts all lines).
    raw_lines: list[dict] = []
    with open(path) as f:
        for line in f:
            raw_lines.append(json.loads(line))

    if split == "test":
        n_val = max(1, int(len(raw_lines) * (1.0 - test_fraction)))
        raw_lines = raw_lines[n_val:]          # last 20% by raw index
    # split=="all" uses all raw_lines

    all_records: list[LabeledMessage] = []
    for d in raw_lines:
        if exclude_corrupted and d.get("format_id", "").endswith("_corrupted"):
            continue
        all_records.append(LabeledMessage(
            bytes_hex=d["bytes_hex"],
            boundary_per_gap=d["boundary_per_gap"],
            field_type_per_byte=d["field_type_per_byte"],
            format_id=d["format_id"],
        ))

    records = all_records

    if max_msgs is not None and len(records) > max_msgs:
        rng = random.Random(seed)
        records = rng.sample(records, max_msgs)

    return records


def messages_to_hex_lines(messages: list[LabeledMessage]) -> list[str]:
    """Return uppercase hex strings for BI stdin."""
    return [m.bytes_hex.upper() for m in messages]
