"""Helper for loading BinaryInferno hex-per-line dataset files."""

from __future__ import annotations

import pathlib
from typing import Optional

_BI_DATASET_ROOT = pathlib.Path(__file__).parents[2] / "data" / "tier3_benchmark"
_TOP_LEVEL_COUNTS = (100, 500, 1000)


def _dataset_path(protocol: str, n_messages: int) -> pathlib.Path:
    count = min(_TOP_LEVEL_COUNTS, key=lambda c: abs(c - n_messages))
    return _BI_DATASET_ROOT / "top-level" / str(count) / f"{protocol}.txt.input"


def load_bi_dataset(protocol: str, n_messages: int = 100) -> list[bytes]:
    """Read BI hex-per-line file and return up to n_messages byte arrays."""
    path = _dataset_path(protocol, n_messages)
    if not path.exists():
        raise FileNotFoundError(f"BI dataset not found: {path}")

    messages: list[bytes] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(bytes.fromhex(line))
            except ValueError:
                continue
            if len(messages) >= n_messages:
                break
    return messages


if __name__ == "__main__":
    msgs = load_bi_dataset("bgp", 100)
    print(f"Loaded {len(msgs)} BGP messages")
    for i, m in enumerate(msgs[:3]):
        print(f"  [{i}] len={len(m):3d}  hex={m.hex()[:40]}...")
