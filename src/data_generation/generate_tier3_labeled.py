"""
Generate labeled training data from BI benchmark (tier3) messages.

Uses eval_harness.py deterministic parsers to produce the same JSONL format
as Tier 1 and Tier 2:
  bytes_hex, field_type_per_byte, boundary_per_gap, endianness, format_id

Boundary labels: derived from ground_truth_for_sample() — i.e. after
sample-level constant-field merging, matching evaluation methodology exactly
so training targets and evaluation ground truth are consistent.

Field-type labels: mapped from parser field names via a keyword lookup table.

Output: data/tier3_labeled/{protocol}/messages.jsonl

Usage:
    python -m src.data_generation.generate_tier3_labeled \\
        --bench_dir  data/tier3_benchmark/top-level/1000 \\
        --output_dir data/tier3_labeled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.eval_harness import (
    PARSERS, PROTOCOL_ENDIAN, ground_truth_for_sample,
)
from src.data_generation.label_format import FIELD_TYPE_IDS, validate_message


# ── Field-name → taxonomy type ────────────────────────────────────────────────
# Exact-match first, then substring fallback.

_EXACT: dict[str, str] = {
    # LENGTH
    "length": "LENGTH", "payload_len": "LENGTH", "tcp_length": "LENGTH",
    "byte_count": "LENGTH", "llen": "LENGTH",
    # TIMESTAMP
    "ref_timestamp": "TIMESTAMP", "origin_timestamp": "TIMESTAMP",
    "receive_timestamp": "TIMESTAMP", "transmit_timestamp": "TIMESTAMP",
    # ADDRESS
    "ciaddr": "ADDRESS", "yiaddr": "ADDRESS", "siaddr": "ADDRESS",
    "giaddr": "ADDRESS", "chaddr": "ADDRESS", "ref_clock_id": "ADDRESS",
    "src": "ADDRESS", "dst": "ADDRESS",
    # CHECKSUM
    "header_crc": "CHECKSUM", "signature": "CHECKSUM",
    # COUNTER
    "xid": "COUNTER", "txid": "COUNTER", "transaction_id": "COUNTER",
    "session_id": "COUNTER", "message_id": "COUNTER",
    "mid": "COUNTER", "uid": "COUNTER", "pid": "COUNTER",
    "sequence": "COUNTER", "next_command": "COUNTER",
    # FLAGS
    "flags": "FLAGS", "flags2": "FLAGS", "li_vn_mode": "FLAGS",
    "control": "FLAGS", "status": "FLAGS",
    "credits": "FLAGS", "credit_charge": "FLAGS",
    # TYPE_TAG
    "type": "TYPE_TAG", "command": "TYPE_TAG", "stx": "TYPE_TAG",
    "function_code": "TYPE_TAG", "op": "TYPE_TAG",
    # QUANTITY
    "qdcount": "QUANTITY", "q1_count": "QUANTITY", "q2_count": "QUANTITY",
    "word_count": "QUANTITY", "hops": "QUANTITY",
    "hlen": "QUANTITY", "htype": "QUANTITY",
    # INTEGER
    "stratum": "INTEGER", "poll": "INTEGER", "precision": "INTEGER",
    "root_delay": "INTEGER", "root_dispersion": "INTEGER",
    "secs": "INTEGER", "tid": "INTEGER", "tree_id": "INTEGER",
    "pid_high": "INTEGER", "system_id": "INTEGER",
    "component_id": "INTEGER", "unit_id": "INTEGER",
    # RESERVED / PADDING
    "marker": "RESERVED", "magic": "RESERVED", "protocol": "RESERVED",
    "protocol_id": "RESERVED", "structure_size": "RESERVED",
    "start": "RESERVED", "reserved": "RESERVED",
    "null_terminator": "PADDING",
    # ASCII / OPAQUE
    "sname": "ASCII", "file": "OPAQUE", "payload": "OPAQUE",
    "data": "OPAQUE", "body": "OPAQUE",
    "security_features": "OPAQUE",
}

_SUBSTR: list[tuple[str, str]] = [
    # checked in order; first match wins
    ("timestamp",    "TIMESTAMP"),
    ("addr",         "ADDRESS"),
    ("crc",          "CHECKSUM"),
    ("checksum",     "CHECKSUM"),
    ("signature",    "CHECKSUM"),
    ("_length",      "LENGTH"),
    ("_len",         "LENGTH"),
    ("length",       "LENGTH"),
    ("_code",        "ENUM"),
    ("_type",        "TYPE_TAG"),
    ("flag",         "FLAGS"),
    ("count",        "QUANTITY"),
    ("_id",          "COUNTER"),
    ("data",         "OPAQUE"),
    ("payload",      "OPAQUE"),
    ("body",         "OPAQUE"),
    ("value",        "OPAQUE"),
    ("block",        "OPAQUE"),
    ("trunc",        "OPAQUE"),
    ("pad",          "PADDING"),
    ("reserved",     "RESERVED"),
]


def _field_name_to_type(name: str) -> int:
    n = name.lower()
    if n in _EXACT:
        return FIELD_TYPE_IDS[_EXACT[n]]
    for substr, ttype in _SUBSTR:
        if substr in n:
            return FIELD_TYPE_IDS[ttype]
    return FIELD_TYPE_IDS["INTEGER"]


# ── Per-message label generation ──────────────────────────────────────────────

def _labels_for_message(
    msg:            bytes,
    boundary_set:   set[int],    # from ground_truth_for_sample (merged)
    parser,
) -> tuple[list[int], list[int]]:
    """
    Return (field_type_per_byte, boundary_per_gap) for one message.

    boundary_per_gap uses the pre-computed merged boundary set.
    field_type_per_byte is derived from the raw parser fields.
    """
    n = len(msg)

    # Field types from raw parser (independent of merging)
    ft = [FIELD_TYPE_IDS["UNKNOWN"]] * n
    for f in parser(msg):
        ftype = _field_name_to_type(f.name)
        for i in range(f.start, min(f.end, n)):
            ft[i] = ftype

    # Boundaries from merged ground truth
    bg = [1 if (i + 1) in boundary_set else 0 for i in range(n - 1)]

    return ft, bg


# ── Protocol generation ───────────────────────────────────────────────────────

def _generate_protocol(
    proto:      str,
    messages:   list[bytes],
    output_dir: Path,
) -> int:
    """Write messages.jsonl for one protocol. Returns number of records written."""
    parser  = PARSERS[proto]
    _endian_map = {"BE": "big", "LE": "little"}
    endian  = _endian_map.get(PROTOCOL_ENDIAN.get(proto, "BE"), "big")
    out_dir = output_dir / proto
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merged boundaries for ALL messages together (matches evaluation GT)
    gt_sets = ground_truth_for_sample(proto, messages)

    written = 0
    with open(out_dir / "messages.jsonl", "w") as f:
        for idx, (msg, boundary_set) in enumerate(zip(messages, gt_sets)):
            if len(msg) < 2:
                continue

            ft, bg = _labels_for_message(msg, boundary_set, parser)

            record = {
                "bytes_hex":           msg.hex(),
                "field_type_per_byte": ft,
                "boundary_per_gap":    bg,
                "endianness":          endian,
                "format_id":           f"tier3_{proto}_{idx:04d}",
            }
            try:
                validate_message(record)
            except AssertionError as e:
                print(f"    [{proto}] msg {idx} validation error: {e} — skipped")
                continue

            f.write(json.dumps(record) + "\n")
            written += 1

    return written


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_tier3_labeled(
    bench_dir:  str = "data/tier3_benchmark/top-level/1000",
    output_dir: str = "data/tier3_labeled",
    protocols:  list[str] | None = None,
) -> None:
    bench_dir  = Path(bench_dir)
    output_dir = Path(output_dir)

    target_protocols = protocols if protocols else list(PARSERS.keys())

    total = 0
    for proto in target_protocols:
        input_file = bench_dir / f"{proto}.txt.input"
        if not input_file.exists():
            print(f"  [{proto}] SKIP — {input_file} not found")
            continue

        # Load raw bytes
        messages: list[bytes] = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(bytes.fromhex(line))
                    except ValueError:
                        pass

        if not messages:
            print(f"  [{proto}] SKIP — no messages loaded")
            continue

        print(f"  [{proto}] {len(messages)} messages ...", end=" ", flush=True)
        n = _generate_protocol(proto, messages, output_dir)
        print(f"{n} written → {output_dir / proto / 'messages.jsonl'}")
        total += n

    print(f"\nDone. {total} total labeled messages across {len(target_protocols)} protocols.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir",   default="data/tier3_benchmark/top-level/1000")
    p.add_argument("--output_dir",  default="data/tier3_labeled")
    p.add_argument("--protocols",   nargs="*", default=None,
                   help="Subset of protocols to generate (default: all in PARSERS)")
    args = p.parse_args()
    generate_tier3_labeled(**vars(args))
