"""
Tier 2 generation driver.

For each protocol:
  1. Generate Scapy packets with varied field values
  2. Extract per-byte labels via scapy_extractor
  3. Apply controlled corruptions to 10% of messages
  4. Write to data/tier2_scapy/<protocol>/messages.jsonl
"""

from __future__ import annotations

import json
import pathlib
import random
import struct
import sys

from src.data_generation.label_format import FIELD_TYPE_IDS, validate_message
from src.data_generation.scapy_extractor import make_message_dict, extract_field_layout
from src.data_generation.protocol_generators import SCAPY_PROTOCOLS, gen_bgp_raw


_UNK = FIELD_TYPE_IDS["UNKNOWN"]


# ── Corruption helpers ────────────────────────────────────────────────────────

def _apply_corruption(msg_dict: dict, rng: random.Random) -> dict:
    """
    Apply one of two corruptions (50/50):
      A) Flip 1-3 random bytes → label them UNKNOWN
      B) Truncate a random suffix → suffix bytes labeled UNKNOWN
    Returns a NEW dict; original is unchanged.
    """
    raw = bytes.fromhex(msg_dict["bytes_hex"])
    ft = list(msg_dict["field_type_per_byte"])
    n = len(raw)

    if n < 4:
        return msg_dict  # too short to corrupt meaningfully

    corruption = rng.choice(["flip", "truncate"])

    raw = bytearray(raw)
    if corruption == "flip":
        n_flips = rng.randint(1, min(3, n))
        for _ in range(n_flips):
            idx = rng.randint(0, n - 1)
            raw[idx] ^= rng.randint(1, 255)
            ft[idx] = _UNK
    else:  # truncate tail
        keep = rng.randint(max(1, n // 2), n - 1)
        raw = raw[:keep]
        ft = ft[:keep]

    raw = bytes(raw)
    n_new = len(raw)
    bg_new = [0] * max(n_new - 1, 0)

    # Recompute boundary_per_gap: keep original gap labels where still valid
    orig_bg = msg_dict["boundary_per_gap"]
    for i in range(min(len(orig_bg), n_new - 1)):
        bg_new[i] = orig_bg[i]

    new_msg = {
        "bytes_hex": raw.hex(),
        "field_type_per_byte": ft,
        "boundary_per_gap": bg_new,
        "endianness": msg_dict["endianness"],
        "format_id": msg_dict["format_id"] + "_corrupted",
    }
    validate_message(new_msg)
    return new_msg


# ── BGP raw bytes labeler ─────────────────────────────────────────────────────

def _label_bgp_raw(raw: bytes, format_id: str) -> dict:
    """
    Hand-label a raw BGP message:
      - Bytes 0-15 : marker (FLAGS)
      - Bytes 16-17: length (LENGTH)
      - Byte  18   : type   (TYPE_TAG)
      - Bytes 19+  : OPEN payload (INTEGER for version/asn/hold_time, ADDRESS for bgp_id)
    """
    n = len(raw)
    ft = [FIELD_TYPE_IDS["UNKNOWN"]] * n
    bg = [0] * max(n - 1, 0)

    # Marker
    for i in range(min(16, n)):
        ft[i] = FIELD_TYPE_IDS["FLAGS"]
    if 15 < n - 1:
        bg[15] = 1

    # Length
    for i in range(16, min(18, n)):
        ft[i] = FIELD_TYPE_IDS["LENGTH"]
    if 17 < n - 1:
        bg[17] = 1

    # Type
    if 18 < n:
        ft[18] = FIELD_TYPE_IDS["TYPE_TAG"]
    if 18 < n - 1:
        bg[18] = 1

    # OPEN payload if present
    if n > 19:
        off = 19
        # version (1B)
        ft[off] = FIELD_TYPE_IDS["INTEGER"]
        if off < n - 1:
            bg[off] = 1
        off += 1
        # asn (2B)
        for i in range(off, min(off + 2, n)):
            ft[i] = FIELD_TYPE_IDS["COUNTER"]
        if off + 1 < n - 1:
            bg[off + 1] = 1
        off += 2
        # hold_time (2B)
        for i in range(off, min(off + 2, n)):
            ft[i] = FIELD_TYPE_IDS["INTEGER"]
        if off + 1 < n - 1:
            bg[off + 1] = 1
        off += 2
        # bgp_id (4B)
        for i in range(off, min(off + 4, n)):
            ft[i] = FIELD_TYPE_IDS["ADDRESS"]
        if off + 3 < n - 1:
            bg[off + 3] = 1
        off += 4
        # opt_len (1B)
        if off < n:
            ft[off] = FIELD_TYPE_IDS["LENGTH"]
        off += 1

    msg = {
        "bytes_hex": raw.hex(),
        "field_type_per_byte": ft,
        "boundary_per_gap": bg,
        "endianness": "big",
        "format_id": format_id,
    }
    validate_message(msg)
    return msg


# ── Main generation driver ─────────────────────────────────────────────────────

def generate_tier2(
    output_dir: str | pathlib.Path,
    protocols: list[str] | None = None,
    seed: int = 42,
) -> None:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    proto_list = protocols or list(SCAPY_PROTOCOLS.keys())

    for proto_name in proto_list:
        gen_fn, n_msgs, has_scapy_pkt = SCAPY_PROTOCOLS[proto_name]
        print(f"\n[{proto_name}] generating {n_msgs} messages …")

        proto_dir = output_dir / proto_name
        proto_dir.mkdir(exist_ok=True)
        out_path = proto_dir / "messages.jsonl"

        random.seed(seed + hash(proto_name) % 10000)

        pkts = gen_fn(n_msgs)

        written = 0
        skipped = 0

        with open(out_path, "w") as f:
            for i, pkt in enumerate(pkts):
                fmt_id = f"tier2_{proto_name}_{i:05d}"
                try:
                    if has_scapy_pkt:
                        msg = make_message_dict(pkt, fmt_id, endian="big")
                    else:
                        # raw bytes (bgp_raw)
                        msg = _label_bgp_raw(pkt, fmt_id)
                except Exception as e:
                    skipped += 1
                    continue

                # Apply corruption to 10% of messages
                if rng.random() < 0.10:
                    try:
                        msg = _apply_corruption(msg, rng)
                    except Exception:
                        pass  # keep original on failure

                f.write(json.dumps(msg) + "\n")
                written += 1

        print(f"  → {written} written, {skipped} skipped  [{out_path}]")


if __name__ == "__main__":
    generate_tier2("data/tier2_scapy")
