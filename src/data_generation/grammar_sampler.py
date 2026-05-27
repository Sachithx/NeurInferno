"""
Grammar-sampled synthetic binary protocol generator.

Grammar (from BinaryInferno §VII):
  PATTERN  := L(V)^[L] | T L(V)^[L]
  T,L,Q    := BYTE^N  where N in {1,2,3,4}
  V        := BYTE
  VLFIELD  := PATTERN
             | Q(PATTERN)^[Q]
             | Q(V V...V)^[Q]  with 2-8 V's
  FIELD    := VLFIELD | BYTE^P  with P >= 1
  FORMAT   := FIELD | FIELD FORMAT
  STAR-PAT := PATTERN* BYTE^S  where S >= 0
  STAR-FMT := FORMAT | FORMAT STAR-PAT | STAR-PAT

Segment kinds and their format spec dicts:
  fixed       {"kind":"fixed",   "width":int, "personality":str}
  lv          {"kind":"lv",      "l_width":int}
  tlv         {"kind":"tlv",     "t_width":int, "l_width":int}
  qlv         {"kind":"qlv",     "q_width":int, "l_width":int}
  qtlv        {"kind":"qtlv",    "q_width":int, "t_width":int, "l_width":int}
  qvbytes     {"kind":"qvbytes", "q_width":int, "v_count":int}
  star_lv     {"kind":"star_lv", "l_width":int,               "suffix":int}
  star_tlv    {"kind":"star_tlv","t_width":int, "l_width":int, "suffix":int}
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from src.data_generation.label_format import FIELD_TYPE_IDS, validate_message

PERSONALITIES = ["constant", "counter", "float", "ascii", "integer", "address"]

_PERSONALITY_TO_FTYPE = {
    "constant": FIELD_TYPE_IDS["ENUM"],
    "counter":  FIELD_TYPE_IDS["COUNTER"],
    "float":    FIELD_TYPE_IDS["FLOAT"],
    "ascii":    FIELD_TYPE_IDS["ASCII"],
    "integer":  FIELD_TYPE_IDS["INTEGER"],
    "address":  FIELD_TYPE_IDS["ADDRESS"],
}

L = FIELD_TYPE_IDS["LENGTH"]
T = FIELD_TYPE_IDS["TYPE_TAG"]
Q = FIELD_TYPE_IDS["QUANTITY"]
V = FIELD_TYPE_IDS["OPAQUE"]


# ---------------------------------------------------------------------------
# Byte-value samplers for fixed-width field personalities
# ---------------------------------------------------------------------------

def _sample_fixed_bytes(width: int, personality: str, rng: np.random.Generator,
                        counter_state: dict) -> bytes:
    if personality == "constant":
        val = int(rng.integers(0, 256))
        return bytes([val] * width)

    if personality == "counter":
        n = counter_state.get("n", 0)
        counter_state["n"] = n + 1
        val = n & ((1 << (width * 8)) - 1)
        return val.to_bytes(width, "big")

    if personality == "float":
        if width == 4:
            f = float(rng.standard_normal() * 100)
            return struct.pack(">f", f)
        if width == 8:
            f = float(rng.standard_normal() * 100)
            return struct.pack(">d", f)
        # For other widths fall through to random bytes
        return bytes(rng.integers(0, 256, size=width, dtype=np.uint8).tolist())

    if personality == "ascii":
        chars = [int(rng.integers(0x20, 0x7F)) for _ in range(width)]
        return bytes(chars)

    if personality == "address":
        return bytes([int(rng.integers(0, 256)) for _ in range(width)])

    # integer (default)
    return bytes(rng.integers(0, 256, size=width, dtype=np.uint8).tolist())


# ---------------------------------------------------------------------------
# Format sampling
# ---------------------------------------------------------------------------

def _sample_width(max_w: int, rng: np.random.Generator) -> int:
    """Sample a field width N in {1,2,3,4} up to max_w."""
    return int(rng.choice(range(1, min(max_w, 4) + 1)))


def _sample_field(max_w: int, rng: np.random.Generator) -> dict:
    """Sample a single FIELD (VLFIELD or fixed BYTE^P)."""
    kind = rng.choice(["fixed", "lv", "tlv", "qlv", "qtlv", "qvbytes"],
                      p=[0.35, 0.20, 0.15, 0.10, 0.08, 0.12])
    if kind == "fixed":
        width = int(rng.integers(1, 9))  # 1-8 bytes
        personality = rng.choice(PERSONALITIES)
        return {"kind": "fixed", "width": int(width), "personality": str(personality)}
    if kind == "lv":
        return {"kind": "lv", "l_width": _sample_width(max_w, rng)}
    if kind == "tlv":
        return {"kind": "tlv", "t_width": _sample_width(max_w, rng),
                "l_width": _sample_width(max_w, rng)}
    if kind == "qlv":
        return {"kind": "qlv", "q_width": _sample_width(max_w, rng),
                "l_width": _sample_width(max_w, rng)}
    if kind == "qtlv":
        return {"kind": "qtlv", "q_width": _sample_width(max_w, rng),
                "t_width": _sample_width(max_w, rng),
                "l_width": _sample_width(max_w, rng)}
    # qvbytes
    return {"kind": "qvbytes", "q_width": _sample_width(max_w, rng),
            "v_count": int(rng.integers(2, 9))}


def _sample_star_pat(max_w: int, rng: np.random.Generator) -> dict:
    """Sample a STAR-PAT = PATTERN* BYTE^S."""
    use_tag = rng.random() < 0.4
    suffix = int(rng.integers(0, 5))
    if use_tag:
        return {"kind": "star_tlv", "t_width": _sample_width(max_w, rng),
                "l_width": _sample_width(max_w, rng), "suffix": suffix}
    return {"kind": "star_lv", "l_width": _sample_width(max_w, rng), "suffix": suffix}


def sample_format(max_fields: int = 5, max_pattern_byte_width: int = 4,
                  rng: np.random.Generator | None = None) -> dict:
    """Sample a FORMAT or STAR-FMT from the grammar."""
    if rng is None:
        rng = np.random.default_rng()

    fmt_type = rng.choice(["format", "format_star", "star_only"],
                          p=[0.55, 0.30, 0.15])
    segments: list[dict] = []

    if fmt_type in ("format", "format_star"):
        n_fields = int(rng.integers(1, max_fields + 1))
        for _ in range(n_fields):
            segments.append(_sample_field(max_pattern_byte_width, rng))

    if fmt_type in ("format_star", "star_only"):
        segments.append(_sample_star_pat(max_pattern_byte_width, rng))

    return {"type": fmt_type, "segments": segments}


# ---------------------------------------------------------------------------
# Message instantiation
# ---------------------------------------------------------------------------

def _encode_uint(val: int, width: int, endian: str) -> bytes:
    return val.to_bytes(width, endian)


def _decode_uint(data: bytes, endian: str) -> int:
    return int.from_bytes(data, endian)


def _instantiate_segment(seg: dict, endian: str, rng: np.random.Generator,
                         counter_state: dict, max_len_value: int = 30,
                         max_q_value: int = 8, max_star_reps: int = 6
                         ) -> tuple[bytes, list[int], list[int]]:
    """
    Instantiate one segment.

    Returns (raw_bytes, field_types_per_byte, boundaries_at_ends_of_fields).
    boundaries_at_ends_of_fields: list of indices (into the returned bytes)
    where a field boundary occurs AFTER that byte index.
    We convert these to boundary_per_gap in the caller.
    """
    kind = seg["kind"]

    if kind == "fixed":
        w = seg["width"]
        b = _sample_fixed_bytes(w, seg["personality"], rng, counter_state)
        ft = _PERSONALITY_TO_FTYPE[seg["personality"]]
        return b, [ft] * w, [w - 1]  # boundary after last byte

    if kind == "lv":
        lw = seg["l_width"]
        length_val = int(rng.integers(1, max_len_value + 1))
        length_bytes = _encode_uint(length_val, lw, endian)
        value_bytes = bytes(rng.integers(0, 256, size=length_val, dtype=np.uint8).tolist())
        raw = length_bytes + value_bytes
        ft = [L] * lw + [V] * length_val
        # Two fields: L field ends at lw-1, V field ends at lw+length_val-1
        bounds = [lw - 1, lw + length_val - 1]
        return raw, ft, bounds

    if kind == "tlv":
        tw, lw = seg["t_width"], seg["l_width"]
        tag_bytes = _encode_uint(int(rng.integers(0, 16)), tw, endian)
        length_val = int(rng.integers(1, max_len_value + 1))
        length_bytes = _encode_uint(length_val, lw, endian)
        value_bytes = bytes(rng.integers(0, 256, size=length_val, dtype=np.uint8).tolist())
        raw = tag_bytes + length_bytes + value_bytes
        ft = [T] * tw + [L] * lw + [V] * length_val
        bounds = [tw - 1, tw + lw - 1, tw + lw + length_val - 1]
        return raw, ft, bounds

    if kind == "qlv":
        qw, lw = seg["q_width"], seg["l_width"]
        q_val = int(rng.integers(1, max_q_value + 1))
        q_bytes = _encode_uint(q_val, qw, endian)
        raw = q_bytes
        ft: list[int] = [Q] * qw
        # boundary after Q field
        bounds: list[int] = [qw - 1]
        offset = qw
        for _ in range(q_val):
            length_val = int(rng.integers(1, max_len_value + 1))
            length_bytes = _encode_uint(length_val, lw, endian)
            value_bytes = bytes(rng.integers(0, 256, size=length_val, dtype=np.uint8).tolist())
            raw += length_bytes + value_bytes
            ft += [L] * lw + [V] * length_val
            bounds += [offset + lw - 1, offset + lw + length_val - 1]
            offset += lw + length_val
        return raw, ft, bounds

    if kind == "qtlv":
        qw, tw, lw = seg["q_width"], seg["t_width"], seg["l_width"]
        q_val = int(rng.integers(1, max_q_value + 1))
        q_bytes = _encode_uint(q_val, qw, endian)
        raw = q_bytes
        ft = [Q] * qw
        bounds = [qw - 1]
        offset = qw
        for _ in range(q_val):
            tag_bytes = _encode_uint(int(rng.integers(0, 16)), tw, endian)
            length_val = int(rng.integers(1, max_len_value + 1))
            length_bytes = _encode_uint(length_val, lw, endian)
            value_bytes = bytes(rng.integers(0, 256, size=length_val, dtype=np.uint8).tolist())
            raw += tag_bytes + length_bytes + value_bytes
            ft += [T] * tw + [L] * lw + [V] * length_val
            bounds += [offset + tw - 1, offset + tw + lw - 1,
                       offset + tw + lw + length_val - 1]
            offset += tw + lw + length_val
        return raw, ft, bounds

    if kind == "qvbytes":
        qw, vc = seg["q_width"], seg["v_count"]
        q_val = int(rng.integers(1, max_q_value + 1))
        q_bytes = _encode_uint(q_val, qw, endian)
        raw = q_bytes
        ft = [Q] * qw
        bounds = [qw - 1]
        offset = qw
        for _ in range(q_val):
            vbs = bytes(rng.integers(0, 256, size=vc, dtype=np.uint8).tolist())
            raw += vbs
            ft += [V] * vc
            bounds.append(offset + vc - 1)
            offset += vc
        return raw, ft, bounds

    if kind == "star_lv":
        lw, suffix = seg["l_width"], seg["suffix"]
        n_reps = int(rng.integers(0, max_star_reps + 1))
        raw = b""
        ft = []
        bounds = []
        offset = 0
        for _ in range(n_reps):
            length_val = int(rng.integers(1, max_len_value + 1))
            length_bytes = _encode_uint(length_val, lw, endian)
            value_bytes = bytes(rng.integers(0, 256, size=length_val, dtype=np.uint8).tolist())
            raw += length_bytes + value_bytes
            ft += [L] * lw + [V] * length_val
            bounds += [offset + lw - 1, offset + lw + length_val - 1]
            offset += lw + length_val
        if suffix > 0:
            suf_bytes = bytes(rng.integers(0, 256, size=suffix, dtype=np.uint8).tolist())
            raw += suf_bytes
            ft += [V] * suffix
            bounds.append(offset + suffix - 1)
        elif n_reps == 0:
            # empty star with no suffix — return 1 padding byte to keep msg non-empty
            raw = bytes([0])
            ft = [FIELD_TYPE_IDS["PADDING"]]
            bounds = [0]
        return raw, ft, bounds

    if kind == "star_tlv":
        tw, lw, suffix = seg["t_width"], seg["l_width"], seg["suffix"]
        n_reps = int(rng.integers(0, max_star_reps + 1))
        raw = b""
        ft = []
        bounds = []
        offset = 0
        for _ in range(n_reps):
            tag_bytes = _encode_uint(int(rng.integers(0, 16)), tw, endian)
            length_val = int(rng.integers(1, max_len_value + 1))
            length_bytes = _encode_uint(length_val, lw, endian)
            value_bytes = bytes(rng.integers(0, 256, size=length_val, dtype=np.uint8).tolist())
            raw += tag_bytes + length_bytes + value_bytes
            ft += [T] * tw + [L] * lw + [V] * length_val
            bounds += [offset + tw - 1, offset + tw + lw - 1,
                       offset + tw + lw + length_val - 1]
            offset += tw + lw + length_val
        if suffix > 0:
            suf_bytes = bytes(rng.integers(0, 256, size=suffix, dtype=np.uint8).tolist())
            raw += suf_bytes
            ft += [V] * suffix
            bounds.append(offset + suffix - 1)
        elif n_reps == 0:
            raw = bytes([0])
            ft = [FIELD_TYPE_IDS["PADDING"]]
            bounds = [0]
        return raw, ft, bounds

    raise ValueError(f"Unknown segment kind: {kind!r}")


def instantiate_message(format_spec: dict, rng: np.random.Generator,
                        endian: str | None = None) -> tuple[bytes, list[int], list[int]]:
    """
    Generate one valid message from a format spec.

    Returns (raw_bytes, field_type_per_byte, boundary_per_gap).
    """
    if endian is None:
        endian = rng.choice(["big", "little"])

    counter_state: dict = {}
    all_bytes = b""
    all_ft: list[int] = []
    all_bound_ends: list[int] = []  # absolute byte indices where fields end
    offset = 0

    for seg in format_spec["segments"]:
        raw, ft, bounds = _instantiate_segment(seg, endian, rng, counter_state)
        all_bytes += raw
        all_ft += ft
        # bounds are relative to this segment; make absolute
        all_bound_ends += [offset + b for b in bounds]
        offset += len(raw)

    # Convert "field ends at index i" → boundary_per_gap[i] = 1 for i < n-1
    n = len(all_bytes)
    boundary_per_gap = [0] * max(n - 1, 0)
    for end_idx in all_bound_ends:
        gap_idx = end_idx  # gap between byte[end_idx] and byte[end_idx+1]
        if 0 <= gap_idx < n - 1:
            boundary_per_gap[gap_idx] = 1

    return all_bytes, all_ft, boundary_per_gap


def emit_labels(format_spec: dict, message_bytes: bytes,
                field_type_per_byte: list[int],
                boundary_per_gap: list[int],
                endian: str, format_id: str) -> dict:
    """Wrap instantiation results into the ground-truth JSON format."""
    msg = {
        "bytes_hex": message_bytes.hex(),
        "field_type_per_byte": field_type_per_byte,
        "boundary_per_gap": boundary_per_gap,
        "endianness": endian,
        "format_id": format_id,
    }
    validate_message(msg)
    return msg


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------

def generate_tier1(
    output_dir: str | Path,
    n_formats: int = 500,
    n_messages_per_format: int = 200,
    max_fields: int = 5,
    max_pattern_byte_width: int = 4,
    seed: int = 42,
) -> None:
    """Generate tier1_grammar data: n_formats × n_messages_per_format messages."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for fmt_idx in range(n_formats):
        fmt_id = f"tier1_format_{fmt_idx:04d}"
        fmt_dir = output_dir / fmt_id
        fmt_dir.mkdir(exist_ok=True)

        fmt_spec = sample_format(max_fields, max_pattern_byte_width, rng)

        # Save the format spec for reproducibility
        with open(fmt_dir / "format_spec.json", "w") as f:
            json.dump(fmt_spec, f)

        out_path = fmt_dir / "messages.jsonl"
        with open(out_path, "w") as f:
            for msg_idx in range(n_messages_per_format):
                # Alternate big/little endian in halves
                endian = "big" if msg_idx < n_messages_per_format // 2 else "little"
                try:
                    raw, ft, bg = instantiate_message(fmt_spec, rng, endian=endian)
                except Exception as e:
                    # Skip degenerate messages (very rare)
                    continue

                if len(raw) == 0:
                    continue

                msg_dict = emit_labels(fmt_spec, raw, ft, bg, endian, fmt_id)
                f.write(json.dumps(msg_dict) + "\n")

        if (fmt_idx + 1) % 50 == 0:
            print(f"  Generated {fmt_idx + 1}/{n_formats} formats")

    print(f"Done: {n_formats} formats → {output_dir}")
