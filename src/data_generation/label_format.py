"""Unified field-type taxonomy and per-message ground-truth format."""

from __future__ import annotations

FIELD_TYPES: dict[int, str] = {
    0:  "UNKNOWN",
    1:  "LENGTH",
    2:  "TYPE_TAG",
    3:  "QUANTITY",
    4:  "TIMESTAMP",
    5:  "ADDRESS",
    6:  "PORT",
    7:  "FLAGS",
    8:  "CHECKSUM",
    9:  "COUNTER",
    10: "ASCII",
    11: "ENUM",
    12: "FLOAT",
    13: "INTEGER",
    14: "OPAQUE",
    15: "PADDING",
    16: "RESERVED",
}

FIELD_TYPE_IDS: dict[str, int] = {v: k for k, v in FIELD_TYPES.items()}


def validate_message(msg: dict) -> None:
    """Assert structural invariants on a ground-truth message dict."""
    assert "bytes_hex" in msg, "missing bytes_hex"
    assert "field_type_per_byte" in msg, "missing field_type_per_byte"
    assert "boundary_per_gap" in msg, "missing boundary_per_gap"
    assert "endianness" in msg, "missing endianness"
    assert "format_id" in msg, "missing format_id"

    n_bytes = len(msg["bytes_hex"]) // 2
    assert len(msg["bytes_hex"]) % 2 == 0, "bytes_hex length not even"
    assert len(msg["field_type_per_byte"]) == n_bytes, (
        f"field_type_per_byte len {len(msg['field_type_per_byte'])} != n_bytes {n_bytes}"
    )
    assert len(msg["boundary_per_gap"]) == max(n_bytes - 1, 0), (
        f"boundary_per_gap len {len(msg['boundary_per_gap'])} != n_bytes-1 {n_bytes - 1}"
    )

    for ft in msg["field_type_per_byte"]:
        assert ft in FIELD_TYPES, f"unknown field type id {ft}"
    for b in msg["boundary_per_gap"]:
        assert b in (0, 1), f"boundary value must be 0 or 1, got {b}"

    assert msg["endianness"] in ("big", "little"), (
        f"endianness must be 'big' or 'little', got {msg['endianness']!r}"
    )
