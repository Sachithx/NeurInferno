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


# ── Coarse 8-class taxonomy for cross-protocol transfer ──────────────────────
#
# Merges fine classes with similar protocol-agnostic byte-level signatures.
# UNKNOWN (0) is the ignore-index — kept but never predicted.
# RESERVED → FLAGS (per brief): reserved bits share the static/zero-valued
#   statistical signature of flags fields, unlike OPAQUE blobs.
# DHCP note: GT labels whole option-value blobs as OPAQUE even when the model
#   sees internal TLV structure (type + length + data sub-fields). This is a
#   GT-convention mismatch, not a model error. Expect DHCP OPAQUE F1 ≈ 0
#   in fine-grained 17-class eval; the coarse merge absorbs most of the hit.

COARSE_TYPES: dict[int, str] = {
    0: "UNKNOWN",    # ignore-index, excluded from all metrics
    1: "LENGTH",
    2: "TYPE_TAG",
    3: "ADDRESS",
    4: "PORT",
    5: "FLAGS",
    6: "CHECKSUM",
    7: "NUMERIC",
    8: "OPAQUE",
}
COARSE_TYPE_IDS: dict[str, int] = {v: k for k, v in COARSE_TYPES.items()}
N_COARSE_TYPES  = len(COARSE_TYPES)         # 9 including UNKNOWN
N_COARSE_PRED   = N_COARSE_TYPES - 1        # 8 predicted classes (excl. UNKNOWN)
COARSE_LABELS   = [COARSE_TYPES[i] for i in range(1, N_COARSE_TYPES)]  # ordered

# Fine-to-coarse mapping tensor index: FINE_TO_COARSE[fine_id] = coarse_id
# Build as a plain dict first; a torch LongTensor version is built lazily by consumers.
FINE_TO_COARSE: dict[int, int] = {
    0:  0,  # UNKNOWN     → UNKNOWN     (ignore-index)
    1:  COARSE_TYPE_IDS["LENGTH"],
    2:  COARSE_TYPE_IDS["TYPE_TAG"],
    3:  COARSE_TYPE_IDS["NUMERIC"],   # QUANTITY
    4:  COARSE_TYPE_IDS["NUMERIC"],   # TIMESTAMP  (singleton in NTP/ICMP)
    5:  COARSE_TYPE_IDS["ADDRESS"],
    6:  COARSE_TYPE_IDS["PORT"],
    7:  COARSE_TYPE_IDS["FLAGS"],
    8:  COARSE_TYPE_IDS["CHECKSUM"],
    9:  COARSE_TYPE_IDS["NUMERIC"],   # COUNTER
    10: COARSE_TYPE_IDS["OPAQUE"],    # ASCII      (singleton in DHCP)
    11: COARSE_TYPE_IDS["FLAGS"],     # ENUM
    12: COARSE_TYPE_IDS["NUMERIC"],   # FLOAT      (singleton in NTP)
    13: COARSE_TYPE_IDS["NUMERIC"],   # INTEGER    (catch-all)
    14: COARSE_TYPE_IDS["OPAQUE"],    # OPAQUE
    15: COARSE_TYPE_IDS["OPAQUE"],    # PADDING
    16: COARSE_TYPE_IDS["FLAGS"],     # RESERVED   (static zeros → FLAGS-like)
}


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
