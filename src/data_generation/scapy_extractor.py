"""
Scapy-based field layout extractor for binary protocol messages.

Walks Scapy's fields_desc for each layer, computes per-byte field type
labels using bit-position tracking, and emits standard ground-truth dicts.
"""

from __future__ import annotations

import random
import struct
from typing import Optional

import scapy.fields as sf
from scapy.packet import Packet

from src.data_generation.label_format import FIELD_TYPE_IDS, validate_message

# ── Taxonomy type IDs (short aliases) ──────────────────────────────────────
_UNK  = FIELD_TYPE_IDS["UNKNOWN"]
_LEN  = FIELD_TYPE_IDS["LENGTH"]
_TYPE = FIELD_TYPE_IDS["TYPE_TAG"]
_QTY  = FIELD_TYPE_IDS["QUANTITY"]
_TS   = FIELD_TYPE_IDS["TIMESTAMP"]
_ADDR = FIELD_TYPE_IDS["ADDRESS"]
_PORT = FIELD_TYPE_IDS["PORT"]
_FLAG = FIELD_TYPE_IDS["FLAGS"]
_CSUM = FIELD_TYPE_IDS["CHECKSUM"]
_CTR  = FIELD_TYPE_IDS["COUNTER"]
_ASCII= FIELD_TYPE_IDS["ASCII"]
_ENUM = FIELD_TYPE_IDS["ENUM"]
_FLT  = FIELD_TYPE_IDS["FLOAT"]
_INT  = FIELD_TYPE_IDS["INTEGER"]
_OPAQ = FIELD_TYPE_IDS["OPAQUE"]
_PAD  = FIELD_TYPE_IDS["PADDING"]
_RESV = FIELD_TYPE_IDS["RESERVED"]


# ── Name-based heuristics (checked first) ──────────────────────────────────

_NAME_CONTAINS_TO_TYPE: list[tuple[list[str], int]] = [
    (["chksum", "checksum", "crc", "fcs"],           _CSUM),
    (["port", "sport", "dport"],                      _PORT),
    (["seq", "seqno", "seqnum", "sequence"],          _CTR),
    (["ack", "ackno", "ackseq"],                      _CTR),
    (["xid", "transid", "trans_id", "transaction"],   _CTR),
    (["ttl", "hop_limit", "hoplimit"],                _INT),
    (["proto", "protocol", "type", "opcode", "op",
      "msgtype", "msg_type", "funccode", "func_code",
      "code", "rcode", "qtype", "rrtype"],            _TYPE),
    (["flags", "flag", "control", "ctrl"],            _FLAG),
    (["len", "length", "size", "hlen", "dlen",
      "msglen", "data_len", "paylen"],                _LEN),
    (["pad", "padding", "align"],                     _PAD),
    (["reserved", "rsv", "mbz", "unused"],            _RESV),
    (["id", "ident", "identifier"],                   _CTR),
    (["timestamp", "time", "ts"],                     _TS),
    (["version", "ver"],                              _TYPE),
    (["window", "wscale"],                            _INT),
    (["stratum", "poll", "precision"],                _INT),
    (["offset", "ref", "orig", "recv", "sent"],       _TS),
    # ARP / link-layer address fields
    (["hwsrc", "hwdst", "psrc", "pdst",
      "gaddr", "addr"],                               _ADDR),
]


def _name_heuristic(name: str) -> Optional[int]:
    """Return a type ID from field name heuristics, or None."""
    low = name.lower()
    for keywords, tid in _NAME_CONTAINS_TO_TYPE:
        for kw in keywords:
            if kw in low:
                return tid
    return None


# ── Class-based taxonomy mapping ───────────────────────────────────────────

def _unwrap(f) -> object:
    """Strip Emph / ConditionalField wrappers."""
    while type(f).__name__ in ("Emph", "ConditionalField"):
        f = f.fld
    return f


def scapy_field_to_taxonomy_type(field, layer: Optional[Packet] = None) -> int:
    """Map a Scapy field to the unified taxonomy type ID."""
    f = _unwrap(field)
    cls = type(f).__name__

    # ASN1 fields — use name heuristic, fall back to OPAQUE
    if type(f).__module__.startswith("scapy.asn1"):
        name = getattr(f, "name", "").lower()
        nh = _name_heuristic(name)
        return nh if nh is not None else _OPAQ

    name = f.name.lower()

    # Exact-class matches
    if cls in ("IPField", "SourceIPField", "DestIPField",
               "IP6Field", "SourceIP6Field", "DestIP6Field",
               "MACField", "LEMACField", "OUIField"):
        return _ADDR

    if cls in ("IEEEFloatField", "IEEEDoubleField", "BCDFloatField",
               "FixedPointField", "ScalingField"):
        return _FLT

    if cls in ("UTCTimeField", "SecondsIntField", "TimeStampField"):
        return _TS

    if cls in ("LenField", "FieldLenField", "LEFieldLenField",
               "BitFieldLenField", "BitLenField"):
        return _LEN

    if cls in ("FlagsField", "MultiFlagsField", "XByteField",
               "XBitField", "XShortField", "XIntField", "XLongField",
               "XLE3BytesField", "XLEIntField", "XLELongField",
               "XLEShortField", "XNBytesField"):
        # Apply name heuristic first (covers chksum, port, seq, id, etc.)
        nh = _name_heuristic(name)
        if nh in (_CSUM, _PORT, _CTR, _LEN, _TYPE, _TS, _ADDR):
            return nh
        if cls == "FlagsField":
            return _FLAG
        return _FLAG  # XByteField etc. usually flags/control bytes

    if cls in ("ByteEnumField", "ShortEnumField", "IntEnumField",
               "LongEnumField", "BitEnumField", "LEShortEnumField",
               "LEIntEnumField", "LELongEnumField", "BitEnumField",
               "CharEnumField", "XByteEnumField", "XShortEnumField",
               "XLEIntEnumField", "XLEShortEnumField"):
        nh = _name_heuristic(name)
        if nh in (_CSUM, _PORT, _LEN, _CTR, _ADDR, _TYPE):
            return nh
        return _ENUM

    if cls in ("MultipleTypeField",):
        # ARP hwsrc/hwdst/psrc/pdst — use name heuristic, default ADDRESS
        nh = _name_heuristic(name)
        return nh if nh is not None else _ADDR

    if cls in ("StrField", "StrFixedLenField", "StrLenField",
               "StrNullField", "BoundStrLenField", "XStrField",
               "XStrFixedLenField", "XStrLenField", "StrEnumField",
               "StrFixedLenEnumField", "StrLenEnumField",
               "NetBIOSNameField"):
        return _ASCII

    if cls in ("PadField", "ReversePadField"):
        return _PAD

    if cls in ("UUIDField", "UUIDEnumField"):
        return _CTR

    if cls in ("PacketField", "PacketLenField", "PacketListField",
               "_DNSPacketListField"):
        return _OPAQ

    # Name heuristics (after class check)
    nh = _name_heuristic(name)
    if nh is not None:
        return nh

    # Fallthrough: generic numerics
    if cls in ("ByteField", "OByteField", "SignedByteField", "YesNoByteField"):
        return _INT
    if cls in ("ShortField", "LEShortField", "SignedShortField",
               "LESignedShortField"):
        return _INT
    if cls in ("IntField", "LEIntField", "SignedIntField",
               "LESignedIntField", "ThreeBytesField", "LEThreeBytesField"):
        return _INT
    if cls in ("LongField", "LELongField", "SignedLongField",
               "LESignedLongField"):
        return _INT
    if cls in ("NBytesField", "LEX3BytesField", "X3BytesField"):
        return _INT
    if cls in ("BitField", "_BitField"):
        # Small BitField → FLAGS, larger → INTEGER
        size = getattr(f, "size", 8)  # bit count
        return _FLAG if size <= 4 else _INT

    return _INT  # safe default


# ── Field bit-size helper ───────────────────────────────────────────────────

def _field_bit_size(field, layer: Packet) -> int:
    """Return the actual bit size of a field for the given layer instance."""
    if type(field).__name__ == "ConditionalField":
        try:
            if not field.cond(layer):
                return 0
        except Exception:
            pass
        return _field_bit_size(field.fld, layer)

    if type(field).__name__ == "Emph":
        return _field_bit_size(field.fld, layer)

    # ASN1 fields (SNMP etc.) — fall back to serialization-based sizing
    if type(field).__module__.startswith("scapy.asn1"):
        try:
            val = layer.getfieldval(field.name)
            buf = field.addfield(layer, b"", val)
            return len(buf) * 8
        except Exception:
            return 0

    try:
        val = layer.getfieldval(field.name)
        sz = field.i2len(layer, val)
        if isinstance(sz, float):
            return max(int(round(sz * 8)), 0)
        return max(int(sz) * 8, 0)
    except Exception:
        try:
            return max(int(round(field.sz * 8)), 0)
        except Exception:
            return 0


# ── Core extractor ──────────────────────────────────────────────────────────

def extract_field_layout(pkt: Packet) -> tuple[list[int], list[int]]:
    """
    Walk all layers and return (field_type_per_byte, boundary_per_gap).

    Uses bit-position tracking so BitFields and sub-byte fields are handled
    correctly. When multiple sub-byte fields share a byte, the FIRST one that
    starts in that byte wins (earlier fields have higher semantic priority).
    """
    raw = bytes(pkt)
    n = len(raw)
    ft = [_OPAQ] * n
    labeled = [False] * n  # has this byte been assigned yet?
    boundary_after = [False] * (n - 1) if n > 1 else []  # gap labels

    layer = pkt
    layer_start = 0

    while layer is not None and layer.__class__.__name__ not in ("NoPayload", "Raw"):
        header_raw = layer.self_build()
        layer_len = len(header_raw)

        bit_pos = 0  # bit offset within this layer's header
        for field in layer.fields_desc:
            sz_bits = _field_bit_size(field, layer)
            if sz_bits <= 0:
                continue

            start_bit = bit_pos
            end_bit = bit_pos + sz_bits - 1
            start_byte = start_bit // 8
            end_byte = min(end_bit // 8, layer_len - 1)

            ftype = scapy_field_to_taxonomy_type(field, layer)
            abs_start = layer_start + start_byte
            abs_end = layer_start + end_byte

            for b in range(abs_start, min(abs_end + 1, n)):
                if not labeled[b]:
                    ft[b] = ftype
                    labeled[b] = True

            # Boundary after byte-aligned field ends
            if (bit_pos + sz_bits) % 8 == 0:
                abs_end_aligned = layer_start + (bit_pos + sz_bits) // 8 - 1
                if 0 <= abs_end_aligned < n - 1:
                    boundary_after[abs_end_aligned] = True

            bit_pos += sz_bits

        layer_start += layer_len
        layer = layer.payload if hasattr(layer, "payload") else None

    boundary_per_gap = [1 if b else 0 for b in boundary_after]
    return ft, boundary_per_gap


def make_message_dict(pkt: Packet, format_id: str,
                      endian: str = "big") -> dict:
    """Produce validated ground-truth dict from a Scapy packet."""
    raw = bytes(pkt)
    ft, bg = extract_field_layout(pkt)
    msg = {
        "bytes_hex": raw.hex(),
        "field_type_per_byte": ft,
        "boundary_per_gap": bg,
        "endianness": endian,
        "format_id": format_id,
    }
    validate_message(msg)
    return msg
