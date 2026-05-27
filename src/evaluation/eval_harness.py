"""
eval_harness.py — deterministic protocol parsers and boundary evaluation.

Architecture
------------
  1.  Field       — named byte interval  [start, end)
  2.  parse_*     — bytes → list[Field]  (raw, unmerged, per message)
  3.  apply_constant_merging
                  — list[list[Field]] × list[bytes] → list[set[int]]
                    sample-level: removes boundaries between adjacent fields
                    that are constant across every message in the sample
  4.  ground_truth_for_sample(protocol, messages) → list[set[int]]
  5.  evaluate    — (messages, detected_boundaries, protocol) → metrics dict

Evaluation methodology follows BinaryInferno (Chandler 2023):
  precision = TP / (TP + FP)       [1.0 when both detected and GT are empty]
  recall    = TP / (TP + FN)       [1.0 when GT is empty]
  FPR       = FP / (FP + TN)
  TN        = (msg_len - 1) - TP - FP - FN
Average metrics are computed across the message collection.

Fixes applied vs. original
---------------------------
  [FIX-1]  Tutorial parser: header is type(1) + length(2) + timestamp(4),
           not total_length(4) + message_id(4) + type(1).  Verified from
           the paper's running example: 01 000D 60A67AED 05 APPLE.

  [FIX-2]  Mirai parser: bytes 2–7 are expanded into their proper sub-fields
           (txid, flags, qdcount) instead of a single pre-merged
           "constant_prefix" blob.  apply_constant_merging now decides
           whether to remove their interior boundaries, which is the correct
           contract.  Dataset-verified constants (txid=0x0000,
           flags=0x0001, qdcount=0x0a) will be merged automatically.

  [FIX-3]  DNP3 parser: data section is block-structured (IEEE 1815) —
           every 16 bytes of user data are followed by a 2-byte CRC.
           These boundaries appear in the paper's ground truth; their
           absence explained the anomalously low DNP3 recall.

  [FIX-4]  SMB1 parser: word_count (wc) is capped so that a large or
           malformed wc byte cannot generate fields beyond the message end
           before valid_fields can filter them.  data_end is also capped.

  [FIX-5]  DHCP parser: option value truncation is handled explicitly; the
           code no longer appends a value field whose end would exceed
           len(msg) and then silently drops it from a different code path,
           leaving an orphaned code+length pair in the field list.

  [FIX-6]  _boundary_metrics: precision is 1.0 (not 0.0) when both the
           detected set and the GT set are empty, matching the paper's
           Section IX.A specification for single-field / fully-merged
           formats.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable


# ── Core data structure ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Field:
    name: str
    start: int
    end: int

    def value(self, msg: bytes) -> bytes:
        return msg[self.start:self.end]


def valid_fields(fields: list[Field], msg_len: int) -> list[Field]:
    """Drop fields that are empty or whose interval exceeds the message."""
    return [
        f for f in fields
        if 0 <= f.start < f.end <= msg_len
    ]


def fields_to_boundaries(fields: list[Field], msg_len: int) -> set[int]:
    """Interior right-endpoints of valid fields (positions 1 .. msg_len-1)."""
    return {f.end for f in fields if 0 < f.end < msg_len}


# ── Protocol parsers (unmerged) ───────────────────────────────────────────────

def parse_ntp48(msg: bytes) -> list[Field]:
    """
    RFC 5905 §7.3 — 48-byte fixed NTP packet header.
    All fields are verified against the RFC field table.
    """
    fields = [
        Field("li_vn_mode",          0,  1),   # 1 B: LI(2b) + VN(3b) + Mode(3b)
        Field("stratum",             1,  2),   # 1 B
        Field("poll",                2,  3),   # 1 B: log2 seconds
        Field("precision",           3,  4),   # 1 B: signed, log2 seconds
        Field("root_delay",          4,  8),   # 4 B: NTP short format
        Field("root_dispersion",     8, 12),   # 4 B: NTP short format
        Field("ref_clock_id",       12, 16),   # 4 B: reference clock identifier
        Field("ref_timestamp",      16, 24),   # 8 B: NTP timestamp
        Field("origin_timestamp",   24, 32),   # 8 B: NTP timestamp
        Field("receive_timestamp",  32, 40),   # 8 B: NTP timestamp
        Field("transmit_timestamp", 40, 48),   # 8 B: NTP timestamp
    ]
    return valid_fields(fields, len(msg))


def parse_mavlink_top(msg: bytes) -> list[Field]:
    """
    MAVLink v1 frame: STX(1) + LEN(1) + SEQ(1) + SYS(1) + COMP(1) +
                      MSG_ID(1) + PAYLOAD(LEN) + CRC(2).
    payload_len is read from msg[1] to set variable-length payload boundary.
    """
    if len(msg) < 8:
        return []
    payload_len = msg[1]
    payload_end = 6 + payload_len
    fields = [
        Field("stx",          0, 1),
        Field("payload_len",  1, 2),
        Field("sequence",     2, 3),
        Field("system_id",    3, 4),
        Field("component_id", 4, 5),
        Field("message_id",   5, 6),
        Field("payload",      6, payload_end),
        Field("crc",          payload_end, payload_end + 2),
    ]
    return valid_fields(fields, len(msg))


def parse_tutorial(msg: bytes) -> list[Field]:
    """
    Tutorial protocol (jhalon.github.io/reverse-engineering-protocols).

    [FIX-1] Correct header layout verified from the paper's running example
    (Section III): 01 000D 60A67AED 05 4150504C45
      byte  0    : type       (1 B) — 0x01
      bytes 1–2  : length     (2 B) — 0x000D = 13 (total message length)
      bytes 3–6  : timestamp  (4 B) — 0x60A67AED (Unix epoch, May 2021)
      bytes 7+   : LV pairs until null terminator

    The original parser had total_length(4)+message_id(4)+type(1) which
    decodes the first four bytes as 0x01000D60 = 16,781,664 — nonsensical
    as a length — and was completely wrong.
    """
    fields = [
        Field("type",      0, 1),   # 1 B: message type
        Field("length",    1, 3),   # 2 B: total message length (BE)
        Field("timestamp", 3, 7),   # 4 B: Unix epoch timestamp (BE)
    ]
    offset = 7
    idx = 0
    while offset < len(msg):
        lv_len = msg[offset]
        if lv_len == 0:
            fields.append(Field("null_terminator", offset, offset + 1))
            break
        value_end = offset + 1 + lv_len
        fields.append(Field(f"lv_{idx}_length", offset,     offset + 1))
        fields.append(Field(f"lv_{idx}_value",  offset + 1, value_end))
        offset = value_end
        idx += 1
    return valid_fields(fields, len(msg))


def parse_bgp_top(msg: bytes) -> list[Field]:
    """
    RFC 4271 §4.1 — BGP message header (top-level, mixed-type sample).
    Only the common 19-byte prefix is parsed; per-type body is left as
    an opaque payload blob.
      bytes  0–16 : Marker (16 B, all 0xFF)
      bytes 16–18 : Length (2 B, BE)
      byte  18    : Type   (1 B)
      bytes 19+   : Payload (type-specific body)
    """
    fields = [
        Field("marker",  0, 16),
        Field("length", 16, 18),
        Field("type",   18, 19),
    ]
    if len(msg) > 19:
        fields.append(Field("payload", 19, len(msg)))
    return valid_fields(fields, len(msg))


def parse_dhcp(msg: bytes) -> list[Field]:
    """
    RFC 2131 fixed header (240 B) + TLV options.

    Fixed fields follow RFC 2131 Table 1 exactly.  chaddr is the full
    16-byte RFC field (hardware address + padding); any dataset-specific
    split at byte 34 is not applied here.

    [FIX-5] Option parsing now explicitly guards against a value_end that
    would exceed len(msg).  When a truncated option is detected the field
    is recorded as a single truncated entry and the loop exits, preventing
    an orphaned code+length pair in the field list.
    """
    fields = [
        Field("op",      0,   1),
        Field("htype",   1,   2),
        Field("hlen",    2,   3),
        Field("hops",    3,   4),
        Field("xid",     4,   8),
        Field("secs",    8,  10),
        Field("flags",  10,  12),
        Field("ciaddr", 12,  16),
        Field("yiaddr", 16,  20),
        Field("siaddr", 20,  24),
        Field("giaddr", 24,  28),
        Field("chaddr", 28,  44),   # 16 B: RFC chaddr field (incl. padding)
        Field("sname",  44, 108),   # 64 B: server host name
        Field("file",  108, 236),   # 128 B: boot file name
        Field("magic", 236, 240),   # 4 B: magic cookie (99.130.83.99)
    ]
    offset = 240
    opt_idx = 0
    while offset < len(msg):
        code = msg[offset]

        # End option — single byte, terminates option list
        if code == 0xFF:
            fields.append(Field("opt_end", offset, offset + 1))
            break

        # Pad option — single byte, no length, no value
        if code == 0x00:
            fields.append(Field(f"opt_{opt_idx}_pad", offset, offset + 1))
            offset += 1
            opt_idx += 1
            continue

        # Need at least 2 bytes (code + length) to proceed
        if offset + 2 > len(msg):
            fields.append(Field(f"opt_{opt_idx}_trunc", offset, len(msg)))
            break

        llen = msg[offset + 1]
        value_end = offset + 2 + llen

        # [FIX-5] Check whether the value would extend past the message.
        # If so, record a single truncated field and stop — do not emit the
        # code and length fields separately, which would leave them without
        # their corresponding value field.
        if value_end > len(msg):
            fields.append(Field(f"opt_{opt_idx}_trunc", offset, len(msg)))
            break

        fields.append(Field(f"opt_{opt_idx}_code",   offset,     offset + 1))
        fields.append(Field(f"opt_{opt_idx}_length", offset + 1, offset + 2))
        fields.append(Field(f"opt_{opt_idx}_value",  offset + 2, value_end))
        offset = value_end
        opt_idx += 1

    return valid_fields(fields, len(msg))


def parse_dnp3_link(msg: bytes) -> list[Field]:
    """
    IEEE 1815 DNP3 link-layer frame.

    Fixed link header (10 B):
      bytes 0–2  : start bytes (0x0564)
      byte  2    : length
      byte  3    : control
      bytes 4–6  : destination address (LE)
      bytes 6–8  : source address (LE)
      bytes 8–10 : header CRC

    [FIX-3] Data section block structure:
      Every 16 bytes of application data are followed by a 2-byte CRC.
      The final (possibly short) data block is also followed by 2-byte CRC.
      These intra-data boundaries appear in the paper's ground truth;
      their absence explained DNP3 recall = 0.31 in BinaryInferno Table V
      (only the 6 link-header boundaries were being detected).
    """
    fields = [
        Field("start",      0,  2),
        Field("length",     2,  3),
        Field("control",    3,  4),
        Field("dst",        4,  6),
        Field("src",        6,  8),
        Field("header_crc", 8, 10),
    ]
    offset = 10
    block_idx = 0
    while offset < len(msg):
        remaining = len(msg) - offset
        # Need at least 3 bytes for a minimal data+CRC block (1 data + 2 CRC)
        if remaining <= 2:
            break
        block_data_len = min(16, remaining - 2)
        fields.append(Field(
            f"data_block_{block_idx}",
            offset,
            offset + block_data_len,
        ))
        fields.append(Field(
            f"crc_{block_idx}",
            offset + block_data_len,
            offset + block_data_len + 2,
        ))
        offset += block_data_len + 2
        block_idx += 1

    return valid_fields(fields, len(msg))


def parse_mirai(msg: bytes) -> list[Field]:
    """
    Mirai C2 DNS-over-TCP: Q(V^5)^[Q] · Q(TL(V)[L])^[Q]

    [FIX-2] Bytes 2–7 are expanded into their proper semantic sub-fields
    instead of being bundled into a single "constant_prefix" blob.

      bytes 0–2  : tcp_length  (2 B)
      bytes 2–4  : txid        (2 B) — constant 0x0000 in captures
      bytes 4–6  : flags       (2 B) — constant 0x0001 in captures
      bytes 6–7  : qdcount     (1 B) — constant 0x0a   in captures

    apply_constant_merging will fuse txid+flags and flags+qdcount
    boundaries automatically when they are constant, reproducing the
    same net effect as the old hardcoded blob — but correctly, because
    the merger checks constancy from data rather than assuming it.
    It also leaves the interior boundaries intact if they ever vary.

    After qdcount the frame is:
      Q(V^5)^[Q1]:  q1_count blocks of 5 bytes each
      Q(TLV)^[Q2]: q2_count type-length-value triples
    """
    fields = [
        Field("tcp_length", 0, 2),
        Field("txid",       2, 4),   # DNS transaction ID
        Field("flags",      4, 6),   # DNS flags
        Field("qdcount",    6, 7),   # question count (= q1 below)
    ]

    if len(msg) < 9:
        return valid_fields(fields, len(msg))

    q1 = msg[7]
    fields.append(Field("q1_count", 7, 8))
    offset = 8
    for i in range(q1):
        block_end = offset + 5
        if block_end > len(msg):
            fields.append(Field(f"q1_block_{i}_trunc", offset, len(msg)))
            return valid_fields(fields, len(msg))
        fields.append(Field(f"q1_block_{i}", offset, block_end))
        offset = block_end

    if offset >= len(msg):
        return valid_fields(fields, len(msg))

    q2 = msg[offset]
    fields.append(Field("q2_count", offset, offset + 1))
    offset += 1

    for i in range(q2):
        if offset + 2 > len(msg):
            fields.append(Field(f"tlv_{i}_trunc", offset, len(msg)))
            break
        llen = msg[offset + 1]
        value_end = offset + 2 + llen
        if value_end > len(msg):
            fields.append(Field(f"tlv_{i}_trunc", offset, len(msg)))
            break
        fields.append(Field(f"tlv_{i}_type",   offset,     offset + 1))
        fields.append(Field(f"tlv_{i}_length", offset + 1, offset + 2))
        fields.append(Field(f"tlv_{i}_value",  offset + 2, value_end))
        offset = value_end

    return valid_fields(fields, len(msg))


def parse_modbus_tcp(msg: bytes) -> list[Field]:
    """
    Modbus Application Protocol v1.1b3 — MBAP header + PDU.

      bytes 0–2  : transaction_id  (2 B, BE)
      bytes 2–4  : protocol_id     (2 B, BE) — always 0x0000
      bytes 4–6  : length          (2 B, BE) — byte count from unit_id onward
      byte  6    : unit_id         (1 B)
      byte  7    : function_code   (1 B)
      bytes 8+   : data            (variable)

    Note: protocol_id is constant (0x0000) across all Modbus TCP messages.
    apply_constant_merging will therefore remove the boundary at byte 4
    whenever protocol_id and its neighbour(s) are also constant, which is
    consistent with BinaryInferno's ground-truth methodology.
    """
    fields = [
        Field("transaction_id", 0, 2),
        Field("protocol_id",    2, 4),
        Field("length",         4, 6),
        Field("unit_id",        6, 7),
        Field("function_code",  7, 8),
    ]
    if len(msg) > 8:
        fields.append(Field("data", 8, len(msg)))
    return valid_fields(fields, len(msg))


def parse_smb1(msg: bytes) -> list[Field]:
    """
    SMB1 common 32-byte header + Q(VV)^[Q] · LL(V)^[LL] body.

    Fixed header (32 B):
      bytes  0–4   : protocol    (0xFF 'S' 'M' 'B')
      byte   4     : command
      bytes  5–9   : status      (NT status, 4 B)
      byte   9     : flags
      bytes 10–12  : flags2
      bytes 12–14  : pid_high
      bytes 14–22  : security_features  (8 B)
      bytes 22–24  : reserved
      bytes 24–26  : tid
      bytes 26–28  : pid
      bytes 28–30  : uid
      bytes 30–32  : mid

    Variable body:
      byte  32     : word_count (wc)
      bytes 33–33+wc*2 : parameter words  (wc × 2 B each)
      bytes 33+wc*2 .. +2 : byte_count (2 B, LE)
      remaining    : data

    [FIX-4] wc is capped to the maximum words that can physically fit
    between byte 33 and the end of the message, preventing field index
    overflow before valid_fields can act.  data_end is also capped.
    """
    fields = [
        Field("protocol",           0,  4),
        Field("command",            4,  5),
        Field("status",             5,  9),
        Field("flags",              9, 10),
        Field("flags2",            10, 12),
        Field("pid_high",          12, 14),
        Field("security_features", 14, 22),
        Field("reserved",          22, 24),
        Field("tid",               24, 26),
        Field("pid",               26, 28),
        Field("uid",               28, 30),
        Field("mid",               30, 32),
    ]

    if len(msg) <= 32:
        return valid_fields(fields, len(msg))

    # [FIX-4] Cap wc to what the message can physically contain.
    raw_wc = msg[32]
    max_wc = max(0, (len(msg) - 33)) // 2
    wc = min(raw_wc, max_wc)

    fields.append(Field("word_count", 32, 33))
    params_start = 33
    params_end   = params_start + wc * 2

    for i in range(wc):
        s = params_start + i * 2
        fields.append(Field(f"param_word_{i}", s, s + 2))

    if params_end + 2 <= len(msg):
        bc = struct.unpack_from("<H", msg, params_end)[0]
        data_start = params_end + 2
        # [FIX-4] Cap data_end so it never exceeds the message length.
        data_end = min(data_start + bc, len(msg))
        fields.append(Field("byte_count", params_end, params_end + 2))
        if data_end > data_start:
            fields.append(Field("data", data_start, data_end))

    return valid_fields(fields, len(msg))


def parse_smb2_top(msg: bytes) -> list[Field]:
    """
    MS-SMB2 §2.2.1 — SMB2 packet header (SYNC form, 64 B) + body.

      bytes  0–4   : protocol_id    (0xFE 'S' 'M' 'B')
      bytes  4–6   : structure_size (always 64)
      bytes  6–8   : credit_charge
      bytes  8–12  : status         (or ChannelSeq+Reserved in SMB 3.x req)
      bytes 12–14  : command
      bytes 14–16  : credits        (CreditRequest / CreditResponse)
      bytes 16–20  : flags
      bytes 20–24  : next_command
      bytes 24–32  : message_id     (8 B)
      bytes 32–36  : reserved       (ProcessId in SYNC; low half of AsyncId
                                     in ASYNC — named "reserved" per spec
                                     since the server MAY ignore it)
      bytes 36–40  : tree_id
      bytes 40–48  : session_id     (8 B)
      bytes 48–64  : signature      (16 B)
      bytes 64+    : body

    Note on ASYNC messages: when SMB2_FLAGS_ASYNC_COMMAND is set, bytes
    32–40 form a single 8-byte AsyncId rather than Reserved(4)+TreeId(4).
    Mixed SYNC/ASYNC samples will show varying bytes in this range;
    apply_constant_merging will retain the boundary at byte 36 since
    tree_id is not constant in such samples.  This is correct behaviour.
    """
    fields = [
        Field("protocol_id",    0,  4),
        Field("structure_size", 4,  6),
        Field("credit_charge",  6,  8),
        Field("status",         8, 12),
        Field("command",       12, 14),
        Field("credits",       14, 16),
        Field("flags",         16, 20),
        Field("next_command",  20, 24),
        Field("message_id",    24, 32),
        Field("reserved",      32, 36),   # ProcessId (SYNC) / AsyncId-low (ASYNC)
        Field("tree_id",       36, 40),
        Field("session_id",    40, 48),
        Field("signature",     48, 64),
    ]
    if len(msg) > 64:
        fields.append(Field("body", 64, len(msg)))
    return valid_fields(fields, len(msg))


# ── Parser registry ───────────────────────────────────────────────────────────

PARSERS: dict[str, Callable[[bytes], list[Field]]] = {
    "ntp48":         parse_ntp48,
    "mavlink":       parse_mavlink_top,
    "tutorial":      parse_tutorial,
    "bgp":           parse_bgp_top,
    "dhcp":          parse_dhcp,
    "dnp3":          parse_dnp3_link,
    "mirai":         parse_mirai,
    "modbus":        parse_modbus_tcp,
    "modbus_binpre": parse_modbus_tcp,   # same format, different dataset label
    "smb":           parse_smb1,
    "smb2":          parse_smb2_top,
}


# ── Sample-level constant-field merging ───────────────────────────────────────

def _field_is_constant(
    parsed: list[list[Field]],
    messages: list[bytes],
    field_index: int,
) -> bool:
    """
    Return True iff field at position field_index carries identical bytes
    in every message in the sample.

    If any message has fewer than field_index+1 fields (which can happen
    in variable-length protocols once valid_fields has truncated), return
    False so that the boundary is conservatively retained.
    """
    values: list[bytes] = []
    for fields, msg in zip(parsed, messages):
        if field_index >= len(fields):
            return False
        values.append(fields[field_index].value(msg))
    return len(set(values)) == 1


def apply_constant_merging(
    parsed: list[list[Field]],
    messages: list[bytes],
) -> list[set[int]]:
    """
    For each pair of adjacent fields (i, i+1) that are BOTH constant
    across the entire sample, remove the boundary between them — i.e.
    exclude field[i].end from the ground-truth boundary set.

    Alignment is by field index; only indices 0 .. min_field_count-1 are
    checked, which correctly handles variable-length tails in DHCP, SMB1,
    Mirai, etc. without comparing fields that are structurally different
    across messages.

    This reproduces the methodology described in BinaryInferno (Chandler
    2023) §IX "Establishing ground-truth": "when adjacent fields in a
    sample had constant values across all messages, we updated our
    ground-truth to merge these fields."
    """
    if not parsed:
        return []

    min_len = min(len(f) for f in parsed)
    skip_end: set[int] = set()   # field indices whose .end boundary is removed
    for i in range(min_len - 1):
        if (_field_is_constant(parsed, messages, i) and
                _field_is_constant(parsed, messages, i + 1)):
            skip_end.add(i)

    result: list[set[int]] = []
    for fields, msg in zip(parsed, messages):
        boundaries: set[int] = set()
        for i, f in enumerate(fields):
            if i in skip_end:
                continue
            if 0 < f.end < len(msg):
                boundaries.add(f.end)
        result.append(boundaries)
    return result


def ground_truth_for_sample(
    protocol: str,
    messages: list[bytes],
) -> list[set[int]]:
    """
    Parse every message with the appropriate protocol parser, apply
    sample-level constant-field merging, and return one boundary set per
    message.  This is the entry point used by evaluate().
    """
    parser = PARSERS[protocol]
    parsed = [parser(msg) for msg in messages]
    return apply_constant_merging(parsed, messages)


# ── Backward-compatible per-message GT shim ───────────────────────────────────
# External callers that use GROUND_TRUTH[proto](msg) receive unmerged parser
# boundaries.  For sample-accurate evaluation use ground_truth_for_sample().

def _make_gt_fn(parser: Callable[[bytes], list[Field]]):
    def gt_fn(msg: bytes) -> set[int]:
        return fields_to_boundaries(parser(msg), len(msg))
    return gt_fn

GROUND_TRUTH: dict[str, Callable[[bytes], set[int]]] = {
    proto: _make_gt_fn(parser) for proto, parser in PARSERS.items()
}


# ── Endianness metadata ───────────────────────────────────────────────────────

PROTOCOL_ENDIAN: dict[str, str] = {
    "bgp":           "BE",
    "dhcp":          "BE",
    "dnp3":          "LE",
    "mavlink":       "LE",
    "mirai":         "BE",
    "modbus":        "BE",
    "modbus_binpre": "BE",
    "ntp48":         "BE",
    "smb":           "LE",
    "smb2":          "LE",
    "tutorial":      "BE",
}


# ── Metric computation ────────────────────────────────────────────────────────

def _boundary_metrics(
    detected: set[int],
    ground_truth: set[int],
    msg_len: int,
) -> dict[str, float]:
    """
    Compute TP/FP/FN/TN and derived metrics for one message.

    Possible boundary positions are {1, 2, ..., msg_len-1}.
    Positives  = ground-truth boundaries.
    Negatives  = adjacent-byte pairs belonging to the same ground-truth field.
    TP = detected ∩ GT
    FP = detected − GT
    FN = GT − detected
    TN = possible − GT − detected

    [FIX-6] precision = 1.0 when both detected and GT are empty.
    This matches BinaryInferno §IX.A: "For formats consisting of a single
    field, or formats where all fields are constant valued for a sample,
    we report precision of 1.0, recall of 1.0, and a FPR of 0.0 if no
    field boundaries are inferred by a tool."
    """
    possible = set(range(1, msg_len))
    gt  = ground_truth & possible
    det = detected     & possible

    tp = len(det & gt)
    fp = len(det - gt)
    fn = len(gt  - det)
    tn = len(possible - gt - det)

    # [FIX-6] precision: 1.0 when both empty (true negative: no spurious
    # boundaries and none expected), 0.0 when detected but GT is empty.
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if fn == 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1     = ((2 * precision * recall) / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, fpr=fpr, f1=f1,
    )


def evaluate(
    messages: list[bytes],
    detected_boundaries: set[int],
    protocol: str,
) -> dict[str, float]:
    """
    Evaluate a detector against ground truth for a protocol sample.

    detected_boundaries: a single fixed-offset set (as returned by the BI
    entropy detector, NI surprise detector, or any other method) applied
    uniformly to all messages in the sample.

    Ground truth is computed per-message via ground_truth_for_sample(),
    which applies sample-level constant-field merging.  Metrics are
    averaged across the message collection.
    """
    gt_sets = ground_truth_for_sample(protocol, messages)
    all_metrics = [
        _boundary_metrics(detected_boundaries, gt, len(msg))
        for msg, gt in zip(messages, gt_sets)
    ]

    n   = len(all_metrics)
    avg = {
        k: sum(m[k] for m in all_metrics) / n
        for k in ("precision", "recall", "fpr", "f1", "tp", "fp", "fn", "tn")
    }
    avg["n_messages"] = float(n)
    avg["n_detected"] = float(len(detected_boundaries))
    return avg


def print_results_table(results: dict) -> None:
    """Pretty-print a {(protocol, detector): metrics} results dict."""
    header = (
        f"{'Protocol':<12} {'Detector':<22} "
        f"{'Prec':>6} {'Rec':>6} {'FPR':>6} {'F1':>6} {'#Det':>5}"
    )
    print(header)
    print("-" * len(header))
    for (proto, detector), metrics in sorted(results.items()):
        print(
            f"{proto:<12} {detector:<22} "
            f"{metrics['precision']:6.3f} {metrics['recall']:6.3f} "
            f"{metrics['fpr']:6.3f} {metrics['f1']:6.3f} "
            f"{int(metrics['n_detected']):5d}"
        )


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """
    Smoke-tests for the three parsers that were changed (Tutorial, Mirai,
    DNP3) plus the precision edge-case fix.  Run with:  python eval_harness.py
    """
    # ── Tutorial: verify the paper's running example decodes correctly ────────
    msgs_tutorial = [
        bytes.fromhex("01000D60A67AED054150504C45"),        # APPLE
        bytes.fromhex("01000E60A67AF9064F52414E4745"),       # ORANGE
        bytes.fromhex("01001160A67B0504504C554D0450454152"), # PLUM + PEAR
    ]
    gt_t = ground_truth_for_sample("tutorial", msgs_tutorial)
    # Boundaries expected at 1 (type|length), 3 (length|timestamp),
    # 7 (timestamp|lv_0_length) then lv boundaries.
    # After constant merging: type(1B) is constant (0x01), length(2B) varies,
    # so boundary at 1 is kept.  Timestamp varies, so 3 and 7 are kept.
    assert 1 in gt_t[0], "Tutorial: expected boundary at byte 1 (type|length)"
    assert 3 in gt_t[0], "Tutorial: expected boundary at byte 3 (length|timestamp)"
    assert 7 in gt_t[0], "Tutorial: expected boundary at byte 7 (timestamp|LV)"
    print("[PASS] Tutorial parser boundaries correct")

    # ── Mirai: sub-fields are now present; merger decides on constancy ────────
    # Build two minimal Mirai-like messages to test the expander.
    # txid=0x0000, flags=0x0001, qdcount=0x0a are constant → merged.
    # tcp_length varies → boundary at 2 survives.
    msg_mirai_a = bytes([
        0x00, 0x16,             # tcp_length (varies)
        0x00, 0x00,             # txid (constant)
        0x00, 0x01,             # flags (constant)
        0x0a,                   # qdcount (constant)
        0x02,                   # q1_count = 2
        0x01, 0x02, 0x03, 0x04, 0x05,   # q1_block_0
        0x01, 0x02, 0x03, 0x04, 0x05,   # q1_block_1
        0x00,                   # q2_count = 0
    ])
    msg_mirai_b = bytes([
        0x00, 0x18,             # tcp_length (different)
        0x00, 0x00,             # txid (same)
        0x00, 0x01,             # flags (same)
        0x0a,                   # qdcount (same)
        0x02,                   # q1_count = 2
        0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x00,
    ])
    parsed_m = [parse_mirai(m) for m in [msg_mirai_a, msg_mirai_b]]
    # Check that txid, flags, qdcount fields exist in the parsed output
    field_names_m = {f.name for f in parsed_m[0]}
    assert "txid"    in field_names_m, "Mirai: txid field missing"
    assert "flags"   in field_names_m, "Mirai: flags field missing"
    assert "qdcount" in field_names_m, "Mirai: qdcount field missing"
    gt_m = apply_constant_merging(parsed_m, [msg_mirai_a, msg_mirai_b])
    # Boundary at 2 (tcp_length|txid) should survive — tcp_length varies
    assert 2 in gt_m[0], "Mirai: boundary at 2 should survive (tcp_length varies)"
    # Boundaries at 4 and 6 should be merged (txid, flags, qdcount all constant)
    assert 4 not in gt_m[0], "Mirai: boundary at 4 should be merged (txid constant)"
    assert 6 not in gt_m[0], "Mirai: boundary at 6 should be merged (flags constant)"
    print("[PASS] Mirai parser sub-fields and constant merging correct")

    # ── DNP3: CRC block structure ─────────────────────────────────────────────
    # Construct a minimal DNP3 frame: 10B header + 18B data
    # (16B data block + 2B CRC)
    dnp3_msg = bytes(range(10)) + bytes(range(16)) + bytes([0xAB, 0xCD])
    parsed_d = parse_dnp3_link(dnp3_msg)
    field_names_d = {f.name for f in parsed_d}
    assert "data_block_0" in field_names_d, "DNP3: data_block_0 missing"
    assert "crc_0"        in field_names_d, "DNP3: crc_0 missing"
    # data_block_0 should be bytes 10–26, crc_0 bytes 26–28
    db0 = next(f for f in parsed_d if f.name == "data_block_0")
    assert db0.start == 10 and db0.end == 26, f"DNP3: data_block_0 offsets wrong: {db0}"
    crc0 = next(f for f in parsed_d if f.name == "crc_0")
    assert crc0.start == 26 and crc0.end == 28, f"DNP3: crc_0 offsets wrong: {crc0}"
    print("[PASS] DNP3 CRC-block structure correct")

    # ── Precision edge case: both empty → 1.0, not 0.0 ───────────────────────
    m = _boundary_metrics(set(), set(), 5)
    assert m["precision"] == 1.0, f"precision should be 1.0 for empty GT+det, got {m['precision']}"
    assert m["recall"]    == 1.0, "recall should be 1.0 for empty GT"
    assert m["fpr"]       == 0.0, "FPR should be 0.0 for empty detected"
    print("[PASS] Precision edge case (empty GT ∩ empty detected) = 1.0")

    # Detected but GT empty → precision 0.0
    m2 = _boundary_metrics({2, 3}, set(), 5)
    assert m2["precision"] == 0.0, "precision should be 0.0 when GT empty but det non-empty"
    print("[PASS] Precision edge case (non-empty detected, empty GT) = 0.0")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()