# Design Decisions and Deviations

## Ground Truth for Evaluation (Phase 5)

**Decision:** LOPO evaluation uses Tier-2 Scapy-extracted messages as the test
set rather than the BI benchmark `.txt.input` files.

**Reason:** The BI benchmark files (`data/tier3_benchmark/top-level/{100,500,1000}/*.txt.input`)
contain raw hex messages with no per-byte or per-gap ground-truth labels.
BinaryInferno is an unsupervised system — the paper authors evaluated it by
manually inspecting output against known protocol specifications, not by
computing automatic metrics against labels.

For automatic metric computation (Precision/Recall/F1), both our model and BI
are evaluated against Scapy-extracted ground truth on the same Tier-2 test
messages.  This gives apples-to-apples comparison on the same ground truth.

## LOPO Protocol Set

Our LOPO protocols (`bgp_raw`, `dhcp`, `modbus`, `ntp`, `dns`, `tcp`, `udp`,
`ip`, `icmp`, `arp`) are the TCP/IP-stack protocols we have in Tier 2 with
ground truth.  The BI benchmark includes some different protocols (DNP3,
Mavlink, Mirai, SMB, SMB2, Tutorial) that we do not cover in Tier 2.
These are noted as future work.

## DNS Compression Pointers

DNS name sections use `_DNSPacketListField` in Scapy, which is extracted as
OPAQUE.  Pointer bytes (0xC0 prefix) inherit the OPAQUE label rather than a
new POINTER type.  "Your call" per the brief — OPAQUE is chosen for simplicity.

## BGP Raw

BGP messages are hand-labeled (`_label_bgp_raw`) in `generate_tier2.py`
because Scapy's BGP dissector does not always expose clean per-byte field
widths.  The hand-labeling covers OPEN and KEEPALIVE message types.

## ByteLM max_len vs Encoder max_len

Stage 1 pretrains ByteLM with `max_len=256`.  Stage 2 uses `max_len=512` for
the encoder.  When loading Stage-1 LM weights into Stage 2, `pos_embedding`
shape mismatches (`[256, 64]` vs `[512, 64]`) are handled by skipping the
mismatched key and using random initialization for the extended positions.
All other LM weights are transferred correctly.

## BI Comparison Methodology

BI's output is parsed from its `INFERRED DESCRIPTION` block.  Each message
line (e.g., `01 000D | 60A67AED 054150504C45`) is split on space and `|`
delimiters; each contiguous hex group is one field; delimiters mark boundaries.
The per-message `boundary_per_gap` is reconstructed directly from this.

BI baseline runs use at most 100 messages per protocol (configurable via
`--bi_max_msgs`) because BI can be slow on large message sets.

## Ablation Checkpoints

The `results/ablations.csv` file is pre-populated with placeholder rows.
Each ablation requires a separate Stage 2 + Stage 3 training run.  Re-run
`train_main.py` with the ablation flag and then `run_eval.py` to fill in
the results.
