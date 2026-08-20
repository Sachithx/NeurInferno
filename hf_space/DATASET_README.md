---
license: apache-2.0
pretty_name: NeurInferno
task_categories:
  - other
tags:
  - protocol
  - binary
  - reverse-engineering
  - field-inference
size_categories:
  - 10K<n<100K
---

# NeurInferno data

Labeled binary messages for field-boundary inference. Two parts:

| Path | What | Size |
|---|---|---|
| `protocols/` | 12 real protocol traces | ~37k messages |
| `grammar/` | 500 synthetic formats | 200 messages each |

Each format is a folder with `messages.jsonl` (one JSON object per line). Grammar folders also have `format_spec.json`.

## Message fields

```json
{
  "bytes_hex": "0001080006040001...",
  "field_type_per_byte": [2, 2, 2, 2, 1, 1],
  "boundary_per_gap": [0, 1, 0, 1, 1],
  "format_id": "tier2_arp_00000",
  "endianness": "big"
}
```

- `bytes_hex` — message bytes as lowercase hex (even length).
- `boundary_per_gap` — length `n_bytes - 1`. `1` = field boundary after that byte.
- `field_type_per_byte` — length `n_bytes`. Type ids below.
- `format_id` — lines ending in `_corrupted` are dropped at evaluation.
- `endianness` — `"big"` or `"little"` (optional).

Type ids: `0` UNKNOWN, `1` LENGTH, `2` TYPE_TAG, `3` QUANTITY, `4` TIMESTAMP,
`5` ADDRESS, `6` PORT, `7` FLAGS, `8` CHECKSUM, `9` COUNTER, `10` ASCII,
`11` ENUM, `12` FLOAT, `13` INTEGER, `14` OPAQUE, `15` PADDING, `16` RESERVED.

The last 20% of lines in each protocol file is the test split (by file order,
before dropping corrupted lines).

## Protocols

| Protocol | Messages | Clean |
|---|---:|---:|
| arp | 4000 | 3569 |
| bgp_raw | 2000 | 1799 |
| dhcp | 4000 | 3584 |
| dns | 6000 | 5420 |
| icmp | 4000 | 3582 |
| igmp | 2000 | 1799 |
| ip | 4000 | 3579 |
| modbus | 1497 | 1340 |
| ntp | 4000 | 3611 |
| ospf | 2000 | 1800 |
| tcp | 4000 | 3618 |
| udp | 4000 | 3570 |

## Download

```bash
huggingface-cli download sachithabey/neurinferno --repo-type dataset --local-dir data
```

That writes `data/protocols/` and `data/grammar/`, which is what `train.sh` and
`eval.sh` expect.
