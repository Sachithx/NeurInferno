# NeurInferno

NeurInferno infers field boundaries in unlabeled binary protocol messages.
A byte-level transformer reads a batch of messages from the same format and
injects cross-message statistics (mean / max / variance at each offset) after
every layer. A frozen byte language model supplies per-byte entropy features.
The training objective is per-gap boundary prediction plus a per-byte field-type
auxiliary loss.

Training is leakage-free leave-one-protocol-out (LOPO) and leave-four-protocol-out
(L4PO): the held-out protocol is excluded from the language model, the main
model, and the fine-tune. Grammar data in `data/grammar/` is synthetic
pretraining; `data/protocols/` holds the twelve labeled protocol traces.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ and a CUDA GPU are required.

## Checkpoints

Weights are not in git (~400 MB). Download them, then evaluate:

```bash
bash download_checkpoints.sh
CUDA_VISIBLE_DEVICES=0 bash eval.sh
```

If `checkpoints/seed789/` is already on disk, skip the download and run
`bash eval.sh` only. Eval writes CSVs under `results/seed789/`.

## Run

Train from scratch, then evaluate. Existing weights under `checkpoints/seed789/`
are skipped, so delete them first:

```bash
rm -rf checkpoints/seed789
CUDA_VISIBLE_DEVICES=0 bash train.sh
```

## Custom data

Every protocol or grammar format is a directory containing one `messages.jsonl`.
Each line is a JSON object:

```json
{
  "bytes_hex": "0001080006040001...",
  "field_type_per_byte": [2, 2, 2, 2, 1, 1, ...],
  "boundary_per_gap": [0, 1, 0, 1, 1, 1, ...],
  "format_id": "myproto_00000",
  "endianness": "big"
}
```

| Field | Required | Meaning |
|---|---|---|
| `bytes_hex` | yes | Message bytes as a lowercase hex string (even length). |
| `boundary_per_gap` | yes | Length `n_bytes - 1`. `1` = field boundary after that byte, `0` = same field. This is the evaluation label. |
| `field_type_per_byte` | yes | Length `n_bytes`. Integer type ids (see below). Used as an auxiliary training target. |
| `format_id` | yes | Format name. Lines whose id ends with `_corrupted` are dropped at eval. |
| `endianness` | no | `"big"` or `"little"`. |

Type ids: `0` UNKNOWN, `1` LENGTH, `2` TYPE_TAG, `3` QUANTITY, `4` TIMESTAMP,
`5` ADDRESS, `6` PORT, `7` FLAGS, `8` CHECKSUM, `9` COUNTER, `10` ASCII,
`11` ENUM, `12` FLOAT, `13` INTEGER, `14` OPAQUE, `15` PADDING, `16` RESERVED.
Use `0` when the type is unknown; `14` for unstructured payload.

The last 20% of lines in each protocol file is held out as the test set (by
file order, before dropping corrupted lines). Put train messages first, test
messages last.

### Add a protocol for training

Create `data/protocols/<name>/messages.jsonl`. Any such folder is loaded as
training data automatically (except the protocol held out in that LOPO/L4PO
fold). Extra grammar formats go under `data/grammar/<name>/messages.jsonl`
the same way.

### Add a protocol to LOPO / L4PO eval

1. Add `data/protocols/<name>/messages.jsonl` as above.
2. Append `<name>` to `LOPO_PROTOCOLS` in `src/training/dataset.py`.
3. Append `<name>` to `LOPO_PROTOCOLS` in `train.sh`.
4. For L4PO, add `<name>` to one or more entries in `L4PO_SPLITS` in `train.sh`.
5. Retrain that fold (`rm -rf checkpoints/seed789/lopo/<name>` then `bash train.sh`),
   or eval if a matching checkpoint already exists.

Directory names must be unique, contain no spaces, and should use only
`[a-z0-9_]` so L4PO split labels stay parseable.
