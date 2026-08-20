# NeurInferno

NeurInferno infers **field boundaries** in unlabeled binary protocol messages.
A byte-level transformer reads a batch of messages from the same format and
injects cross-message statistics (mean / max / variance at each offset) after
every layer. A frozen byte language model supplies per-byte entropy features.

The evaluated output is **per-gap cuts**, not field names.

- **Demo:** [Hugging Face Space](https://huggingface.co/spaces/sachithabey/neurinferno)
- **Weights:** [sachithabey/neurinferno](https://huggingface.co/sachithabey/neurinferno)
- **Data:** [sachithabey/neurinferno](https://huggingface.co/datasets/sachithabey/neurinferno) (dataset tab)

## Install

```bash
pip install -e ".[train]"    # this repo, with training extras
```

From GitHub (inference only: torch, einops, huggingface_hub):

```bash
pip install "neurinferno @ git+https://github.com/Sachithx/NeurInferno.git"
```

Python 3.10+. Inference runs on CPU. Training needs a CUDA GPU.

## Inference (load weights from Hugging Face)

The model needs **several messages of the same format**. One packet is the
wrong input.

```python
from neurinferno import FieldBoundaryModel

model = FieldBoundaryModel.from_pretrained()  # downloads ~15 MB once
hex_lines = [
    "0001080006040001900c2d9bfa4649e7160700000000000043f03612",
    "0001080006040002fbccad5c9fb1d014735252376fd2446375217d01",
    # ... more messages of the same format
]
for result in model.infer(hex_lines, threshold=0.75):
    for seg in result.segments:
        print(f"[{seg.start}:{seg.end}] {seg.hex}")
```

Local checkpoint:

```python
model = FieldBoundaryModel.from_checkpoint("path/to/model.ckpt")
```

CLI:

```bash
neurinferno infer messages.hex --threshold 0.75
```

See `examples/infer_hex.py`.

## Data

```bash
huggingface-cli download sachithabey/neurinferno --repo-type dataset --local-dir data
```

That writes `data/protocols/` (12 labeled traces) and `data/grammar/`
(500 synthetic formats). Training and eval expect this layout.

## Train / eval (from a clone)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"
huggingface-cli download sachithabey/neurinferno --repo-type dataset --local-dir data
bash download_checkpoints.sh          # optional: all LOPO/L4PO folds
CUDA_VISIBLE_DEVICES=0 bash eval.sh
```

Train from scratch (skips folds that already have a checkpoint):

```bash
rm -rf checkpoints/seed789
CUDA_VISIBLE_DEVICES=0 bash train.sh
```

LOPO and L4PO are leakage-free: the held-out protocol is excluded from the
language model, the main model, and the fine-tune.

## Custom data

Every protocol or grammar format is a directory containing one `messages.jsonl`.
Each line is a JSON object:

```json
{
  "bytes_hex": "0001080006040001...",
  "field_type_per_byte": [2, 2, 2, 2, 1, 1],
  "boundary_per_gap": [0, 1, 0, 1, 1],
  "format_id": "myproto_00000",
  "endianness": "big"
}
```

| Field | Required | Meaning |
|---|---|---|
| `bytes_hex` | yes | Message bytes as lowercase hex (even length). |
| `boundary_per_gap` | yes | Length `n_bytes - 1`. `1` = field boundary after that byte. This is the evaluation label. |
| `field_type_per_byte` | yes | Length `n_bytes`. Auxiliary training target (type ids below). |
| `format_id` | yes | Lines whose id ends with `_corrupted` are dropped at eval. |
| `endianness` | no | `"big"` or `"little"`. |

Type ids: `0` UNKNOWN, `1` LENGTH, `2` TYPE_TAG, `3` QUANTITY, `4` TIMESTAMP,
`5` ADDRESS, `6` PORT, `7` FLAGS, `8` CHECKSUM, `9` COUNTER, `10` ASCII,
`11` ENUM, `12` FLOAT, `13` INTEGER, `14` OPAQUE, `15` PADDING, `16` RESERVED.

The last 20% of lines in each protocol file is the test split (by file order,
before dropping corrupted lines).

### Add a protocol

1. Create `data/protocols/<name>/messages.jsonl`.
2. Append `<name>` to `LOPO_PROTOCOLS` in `src/neurinferno/training/dataset.py` and in `train.sh`.
3. For L4PO, add `<name>` to one or more `L4PO_SPLITS` in `train.sh`.
4. Retrain that fold: `rm -rf checkpoints/seed789/lopo/<name>` then `bash train.sh`.

Directory names must be unique and `[a-z0-9_]` only.

## License

Apache-2.0. See `LICENSE`.
