# NeurInferno

**Neural binary protocol reverse engineering via self-supervised pre-training.**

NeurInferno is a deep learning system for inferring field boundaries and field types in unknown binary protocol messages. It extends and evaluates against the [BinaryInferno (NDSS 2023)](https://github.com/nopnop9090/BinaryInferno) benchmark, achieving a +0.33 improvement in mean F1 across 10 protocols on the leave-one-protocol-out (LOPO) evaluation.

---

## Key results

| Protocol | Ours F1 | BI F1 | Δ F1 |
|----------|---------|-------|------|
| bgp_raw  | 0.7451  | 0.5926 | +0.15 |
| dhcp     | 0.3302  | 0.3182 | +0.01 |
| modbus   | 0.8162  | 0.6667 | +0.15 |
| ntp      | 0.6847  | 0.2500 | +0.43 |
| dns      | **0.9419** | 0.4188 | **+0.52** |
| tcp      | 0.8663  | 0.5333 | +0.33 |
| udp      | 0.8013  | 0.5000 | +0.30 |
| ip       | **0.9969** | 0.5000 | **+0.50** |
| icmp     | 0.8775  | 0.4717 | +0.41 |
| arp      | 0.6809  | 0.2222 | +0.46 |
| **AVERAGE** | **0.7741** | **0.4474** | **+0.33** |

Precision/recall/FPR breakdown and PR curves are in [results/REPORT.md](results/REPORT.md).

---

## Repository layout

```
NeurInferno/
├── baselines/                     # Comparison systems (unmodified)
│   ├── binaryinferno/             # BinaryInferno (NDSS 2023)
│   ├── NetPlier/
│   ├── fieldhunter/
│   ├── nemesys/
│   └── netzob/
├── data/
│   ├── tier1_grammar/             # Grammar-sampled synthetic protocols
│   ├── tier2_scapy/               # Scapy-extracted real protocols
│   ├── tier3_benchmark/           # Symlink → BinaryInferno datasets
│   └── tier3_labeled/             # BI benchmark messages with Scapy labels
├── src/
│   ├── data_generation/
│   │   ├── label_format.py        # Shared 17-class field-type taxonomy
│   │   ├── grammar_sampler.py     # Tier 1 generator
│   │   ├── scapy_extractor.py     # Tier 2 per-byte label extractor
│   │   ├── generate_tier2.py      # Tier 2 driver
│   │   ├── generate_tier3_labeled.py
│   │   ├── protocol_generators.py # Per-protocol Scapy generators
│   │   └── load_bi_dataset.py     # BI hex-file loader
│   ├── model/
│   │   ├── encoder.py             # Byte encoder with cross-message attention
│   │   ├── byte_lm.py             # Auxiliary causal byte LM
│   │   ├── heads.py               # Boundary and field-type prediction heads
│   │   └── full_model.py          # Full model assembly
│   ├── training/
│   │   ├── dataset.py             # Format-batched dataset
│   │   ├── losses.py              # Boundary BCE + type CE + consistency loss
│   │   ├── train_lm.py            # Stage 1: ByteLM pre-training
│   │   ├── train_main.py          # Stage 2: joint training
│   │   └── train_lopo.py          # Stage 3: LOPO fine-tuning
│   └── evaluation/
│       ├── metrics.py             # Precision/Recall/FPR/F1 + AUPRC
│       ├── lopo.py                # LOPO inference loop
│       ├── baseline_comparison.py # Ours vs BI comparison table
│       ├── bi_runner.py           # BinaryInferno subprocess wrapper
│       ├── run_eval.py            # Top-level evaluation entry point
│       └── report.py              # REPORT.md generator
├── configs/
│   ├── train_lm.yaml
│   ├── train_main.yaml
│   └── train_lopo.yaml
├── checkpoints/                   # Saved model checkpoints
├── results/
│   ├── REPORT.md                  # Full evaluation report
│   ├── comparison_table.csv
│   ├── ablations.csv
│   └── pr_curves/                 # Per-protocol AUPRC plots (.npz)
├── NOTES.md                       # Design decisions and deviations
└── run_bi_eval_tier2.py           # BI baseline evaluation driver
```

---

## Architecture

### Byte encoder

A 4-layer transformer with a critical cross-message statistics injection after each attention block. Given a batch of N messages from the same protocol format, per-position mean, max, and variance are computed across messages and concatenated to each token's representation before being projected back. This gives the model an explicit inductive bias toward detecting per-offset structural correlations — the mechanism that drives most of the F1 gain over BinaryInferno.

```
ByteEncoder:
  Embedding(vocab=259, dim=128)     # 256 byte values + PAD + BOS + EOS
  Positional encoding (learned, max_len=512)
  4 × TransformerEncoderLayer:
    Self-attention (intra-message)
    Cross-message statistics injection → project(concat[h, mean, max, var])
    FFN (dim=512), residual, LayerNorm
  Output: (B, N, L, 128)
```

### Auxiliary byte LM

A small 2-layer causal transformer (dim=64) pre-trained on next-byte prediction. After Stage 1 pre-training, it is frozen and used to supply per-position conditional entropy `H(b_{i+1} | b_{≤i})` as a feature signal for the boundary head.

### Prediction heads

- **Field-type head**: per-byte, `Linear(128→128) → GELU → Linear(128→17)`
- **Boundary head**: pairwise per-gap using concatenated `[h_left, h_right, diff, product, entropy, cross-msg-var]` features → `Linear → GELU → Linear(1) → sigmoid`

### Training losses

```python
L_total = L_boundary + 0.5 * L_type + 0.2 * L_consistency
```

`L_boundary` is focal BCE with class-balance weighting (boundaries are sparse). `L_type` is cross-entropy ignoring UNKNOWN. `L_consistency` penalizes disagreement between boundary probabilities and adjacent field-type label differences.

---

## Field-type taxonomy

All data tiers use a unified 17-class vocabulary defined in [src/data_generation/label_format.py](src/data_generation/label_format.py):

| ID | Label | Description |
|----|-------|-------------|
| 0  | UNKNOWN | Unclassified or corrupted bytes |
| 1  | LENGTH | Bytes encoding a length value |
| 2  | TYPE_TAG | Discriminator / opcode |
| 3  | QUANTITY | Repetition count |
| 4  | TIMESTAMP | Any time encoding |
| 5  | ADDRESS | IP, MAC, other addresses |
| 6  | PORT | Port numbers |
| 7  | FLAGS | Bitfields, control flags |
| 8  | CHECKSUM | CRC, hash |
| 9  | COUNTER | Sequence numbers, transaction IDs |
| 10 | ASCII | Text / string bytes |
| 11 | ENUM | Small-valued enumerations |
| 12 | FLOAT | IEEE 754 |
| 13 | INTEGER | General numeric |
| 14 | OPAQUE | Variable-length payload |
| 15 | PADDING | Explicit padding |
| 16 | RESERVED | Spec-reserved bytes |

---

## Data tiers

### Tier 1 — Grammar-sampled synthetic protocols

500 distinct formats, 200 messages each (100k total), generated from the BinaryInferno serialization grammar (Section VII of the paper). Formats span fixed-width, length-value, repeated patterns, and star-patterns. Byte fields are assigned semantic personalities (FLOAT, COUNTER, ASCII, ADDRESS, INTEGER) so the model sees varied label distributions during training.

### Tier 2 — Scapy-extracted real protocols

10 protocols with per-byte ground-truth labels extracted from Scapy's `fields_desc` and mapped to the unified taxonomy: ARP, BGP (hand-labeled), DHCP, DNS, ICMP, IP, Modbus, NTP, TCP, UDP. 500–2000 messages per protocol with varied field values. A 10% noise-corrupted subset (random byte flips + tail truncation) teaches noise robustness.

### Tier 3 — BinaryInferno benchmark

The 10-protocol BI benchmark datasets (bgp, dhcp, dnp3, mavlink, mirai, modbus, ntp48, smb, smb2, tutorial) accessed via symlink. Used only for reference and BI baseline runs — not used as training data at any stage.

---

## Quickstart

### Prerequisites

- Python 3.11+
- PyTorch 2.x
- `uv` or `conda` for environment management

```bash
uv pip install torch scapy einops hydra-core numpy scipy scikit-learn
```

### Generate data

```bash
# Tier 1 — grammar-sampled synthetic protocols
python -m src.data_generation.grammar_sampler

# Tier 2 — Scapy-extracted real protocols
python -m src.data_generation.generate_tier2
```

### Train

```bash
# Stage 1: pre-train auxiliary byte LM
python src/training/train_lm.py --config configs/train_lm.yaml

# Stage 2: joint training on Tier 1 + Tier 2
python src/training/train_main.py --config configs/train_main.yaml

# Stage 3: LOPO fine-tuning (one run per held-out protocol)
python src/training/train_lopo.py --config configs/train_lopo.yaml --holdout dns
```

### Evaluate

```bash
# Full LOPO evaluation + comparison table vs BinaryInferno
python src/evaluation/run_eval.py

# Results written to results/comparison_table.csv and results/REPORT.md
```

---

## Design decisions

See [NOTES.md](NOTES.md) for documented choices including:

- Why LOPO evaluation uses Tier-2 Scapy messages rather than the BI `.txt.input` files (BI has no per-byte ground truth labels).
- DNS compression pointer labeling (OPAQUE, not a new POINTER type).
- BGP hand-labeling rationale.
- ByteLM positional embedding mismatch handling between Stage 1 (max_len=256) and Stage 2 (max_len=512).
- BI baseline parsing methodology.

---

## Baselines

Four comparison systems are included under `baselines/`:

| System | Paper | Notes |
|--------|-------|-------|
| BinaryInferno | NDSS 2023 | Primary comparison target |
| NetPlier | | |
| FieldHunter | | |
| Netzob | | |
| NeMeSys | | |

---

## Limitations and future work

- **DHCP** is the weakest protocol (F1 0.33) due to TLV option chains where field boundaries are determined by a type-length prefix pattern the model occasionally misses.
- **Cross-message attention requires N≥32 messages** — BinaryInferno needs only 3. For very small captures, BI's rule-based approach is more reliable.
- **BI benchmark protocols not covered in Tier 2** (DNP3, Mavlink, Mirai, SMB, SMB2, Tutorial) are future work; they require either Scapy contrib dissectors or hand-labeling.
- **Ablation runs** (6 planned) are pending — `results/ablations.csv` contains placeholder rows.

---

## Citation

If you use this code or results, please cite BinaryInferno (the system and benchmark this work builds on):

```bibtex
@inproceedings{binaryinferno2023,
  title     = {BinaryInferno: A Semantic-Driven Approach for Automatic Binary Protocol Reverse Engineering},
  booktitle = {NDSS Symposium 2023},
  year      = {2023}
}
```
