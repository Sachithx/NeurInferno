# Dataset directory

The complete protocol and synthetic datasets are distributed through Hugging
Face and are intentionally not tracked in this Git repository.

Download them into this directory before training or evaluation:

```bash
hf download sachithabey/neurinferno \
  --repo-type dataset \
  --local-dir data
```

The resulting layout is:

```text
data/
├── protocols/
│   └── <protocol>/messages.jsonl
└── grammar/
    └── <format>/
        ├── format_spec.json
        └── messages.jsonl
```

Do not commit downloaded datasets, private captures, or generated traffic.
Small synthetic fixtures needed by tests should live under `tests/fixtures/`.
