---
title: NeurInferno
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Infer field boundaries in unlabeled binary protocol messages
models:
  - sachithabey/neurinferno
datasets:
  - sachithabey/neurinferno
---

# NeurInferno

NeurInferno proposes field boundaries in batches of unlabeled binary protocol
messages. Paste several messages of the same format, one hexadecimal message
per line, and inspect the boundary map, structured segments, confidence view,
or JSON output.

## Input guidance

- Use 4–32 messages from the same message format.
- Each message may contain up to 512 processed bytes.
- Whitespace, `0x`, colons, dashes, and underscores are accepted.
- The output contains boundary predictions, not semantic field names.
- Do not submit credentials, tokens, or sensitive production traffic.

This Space runs on CPU. The first request may take longer while the model is
loaded. Example batches are available directly in the interface.

- [Source](https://github.com/Sachithx/NeurInferno)
- [Python package](https://pypi.org/project/neurinferno/)
- [Model](https://huggingface.co/sachithabey/neurinferno)
- [Dataset](https://huggingface.co/datasets/sachithabey/neurinferno)
