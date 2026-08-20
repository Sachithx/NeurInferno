---
title: NeurInferno
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.29.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Infer field boundaries in unlabeled binary protocol messages
---

# NeurInferno

Paste **several hex messages of the same format** (one per line). The model
reads the batch together and uses cross-message statistics at each byte
offset to predict **field boundaries** (cuts), not field names.

A single packet is the wrong input. Do not paste production traffic that
contains secrets; inputs are processed in memory.

This Space runs on **CPU**. Weights are the LOPO modbus hold-out checkpoint
(~15 MB), trained on the other eleven protocols plus grammar data.

Examples: ARP and IGMP (use the buttons under the text box).
