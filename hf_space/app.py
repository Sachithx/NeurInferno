"""CPU demo: paste same-format hex messages, get field-boundary cuts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neurinferno import FieldBoundaryModel

MAX_LEN = 512
MAX_MSGS = 32
CKPT = ROOT / "weights" / "model.ckpt"

EXAMPLE_ARP = """0001080006040001900c2d9bfa4649e7160700000000000043f03612
0001080006040002fbccad5c9fb1d014735252376fd2446375217d01
0001080006040002a388be2d1d684df0f31908ae1acad25b4dca7aa8
0001080006040001d41cd6668e87ac1e6d7b0000000000008593a2d9
000108000604000150a4a23b295cd9a9e204000000000000a0a543ed
0001080006040002adbde39796886e640dc8b94913e6a7dc56e051b3
0001080006040001900c2d9bfa4649e7160700000000000043f03612
0001080006040002fbccad5c9fb1d014735252376fd2446375217d01"""

EXAMPLE_IGMP = """4500001c00010000010236a6a27e00bae00000011783b717e04d5117
4500001c0001000001023da3284f73ece000000117195f27e0fca8c2
4500001c000100000102c5d695797e8ee0000001227e7b74e0eb8121
4500001c0001000001025324c2e5c3d4e000000111f28787e0b485d1
4500001c0001000001024b681b9072e6e000000117454971e0bfbe89
4500001c0001000001027670d3189055e000000111232238e023ec80
4500001c00010000010236a6a27e00bae00000011783b717e04d5117
4500001c0001000001023da3284f73ece000000117195f27e0fca8c2"""


MODEL = None


def get_model() -> FieldBoundaryModel:
    global MODEL
    if MODEL is None:
        if not CKPT.exists():
            raise FileNotFoundError(
                f"Checkpoint missing at {CKPT}. Deploy should copy weights/model.ckpt."
            )
        MODEL = FieldBoundaryModel.from_checkpoint(CKPT, device="cpu")
    return MODEL


def parse_messages(text: str) -> list[bytes]:
    msgs: list[bytes] = []
    errors: list[str] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        hex_only = re.sub(r"(?i)0x", "", line)
        hex_only = re.sub(r"[^0-9a-fA-F]", "", hex_only)
        if len(hex_only) < 2:
            errors.append(f"line {i}: empty")
            continue
        if len(hex_only) % 2:
            errors.append(f"line {i}: odd number of hex digits")
            continue
        msgs.append(bytes.fromhex(hex_only))
        if len(msgs) >= MAX_MSGS:
            break
    if errors and not msgs:
        raise ValueError("; ".join(errors[:5]))
    if not msgs:
        raise ValueError("Paste at least one hex message (one per line).")
    return msgs


FIELD_SHADES = [
    "#93c5fd",
    "#86efac",
    "#fcd34d",
    "#f9a8d4",
    "#c4b5fd",
    "#67e8f9",
    "#fdba74",
    "#d4d4d8",
]


def field_index_per_byte(n: int, cuts: list[bool]) -> list[int]:
    idx = []
    k = 0
    for i in range(n):
        idx.append(k)
        if i < n - 1 and cuts[i]:
            k += 1
    return idx


def render_html(msgs: list[bytes], cuts: list[list[bool]], note: str) -> str:
    blocks = [f"<p>{note}</p>"]
    for mi, msg in enumerate(msgs):
        n = min(len(msg), MAX_LEN)
        fidx = field_index_per_byte(n, cuts[mi])
        cells = []
        for i in range(n):
            is_cut = i < n - 1 and cuts[mi][i]
            fill = FIELD_SHADES[fidx[i] % len(FIELD_SHADES)]
            border = "3px solid #111827" if is_cut else "1px solid rgba(0,0,0,0.08)"
            hx = format(msg[i], "02x")
            cells.append(
                '<span title="offset %d · field %d%s" style="'
                "display:inline-block;width:28px;height:28px;line-height:28px;"
                "text-align:center;font:11px/28px ui-monospace,monospace;"
                "margin:0;background:%s;border-right:%s;"
                'color:#111;">%s</span>'
                % (
                    i,
                    fidx[i] + 1,
                    " — cut after this byte" if is_cut else "",
                    fill,
                    border,
                    hx,
                )
            )
        blocks.append(
            f"<div style='margin:10px 0 16px;'>"
            f"<div style='font:12px sans-serif;color:#6b7280;'>message {mi + 1} "
            f"({n} bytes, {fidx[-1] + 1 if n else 0} fields)</div>{''.join(cells)}</div>"
        )
    return "<div>" + "".join(blocks) + "</div>"


def infer(text: str, threshold: float):
    try:
        msgs = parse_messages(text)
    except ValueError as e:
        return f"<p style='color:#b91c1c'>{e}</p>", ""
    note = (
        f"{len(msgs)} message(s), threshold={threshold:.2f}. "
        "Each color band is one predicted field (colors cycle; they are not names). "
        "A dark bar is the cut between fields."
    )
    if len(msgs) < 4:
        note += (
            " Warning: the model is built for a batch of same-format messages. "
            "Results with 1–3 lines are weaker."
        )
    results = get_model().infer(msgs, threshold=threshold, max_msgs=MAX_MSGS)
    cuts = [r.cuts for r in results]
    summary_lines = []
    for i, r in enumerate(results):
        parts = [f"[{s.start}:{s.end}] {s.hex}" for s in r.segments]
        summary_lines.append(f"m{i + 1}: " + " | ".join(parts))
    return render_html(msgs[: len(results)], cuts, note), "\n".join(summary_lines)


with gr.Blocks(title="NeurInferno") as demo:
    gr.Markdown(
        """
# NeurInferno
Infer **field boundaries** in unlabeled binary protocol messages.

The model predicts **where fields start and end**. Consecutive fields are
shown as alternating color bands. Colors are only visual grouping, **not**
field names.

Paste **several messages of the same format**, one hex line each
(spaces and `0x` are ok). The model looks across the batch at each offset.
Do not paste secrets.
"""
    )
    inp = gr.Textbox(
        label="Hex messages (one per line)",
        lines=10,
        placeholder="0001080006040001...\n0001080006040002...",
    )
    thr = gr.Slider(0.05, 0.95, value=0.75, step=0.05, label="Boundary threshold")
    btn = gr.Button("Infer boundaries", variant="primary")
    vis = gr.HTML(label="Byte map")
    out = gr.Textbox(label="Predicted segments (byte offsets, hex)", lines=8)
    btn.click(infer, inputs=[inp, thr], outputs=[vis, out])
    gr.Examples(
        examples=[[EXAMPLE_ARP, 0.75], [EXAMPLE_IGMP, 0.75]],
        inputs=[inp, thr],
        label="Examples (ARP, IGMP)",
    )


if __name__ == "__main__":
    demo.launch()
