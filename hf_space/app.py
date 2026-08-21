"""Interactive CPU demo for NeurInferno."""

from __future__ import annotations

import html
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import gradio as gr

from neurinferno import FieldBoundaryModel, MessageResult, parse_hex

MAX_LEN = 512
MAX_MSGS = 32
MIN_RECOMMENDED_MSGS = 4
DEFAULT_THRESHOLD = 0.75
ROOT = Path(__file__).resolve().parent
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

FIELD_COLORS = (
    "#2563eb",
    "#16a34a",
    "#d97706",
    "#db2777",
    "#7c3aed",
    "#0891b2",
    "#ea580c",
    "#475569",
)

EMPTY_METRICS = """
<div class="ni-metrics">
  <div class="ni-metric"><strong>—</strong><span>messages</span></div>
  <div class="ni-metric"><strong>—</strong><span>byte length</span></div>
  <div class="ni-metric"><strong>—</strong><span>fields found</span></div>
  <div class="ni-metric"><strong>—</strong><span>runtime</span></div>
</div>
"""

PLACEHOLDER = """
<div class="ni-empty">
  <div class="ni-empty-icon">⌁</div>
  <strong>Your boundary map will appear here</strong>
  <span>Load an example or paste several same-format messages, then select Analyze.</span>
</div>
"""

MODEL: FieldBoundaryModel | None = None


@dataclass(frozen=True)
class ParsedBatch:
    """Validated messages and their input line numbers."""

    messages: list[bytes]
    line_numbers: list[int]


def get_model() -> FieldBoundaryModel:
    """Load and cache the CPU model on first use."""

    global MODEL
    if MODEL is None:
        if not CKPT.exists():
            raise FileNotFoundError("The model checkpoint is not available in this Space.")
        MODEL = FieldBoundaryModel.from_checkpoint(CKPT, device="cpu")
    return MODEL


def parse_messages(text: str) -> ParsedBatch:
    """Parse one message per non-comment line and report all input errors."""

    messages: list[bytes] = []
    line_numbers: list[int] = []
    errors: list[str] = []

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            message = parse_hex(line)
        except ValueError as exc:
            errors.append(f"Line {line_number}: {exc}")
            continue
        messages.append(message)
        line_numbers.append(line_number)

    if errors:
        detail = "; ".join(errors[:5])
        if len(errors) > 5:
            detail += f"; and {len(errors) - 5} more"
        raise ValueError(detail)
    if not messages:
        raise ValueError("Paste at least one hex message, with one message per line.")
    if len(messages) > MAX_MSGS:
        raise ValueError(f"Received {len(messages)} messages; the maximum is {MAX_MSGS}.")
    return ParsedBatch(messages=messages, line_numbers=line_numbers)


def status_html(message: str, kind: str = "info") -> str:
    """Create an escaped inline status banner."""

    return f'<div class="ni-status ni-{kind}">{html.escape(message)}</div>'


def render_metrics(results: list[MessageResult], elapsed_ms: float) -> str:
    lengths = [result.n_bytes for result in results]
    length_label = str(lengths[0]) if len(set(lengths)) == 1 else f"{min(lengths)}–{max(lengths)}"
    field_count = sum(len(result.segments) for result in results)
    return f"""
<div class="ni-metrics">
  <div class="ni-metric"><strong>{len(results)}</strong><span>messages</span></div>
  <div class="ni-metric"><strong>{length_label}</strong><span>bytes / message</span></div>
  <div class="ni-metric"><strong>{field_count}</strong><span>fields found</span></div>
  <div class="ni-metric"><strong>{elapsed_ms:.0f} ms</strong><span>runtime</span></div>
</div>
"""


def render_byte_map(results: list[MessageResult]) -> str:
    """Render explicit field groups with byte offsets."""

    cards: list[str] = []
    for message_index, result in enumerate(results, start=1):
        groups: list[str] = []
        for field_index, segment in enumerate(result.segments, start=1):
            color = FIELD_COLORS[(field_index - 1) % len(FIELD_COLORS)]
            byte_cells = "".join(
                (
                    f'<span class="ni-byte" title="byte offset {offset}" '
                    f'aria-label="offset {offset}, value {value:02x}">{value:02x}</span>'
                )
                for offset, value in enumerate(bytes.fromhex(segment.hex), start=segment.start)
            )
            groups.append(
                f"""
<div class="ni-field" style="--field-color:{color}">
  <div class="ni-field-label">F{field_index} <span>[{segment.start}:{segment.end}]</span></div>
  <div class="ni-byte-row">{byte_cells}</div>
</div>
"""
            )

        truncated = ' <span class="ni-badge">truncated</span>' if result.truncated else ""
        cards.append(
            f"""
<section class="ni-message">
  <header><strong>Message {message_index}</strong><span>{result.n_bytes} bytes · {len(result.segments)} fields{truncated}</span></header>
  <div class="ni-fields">{"".join(groups)}</div>
</section>
"""
        )
    return '<div class="ni-map">' + "".join(cards) + "</div>"


def segment_rows(results: list[MessageResult]) -> list[list[str | int]]:
    """Build rows for the structured segment table."""

    rows: list[list[str | int]] = []
    for message_index, result in enumerate(results, start=1):
        for field_index, segment in enumerate(result.segments, start=1):
            rows.append(
                [
                    message_index,
                    field_index,
                    segment.start,
                    segment.end,
                    segment.end - segment.start,
                    segment.hex,
                ]
            )
    return rows


def render_confidence(results: list[MessageResult], threshold: float) -> str:
    """Render compact per-gap confidence bars."""

    rows: list[str] = []
    for message_index, result in enumerate(results, start=1):
        bars = "".join(
            (
                f'<span class="ni-score {"ni-selected" if score >= threshold else ""}" '
                f'style="height:{max(3, round(score * 72))}px" '
                f'title="after byte {gap_index}: {score:.3f}" '
                f'aria-label="boundary confidence after byte {gap_index}: {score:.3f}"></span>'
            )
            for gap_index, score in enumerate(result.scores)
        )
        rows.append(
            f"""
<div class="ni-confidence-row">
  <div class="ni-confidence-label">M{message_index}</div>
  <div class="ni-score-strip">{bars or '<span class="ni-no-gaps">No byte gaps</span>'}</div>
</div>
"""
        )
    return f"""
<div class="ni-confidence">
  <div class="ni-legend"><span class="ni-dot"></span> selected at threshold {threshold:.2f}</div>
  {"".join(rows)}
</div>
"""


def result_payload(results: list[MessageResult], threshold: float) -> dict:
    """Return a serializable result payload."""

    return {
        "threshold": threshold,
        "message_count": len(results),
        "messages": [asdict(result) for result in results],
    }


def analyze(text: str, threshold: float):
    """Validate input, run inference, and build all interface outputs."""

    try:
        batch = parse_messages(text)
        started = perf_counter()
        results = get_model().infer(
            batch.messages,
            threshold=threshold,
            max_msgs=MAX_MSGS,
        )
        elapsed_ms = (perf_counter() - started) * 1000
    except (FileNotFoundError, ValueError) as exc:
        return (
            status_html(str(exc), "error"),
            EMPTY_METRICS,
            PLACEHOLDER,
            [],
            PLACEHOLDER,
            {},
        )
    except Exception:
        logging.exception("NeurInferno inference failed")
        return (
            status_html("Inference could not be completed. Please retry.", "error"),
            EMPTY_METRICS,
            PLACEHOLDER,
            [],
            PLACEHOLDER,
            {},
        )

    notices: list[str] = []
    if len(batch.messages) < MIN_RECOMMENDED_MSGS:
        notices.append(f"For stronger results, provide at least {MIN_RECOMMENDED_MSGS} messages.")
    truncated_count = sum(result.truncated for result in results)
    if truncated_count:
        notices.append(f"{truncated_count} message(s) exceeded {MAX_LEN} bytes and were truncated.")
    message = "Analysis complete."
    kind = "success"
    if notices:
        message += " " + " ".join(notices)
        kind = "warning"

    return (
        status_html(message, kind),
        render_metrics(results, elapsed_ms),
        render_byte_map(results),
        segment_rows(results),
        render_confidence(results, threshold),
        result_payload(results, threshold),
    )


def clear_outputs():
    """Restore the interface to its initial state."""

    return (
        "",
        DEFAULT_THRESHOLD,
        status_html("Ready for input."),
        EMPTY_METRICS,
        PLACEHOLDER,
        [],
        PLACEHOLDER,
        {},
    )


CSS = """
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; }
.ni-hero { padding: 1.4rem 0 .65rem; }
.ni-brand { display:flex; align-items:center; gap:.85rem; }
.ni-mark { display:grid; place-items:center; width:46px; height:46px; border-radius:14px;
  color:white; font-size:24px; background:linear-gradient(145deg,#f97316,#dc2626);
  box-shadow:0 10px 24px rgba(234,88,12,.22); }
.ni-hero h1 { margin:0; font-size:clamp(1.8rem,4vw,2.7rem); letter-spacing:-.04em; }
.ni-hero p { margin:.55rem 0 .8rem; max-width:760px; color:var(--body-text-color-subdued); }
.ni-links { display:flex; gap:.55rem; flex-wrap:wrap; }
.ni-links a { padding:.35rem .68rem; border:1px solid var(--border-color-primary);
  border-radius:999px; color:var(--body-text-color); text-decoration:none !important; font-size:.86rem; }
.ni-links a:hover { border-color:#f97316; color:#ea580c; }
.ni-panel { border:1px solid var(--border-color-primary) !important; border-radius:18px !important;
  padding:8px !important; box-shadow:0 12px 34px rgba(15,23,42,.055) !important; }
.ni-help { font-size:.92rem; color:var(--body-text-color-subdued); }
.ni-help ol { padding-left:1.25rem; }
.ni-help li { margin:.45rem 0; }
.ni-status { padding:.78rem .9rem; margin:.55rem 0; border-radius:12px; font-size:.9rem; }
.ni-info { color:#1e40af; background:#eff6ff; border:1px solid #bfdbfe; }
.ni-success { color:#166534; background:#f0fdf4; border:1px solid #bbf7d0; }
.ni-warning { color:#92400e; background:#fffbeb; border:1px solid #fde68a; }
.ni-error { color:#991b1b; background:#fef2f2; border:1px solid #fecaca; }
.ni-metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; margin:.45rem 0 1rem; }
.ni-metric { padding:.9rem 1rem; border:1px solid var(--border-color-primary); border-radius:14px;
  background:var(--block-background-fill); }
.ni-metric strong { display:block; font-size:1.25rem; letter-spacing:-.025em; }
.ni-metric span { color:var(--body-text-color-subdued); font-size:.78rem; }
.ni-empty { min-height:260px; display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; gap:.4rem; color:var(--body-text-color-subdued); border:1px dashed var(--border-color-primary);
  border-radius:14px; }
.ni-empty-icon { font-size:2rem; color:#f97316; }
.ni-map { display:flex; flex-direction:column; gap:.8rem; }
.ni-message { border:1px solid var(--border-color-primary); border-radius:14px; overflow:hidden;
  background:var(--block-background-fill); }
.ni-message header { display:flex; justify-content:space-between; gap:1rem; padding:.7rem .85rem;
  border-bottom:1px solid var(--border-color-primary); font-size:.85rem; }
.ni-message header span { color:var(--body-text-color-subdued); }
.ni-fields { display:flex; gap:.55rem; padding:.85rem; overflow-x:auto; align-items:flex-start; }
.ni-field { flex:0 0 auto; border:1px solid var(--border-color-primary); border-top:3px solid var(--field-color);
  border-radius:9px; overflow:hidden; background:var(--background-fill-secondary); }
.ni-field-label { padding:.28rem .48rem; font:600 .7rem/1.2 ui-sans-serif,system-ui,sans-serif;
  color:var(--body-text-color); border-bottom:1px solid var(--border-color-primary); }
.ni-field-label span { color:var(--body-text-color-subdued); font-weight:500; }
.ni-byte-row { display:flex; padding:.3rem; }
.ni-byte { display:inline-grid; place-items:center; width:27px; height:27px; border-radius:5px;
  font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--body-text-color);
  background:var(--block-background-fill); border:1px solid var(--border-color-primary); margin-right:2px; }
.ni-badge { padding:.12rem .4rem; margin-left:.2rem; border-radius:999px; color:#92400e !important;
  background:#fef3c7; font-size:.68rem; }
.ni-confidence { display:flex; flex-direction:column; gap:.65rem; }
.ni-legend { color:var(--body-text-color-subdued); font-size:.8rem; }
.ni-dot { display:inline-block; width:9px; height:9px; margin-right:.25rem; border-radius:3px; background:#f97316; }
.ni-confidence-row { display:grid; grid-template-columns:38px minmax(0,1fr); gap:.5rem; align-items:end; }
.ni-confidence-label { font:600 .75rem ui-monospace,monospace; color:var(--body-text-color-subdued); }
.ni-score-strip { height:78px; display:flex; align-items:flex-end; gap:2px; overflow-x:auto;
  padding:.2rem .25rem; border-bottom:1px solid var(--border-color-primary); }
.ni-score { display:block; flex:0 0 5px; min-height:3px; background:#94a3b8; border-radius:2px 2px 0 0; }
.ni-score.ni-selected { background:#f97316; }
.ni-no-gaps { align-self:center; color:var(--body-text-color-subdued); font-size:.8rem; }
.ni-footnote { text-align:center; color:var(--body-text-color-subdued); font-size:.78rem; padding:1.2rem 0 .4rem; }
@media (max-width: 760px) {
  .ni-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .ni-message header { flex-direction:column; gap:.2rem; }
}
"""

THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="slate")

with gr.Blocks(title="NeurInferno", theme=THEME, css=CSS, fill_width=True) as demo:
    gr.HTML(
        """
<div class="ni-hero">
  <div class="ni-brand"><div class="ni-mark">⌁</div><h1>NeurInferno</h1></div>
  <p>Discover likely field boundaries in batches of unlabeled binary protocol messages.</p>
  <div class="ni-links">
    <a href="https://github.com/Sachithx/NeurInferno" target="_blank">GitHub ↗</a>
    <a href="https://pypi.org/project/neurinferno/" target="_blank">PyPI ↗</a>
    <a href="https://huggingface.co/sachithabey/neurinferno" target="_blank">Model ↗</a>
    <a href="https://huggingface.co/datasets/sachithabey/neurinferno" target="_blank">Dataset ↗</a>
  </div>
</div>
"""
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=7, elem_classes=["ni-panel"]):
            message_input = gr.Textbox(
                label="Hex messages",
                info="One same-format message per line. Whitespace, 0x, colons, dashes, and underscores are accepted.",
                lines=12,
                max_lines=24,
                placeholder="0001080006040001…\n0001080006040002…",
            )
            with gr.Accordion("Advanced settings", open=False):
                threshold_input = gr.Slider(
                    0.05,
                    0.95,
                    value=DEFAULT_THRESHOLD,
                    step=0.05,
                    label="Boundary threshold",
                    info="Higher values produce fewer, more conservative boundaries.",
                )
            with gr.Row():
                analyze_button = gr.Button("Analyze messages", variant="primary", scale=3)
                clear_button = gr.Button("Clear", scale=1)

        with gr.Column(scale=4, elem_classes=["ni-panel"]):
            gr.Markdown(
                f"""
### How to use it

1. Provide **4–{MAX_MSGS} messages** from the same message format.
2. Keep headers and payload bytes exactly as captured.
3. Select **Analyze messages** and inspect the proposed fields.

The model processes at most **{MAX_LEN} bytes per message** and predicts boundaries, not semantic field names.

<span class="ni-help">Do not submit credentials, tokens, or sensitive production traffic.</span>
"""
            )
            gr.Examples(
                examples=[[EXAMPLE_ARP, DEFAULT_THRESHOLD], [EXAMPLE_IGMP, DEFAULT_THRESHOLD]],
                inputs=[message_input, threshold_input],
                label="Try an example",
            )

    status_output = gr.HTML(status_html("Ready for input."))
    metrics_output = gr.HTML(EMPTY_METRICS)

    with gr.Tabs():
        with gr.Tab("Boundary map"):
            map_output = gr.HTML(PLACEHOLDER)
        with gr.Tab("Segments"):
            table_output = gr.Dataframe(
                headers=["Message", "Field", "Start", "End", "Length", "Hex"],
                datatype=["number", "number", "number", "number", "number", "str"],
                interactive=False,
                label="Predicted segments",
            )
        with gr.Tab("Confidence"):
            gr.Markdown(
                "Each bar is one byte gap. Orange bars meet the selected boundary threshold."
            )
            confidence_output = gr.HTML(PLACEHOLDER)
        with gr.Tab("JSON"):
            json_output = gr.JSON(label="Machine-readable results", value={})

    gr.HTML(
        '<div class="ni-footnote">CPU demo · Inputs are processed for inference and should not contain sensitive data.</div>'
    )

    inference_outputs = [
        status_output,
        metrics_output,
        map_output,
        table_output,
        confidence_output,
        json_output,
    ]
    analyze_button.click(
        analyze,
        inputs=[message_input, threshold_input],
        outputs=inference_outputs,
        api_name="infer",
    )
    clear_button.click(
        clear_outputs,
        inputs=[],
        outputs=[message_input, threshold_input, *inference_outputs],
    )

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch()
