"""
Generate results/REPORT.md from evaluation outputs.

Reads:
  results/comparison_table.csv
  results/pr_curves/{protocol}.npz
  results/ablations.csv
  results/failure_analysis.json  (optional, written by failure_analysis.py)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _format_table(rows: list[dict], cols: list[str]) -> str:
    """Render a list of dicts as a Markdown table."""
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for row in rows:
        cells = " | ".join(str(row.get(c, "")) for c in cols)
        lines.append(f"| {cells} |")
    return "\n".join(lines)


def _pr_curve_section(pr_dir: Path, protocols: list[str]) -> str:
    lines = ["## 3. PR Curves\n"]
    for proto in protocols:
        path = pr_dir / f"{proto}.npz"
        if not path.exists():
            continue
        data   = np.load(path)
        auprc  = float(data["auprc"][0])
        prec   = data["precisions"]
        rec    = data["recalls"]
        # Find operating point closest to threshold=0.5 (index 50 of 100)
        idx50  = min(50, len(prec) - 1)
        lines.append(f"**{proto}**: AUPRC={auprc:.3f}, "
                     f"P@0.5≈{prec[idx50]:.3f}, R@0.5≈{rec[idx50]:.3f}")
    return "\n".join(lines)


def _failure_section(fa_path: Path | None) -> str:
    if fa_path is None or not fa_path.exists():
        return "## 4. Failure-Mode Analysis\n\n_Not yet generated._\n"

    fa = json.loads(fa_path.read_text())
    lines = ["## 4. Failure-Mode Analysis\n"]
    for proto, data in fa.items():
        lines.append(f"\n### {proto}\n")
        fps = data.get("false_positives", [])
        fns = data.get("false_negatives", [])
        if fps:
            lines.append("**False Positives (sample):**")
            for item in fps[:5]:
                lines.append(f"- {item}")
        if fns:
            lines.append("\n**False Negatives (sample):**")
            for item in fns[:5]:
                lines.append(f"- {item}")
    return "\n".join(lines)


def generate_report(results_dir: Path = Path("results")) -> None:
    comp_path = results_dir / "comparison_table.csv"
    abla_path = results_dir / "ablations.csv"
    pr_dir    = results_dir / "pr_curves"
    fa_path   = results_dir / "failure_analysis.json"

    sections = []

    # Header
    sections.append("# Binary Protocol Field Inference — Evaluation Report\n")

    # 1. Comparison table
    sections.append("## 1. Per-Protocol Comparison: Ours vs BinaryInferno\n")
    if comp_path.exists():
        rows = _read_csv(comp_path)
        cols = ["protocol", "Ours P", "Ours R", "Ours FPR", "Ours F1",
                "BI P", "BI R", "BI FPR", "BI F1"]
        sections.append(_format_table(rows, cols))
        sections.append("\n")
    else:
        sections.append("_Not yet generated._\n")

    # 2. Ablation table
    sections.append("## 2. Ablation Study\n")
    if abla_path.exists():
        rows = _read_csv(abla_path)
        non_empty = [r for r in rows if any(v for k, v in r.items() if k != "ablation")]
        if non_empty:
            cols = ["ablation", "avg_precision", "avg_recall", "avg_fpr", "avg_f1"]
            sections.append(_format_table(rows, cols))
        else:
            sections.append("_Ablation runs pending — rows are placeholders._\n")
        sections.append("\n")
    else:
        sections.append("_Not yet generated._\n")

    # 3. PR curves
    protocols = [
        "bgp_raw", "dhcp", "modbus", "ntp", "dns",
        "tcp", "udp", "ip", "icmp", "arp",
    ]
    sections.append(_pr_curve_section(pr_dir, protocols))
    sections.append("\n")

    # 4. Failure analysis
    sections.append(_failure_section(fa_path))
    sections.append("\n")

    # 5. Honest assessment
    sections.append("""## 5. Honest Assessment

### Where the DL approach helps
- **Variable-length fields**: cross-message statistics allow the model to learn
  that position X always has a boundary because types change there across messages
  of the same format — something rule-based systems miss when fields vary.
- **Entropy as a feature**: ByteLM conditional entropy gives useful signal at
  field transitions (high entropy = opaque payload, low = structured header).

### Where it matches or loses to BI
- **Short, fixed-width protocols** (e.g., NTP, ARP): BI's deterministic rules
  are extremely effective and the DL model adds little beyond them.
- **Variable-endian formats**: BI explicitly searches both BE and LE; our model
  learns endianness from training data distribution.
- **Small sample size per test run**: the cross-message attention requires N≥32
  messages to produce meaningful statistics; BI needs only 3.
""")

    report = "\n".join(sections)
    out_path = results_dir / "REPORT.md"
    out_path.write_text(report)
    print(f"Saved report → {out_path}")


if __name__ == "__main__":
    generate_report()
