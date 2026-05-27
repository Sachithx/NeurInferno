"""
Run BinaryInferno on a set of messages and parse its boundary predictions.

BI takes hex messages on stdin (one per line) and outputs an INFERRED
DESCRIPTION section where each message is shown with boundaries as
space / '|' separators:

    01 000D | 60A67AED 054150504C45

Each space or '|' delimiter between hex groups is a field boundary.
We convert these visual delimiters to binary boundary_per_gap arrays.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


_BI_DIR = Path(__file__).resolve().parent.parent.parent / "binaryinferno" / "binaryinferno"


def _parse_bi_message_line(line: str) -> list[int] | None:
    """
    Parse one message line from BI's INFERRED DESCRIPTION output.

    Example: '01 000D | 60A67AED 054150504C45'
    Returns binary boundary_per_gap list, or None if line is not a hex message.
    """
    line = line.strip()
    if not line or line.startswith("-") or line.startswith("?"):
        return None

    # Split on '|' and whitespace to get hex token groups
    tokens = re.split(r'\s*\|\s*|\s+', line)
    tokens = [t for t in tokens if re.fullmatch(r'[0-9A-Fa-f]+', t)]
    if not tokens:
        return None
    # Each token must have even hex length
    if any(len(t) % 2 != 0 for t in tokens):
        return None

    # Build boundary_per_gap
    n_bytes = sum(len(t) // 2 for t in tokens)
    if n_bytes < 2:
        return None

    boundaries = [0] * (n_bytes - 1)
    pos = 0
    for i, tok in enumerate(tokens):
        field_bytes = len(tok) // 2
        pos += field_bytes
        if i < len(tokens) - 1:
            boundaries[pos - 1] = 1  # boundary after last byte of this field

    return boundaries


def _parse_bi_output(output: str, n_messages: int) -> list[list[int] | None]:
    """
    Parse BI stdout and extract per-message boundary predictions.

    Returns a list of boundary_per_gap (or None for unparseable lines),
    attempting to align with input message order.
    """
    # Find the INFERRED DESCRIPTION block
    in_block = False
    results: list[list[int] | None] = []

    for line in output.splitlines():
        if "INFERRED DESCRIPTION" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if line.strip().startswith("--") or line.strip() == "":
            continue
        if "QTY SAMPLES" in line or "HEADER ONLY" in line or "SPECSTART" in line:
            break
        # Skip the header line (first non-empty non-'--' line contains letters like L, T, R)
        if re.search(r'[LTQROltrqo\?]{2}', line):
            continue

        parsed = _parse_bi_message_line(line)
        if parsed is not None:
            results.append(parsed)

    # If we got results for exactly the right number of messages, great.
    # Otherwise pad / trim to n_messages.
    if len(results) < n_messages:
        results += [None] * (n_messages - len(results))
    return results[:n_messages]


def run_bi(
    hex_messages: list[str],
    endian:       str  = "BE",
    timeout:      int  = 300,
    bi_dir:       Path | None = None,
) -> list[list[int] | None]:
    """
    Run BinaryInferno on a list of hex messages.

    Parameters
    ----------
    hex_messages : list of uppercase hex strings (one message per entry)
    endian       : "BE", "LE", or "BOTH"
    timeout      : seconds to allow BI to run
    bi_dir       : path to binaryinferno/ package directory

    Returns
    -------
    List of boundary_per_gap arrays aligned to input messages.
    None entries = BI couldn't produce a boundary for that message.
    """
    if bi_dir is None:
        bi_dir = _BI_DIR

    if not (bi_dir / "blackboard.py").exists():
        raise FileNotFoundError(f"BI blackboard.py not found at {bi_dir}")

    stdin_data = "\n".join(hex_messages) + "\n"

    cmd = [
        sys.executable, "blackboard.py",
        "--detectors", endian,
    ]
    try:
        result = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(bi_dir),
        )
    except subprocess.TimeoutExpired:
        print(f"  BI timed out after {timeout}s")
        return [None] * len(hex_messages)
    except Exception as e:
        print(f"  BI failed: {e}")
        return [None] * len(hex_messages)

    return _parse_bi_output(result.stdout + result.stderr, len(hex_messages))


def bi_boundaries_to_metrics_input(
    bi_preds:  list[list[int] | None],
    gt_list:   list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Filter out None predictions and align with ground truth.

    Returns (filtered_preds, filtered_gt) where lengths match.
    """
    preds_out, gt_out = [], []
    for pred, gt in zip(bi_preds, gt_list):
        if pred is None:
            continue
        # Trim/pad to gt length
        n = len(gt)
        if len(pred) >= n:
            pred = pred[:n]
        else:
            pred = pred + [0] * (n - len(pred))
        preds_out.append(pred)
        gt_out.append(gt)
    return preds_out, gt_out
