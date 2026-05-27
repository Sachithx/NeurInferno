#!/usr/bin/env python3
"""
run_bi_eval_tier2.py

Runs BinaryInferno (blackboard.py) on each tier2_scapy protocol,
parses the SPECSTART/SPECEND output to get predicted field boundaries,
and evaluates against the boundary_per_gap ground truth.

Usage:
    python run_bi_eval_tier2.py [--n_msgs N] [--protocols proto1 proto2 ...]

Output:
    Per-protocol P/R/F1 table and a summary JSON.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
TIER2_DIR  = Path("/home/AD/sachith/BI_NI/data/tier2_scapy")
BI_DIR     = Path("/home/AD/sachith/binaryinferno/binaryinferno")
OUTPUT_DIR = Path("/home/AD/sachith/BI_NI/bi_eval_results")

ALL_PROTOCOLS = sorted([
    p.name for p in TIER2_DIR.iterdir() if p.is_dir()
    and (p / "messages.jsonl").exists()
])

# Endianness hints from tier2 data (use first message's endianness field)
def get_endianness(proto_dir: Path) -> str:
    with open(proto_dir / "messages.jsonl") as f:
        d = json.loads(f.readline())
    return d.get("endianness", "big")


# ── Ground truth parsing ───────────────────────────────────────────────────────

def boundary_per_gap_to_set(boundary_per_gap: list) -> set:
    """Convert boundary_per_gap list (0-indexed gaps) to boundary set (1-indexed positions)."""
    return {i + 1 for i, v in enumerate(boundary_per_gap) if v == 1}


def load_messages(proto: str, n: int) -> list[dict]:
    path = TIER2_DIR / proto / "messages.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= n:
                break
    return records


# ── BinaryInferno runner ───────────────────────────────────────────────────────

def make_bi_input(hex_messages: list[str]) -> str:
    return "?\n--\n" + "\n".join(hex_messages) + "\n--\n"


def run_blackboard(input_txt: str, endian: str, timeout: int = 300) -> str | None:
    """Run blackboard.py, return stdout or None on timeout/failure."""
    endian_flag = "--detectors boundBE" if endian == "big" else "--detectors boundLE"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(input_txt)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, "blackboard.py", "--filename", tmp_path, "--sigmaonly"],
            cwd=str(BI_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT after {timeout}s]")
        return None
    except Exception as e:
        print(f"  [ERROR: {e}]")
        return None
    finally:
        os.unlink(tmp_path)


# ── SPECSTART parser ───────────────────────────────────────────────────────────

def parse_specstart(stdout: str) -> set[int] | None:
    """
    Parse SPECSTART...SPECEND block and return predicted boundary positions
    (1-indexed byte positions where a new field begins).

    Returns None if no SPECSTART block is found.
    """
    m = re.search(r"SPECSTART\n(.*?)\nSPECEND", stdout, re.DOTALL)
    if not m:
        return None

    spec_lines = [l.strip() for l in m.group(1).strip().split("\n") if l.strip()]

    boundaries = set()
    offset = 0

    for line in spec_lines:
        if offset > 0:
            boundaries.add(offset)  # new field starts at this byte position

        # FieldFixed NV or Length NV → fixed width field
        fixed_m = re.match(r"(?:FieldFixed|Length)\s+(\d+)V", line)
        if fixed_m:
            offset += int(fixed_m.group(1))
            continue

        # FieldVar *V or FieldRep → variable/repeated; we track only up to here
        if re.match(r"FieldVar|FieldRep", line):
            break  # can't predict further fixed offsets

        # Unknown line – try to extract a width anyway
        unk_m = re.match(r"\S+\s+(\d+)V", line)
        if unk_m:
            offset += int(unk_m.group(1))
            continue

        # Give up on unknown formats
        break

    return boundaries


# ── Metrics ───────────────────────────────────────────────────────────────────

def boundary_metrics(detected: set, gt: set, msg_len: int) -> dict:
    possible = set(range(1, msg_len))
    det = detected & possible
    gt  = gt & possible

    tp = len(det & gt)
    fp = len(det - gt)
    fn = len(gt - det)
    tn = len(possible - gt - det)

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if fn == 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1     = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, fpr=fpr, f1=f1)


def evaluate_protocol(proto: str, predicted: set, records: list[dict]) -> dict:
    all_m = []
    for rec in records:
        gt = boundary_per_gap_to_set(rec["boundary_per_gap"])
        msg_len = len(rec["bytes_hex"]) // 2
        all_m.append(boundary_metrics(predicted, gt, msg_len))

    n = len(all_m)
    avg = {}
    for k in ["precision", "recall", "fpr", "f1", "tp", "fp", "fn", "tn"]:
        avg[k] = sum(m[k] for m in all_m) / n
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_msgs", type=int, default=100,
                        help="Max messages per protocol (default 100)")
    parser.add_argument("--protocols", nargs="+", default=ALL_PROTOCOLS,
                        help="Protocols to evaluate")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-protocol BI timeout in seconds")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    header = f"{'Protocol':<15} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7}  Status"
    print(header)
    print("-" * len(header))

    for proto in args.protocols:
        proto_dir = TIER2_DIR / proto
        if not (proto_dir / "messages.jsonl").exists():
            print(f"{proto:<15} -- skipped (no data)")
            continue

        records = load_messages(proto, args.n_msgs)
        endian  = get_endianness(proto_dir)
        hex_msgs = [r["bytes_hex"] for r in records]

        print(f"{proto:<15} running BI ({len(records)} msgs, {endian})...", flush=True)
        t0 = time.time()

        bi_input  = make_bi_input(hex_msgs)
        stdout    = run_blackboard(bi_input, endian, timeout=args.timeout)
        elapsed   = time.time() - t0

        if stdout is None:
            print(f"{proto:<15} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}  TIMEOUT/ERROR")
            results[proto] = {"status": "timeout", "elapsed_s": round(elapsed, 1)}
            # Save raw output if any
            continue

        # Save raw BI output
        raw_path = OUTPUT_DIR / f"{proto}_bi_output.txt"
        raw_path.write_text(stdout)

        predicted = parse_specstart(stdout)

        if predicted is None:
            print(f"{proto:<15} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>7}  NO_SPEC")
            results[proto] = {"status": "no_spec", "elapsed_s": round(elapsed, 1)}
            continue

        metrics = evaluate_protocol(proto, predicted, records)
        results[proto] = {
            "status":          "ok",
            "n_msgs":          len(records),
            "endianness":      endian,
            "predicted_boundaries": sorted(predicted),
            "n_predicted":     len(predicted),
            "elapsed_s":       round(elapsed, 1),
            **{k: round(metrics[k], 4) for k in ["precision", "recall", "f1", "fpr",
                                                    "tp", "fp", "fn", "tn"]},
        }

        print(
            f"{proto:<15} {metrics['precision']:>7.4f} {metrics['recall']:>7.4f} "
            f"{metrics['f1']:>7.4f} {metrics['fpr']:>7.4f}  "
            f"ok ({elapsed:.0f}s, {len(predicted)} boundaries)"
        )

    # Save summary — merge into existing file so partial runs don't clobber each other
    summary_path = OUTPUT_DIR / "bi_eval_summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
        except Exception:
            existing = {}
        existing.update(results)
        results = existing
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Print final table
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    if ok:
        print("\n=== Final Summary ===")
        avg_f1 = sum(v["f1"] for v in ok.values()) / len(ok)
        avg_pr = sum(v["precision"] for v in ok.values()) / len(ok)
        avg_re = sum(v["recall"] for v in ok.values()) / len(ok)
        print(f"Evaluated {len(ok)}/{len(results)} protocols successfully")
        print(f"Macro-avg  P={avg_pr:.4f}  R={avg_re:.4f}  F1={avg_f1:.4f}")


if __name__ == "__main__":
    main()
