#!/usr/bin/env python3
"""
run_netzob_eval_tier2.py

Evaluates Netzob's field inference (splitStatic + splitAligned) on
each tier2_scapy protocol against boundary_per_gap ground truth.

Usage:
    /home/AD/sachith/.conda/envs/bini/bin/python run_netzob_eval_tier2.py \
        [--n_msgs N] [--method static|aligned|both] [--protocols proto1 ...]

Output:
    Per-protocol P/R/F1 table + netzob_eval_summary.json
"""

import argparse
import binascii
import json
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
TIER2_DIR  = Path("/home/AD/sachith/BI_NI/data/tier2_scapy")
OUTPUT_DIR = Path("/home/AD/sachith/BI_NI/netzob_eval_results")

ALL_PROTOCOLS = sorted([
    p.name for p in TIER2_DIR.iterdir()
    if p.is_dir() and (p / "messages.jsonl").exists()
])

# ── Netzob import ─────────────────────────────────────────────────────────────
try:
    from netzob.all import Symbol, RawMessage, Format
except ImportError as e:
    print(f"ERROR: Cannot import netzob: {e}")
    print("Install with: pip install -e /home/AD/sachith/BI_NI/netzob "
          "  (plus: bitarray, bintrees, pythoncrc, netaddr, getmac, pylstar)")
    sys.exit(1)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(proto: str, n: int) -> list[dict]:
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


def boundary_per_gap_to_set(boundary_per_gap: list) -> set:
    """0-indexed gap list → 1-indexed boundary position set."""
    return {i + 1 for i, v in enumerate(boundary_per_gap) if v == 1}

# ── Boundary extraction from Netzob Symbol ────────────────────────────────────

def symbol_to_boundaries(symbol) -> set[int]:
    """
    Given a Netzob Symbol after inference, extract predicted field boundary
    positions (1-indexed: boundary at position i means new field starts at byte i).

    Handles:
    - Fixed-size fields: exact byte width known
    - Variable-size or zero-size fields: stop fixed tracking, add boundary and stop
    """
    boundaries = set()
    offset = 0

    for i, field in enumerate(symbol.fields):
        try:
            vals = field.getValues()
        except Exception:
            break

        # Drop empty-value fields (alignment artifacts from splitAligned)
        sizes = [len(v) for v in vals if v is not None]
        if not sizes:
            continue

        min_size = min(sizes)
        max_size = max(sizes)

        # Mark boundary before this field (if not the first)
        if i > 0 and offset > 0:
            boundaries.add(offset)

        if min_size == 0:
            # Zero-size gap: alignment artifact, skip without advancing offset
            continue

        if min_size == max_size:
            # Fixed-width field
            offset += min_size
        else:
            # Variable-width: advance by minimum, then we can't predict further
            offset += min_size
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

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, fpr=fpr, f1=f1)


def evaluate(predicted: set, records: list[dict]) -> dict:
    all_m = []
    for rec in records:
        gt = boundary_per_gap_to_set(rec["boundary_per_gap"])
        msg_len = len(rec["bytes_hex"]) // 2
        all_m.append(boundary_metrics(predicted, gt, msg_len))

    n = len(all_m)
    return {k: sum(m[k] for m in all_m) / n
            for k in ["precision", "recall", "fpr", "f1", "tp", "fp", "fn", "tn"]}


# ── Single protocol runner ────────────────────────────────────────────────────

def run_protocol(proto: str, records: list[dict], method: str) -> dict:
    messages = [RawMessage(data=binascii.unhexlify(r["bytes_hex"])) for r in records]
    symbol   = Symbol(messages=messages)

    t0 = time.time()
    try:
        if method == "static":
            Format.splitStatic(symbol)
        elif method == "aligned":
            Format.splitAligned(symbol, useSemantic=False)
        else:
            raise ValueError(f"Unknown method: {method}")
        elapsed = time.time() - t0
        status  = "ok"
    except Exception as e:
        elapsed = time.time() - t0
        return {"status": "error", "error": str(e), "elapsed_s": round(elapsed, 1),
                "method": method}

    predicted = symbol_to_boundaries(symbol)
    metrics   = evaluate(predicted, records)

    return {
        "status":   "ok",
        "method":   method,
        "n_msgs":   len(records),
        "n_fields": len(symbol.fields),
        "n_predicted": len(predicted),
        "predicted_boundaries": sorted(predicted),
        "elapsed_s": round(elapsed, 1),
        **{k: round(metrics[k], 4) for k in ["precision", "recall", "f1", "fpr",
                                               "tp", "fp", "fn", "tn"]},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_msgs",    type=int, default=100)
    parser.add_argument("--method",    default="both",
                        choices=["static", "aligned", "both"])
    parser.add_argument("--protocols", nargs="+", default=ALL_PROTOCOLS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = ["static", "aligned"] if args.method == "both" else [args.method]

    # Load existing summary for merge
    summary_path = OUTPUT_DIR / "netzob_eval_summary.json"
    if summary_path.exists():
        try:
            results = json.loads(summary_path.read_text())
        except Exception:
            results = {}
    else:
        results = {}

    hdr = "{:<12} {:<9} {:>7} {:>7} {:>7} {:>7}  {}"
    sep = "-" * 65
    print(hdr.format("Protocol", "Method", "P", "R", "F1", "FPR", "Notes"))
    print(sep)

    for proto in args.protocols:
        if not (TIER2_DIR / proto / "messages.jsonl").exists():
            print(f"{proto:<12} -- skipped (no data)")
            continue

        records = load_records(proto, args.n_msgs)

        for method in methods:
            key = f"{proto}_{method}"
            res = run_protocol(proto, records, method)
            results[key] = res

            if res["status"] == "ok":
                note = f"ok ({res['elapsed_s']:.0f}s, {res['n_predicted']} boundaries, {res['n_fields']} fields)"
                print(hdr.format(proto, method,
                    res["precision"], res["recall"], res["f1"], res["fpr"], note))
            else:
                print(hdr.format(proto, method, "N/A", "N/A", "N/A", "N/A",
                    f"ERROR: {res.get('error','?')[:60]}"))

    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Final summary table
    ok_static  = {k.replace("_static",""):  v for k, v in results.items()
                  if k.endswith("_static")  and v.get("status") == "ok"}
    ok_aligned = {k.replace("_aligned", ""): v for k, v in results.items()
                  if k.endswith("_aligned") and v.get("status") == "ok"}

    if ok_static or ok_aligned:
        print("\n=== Macro Averages ===")
        def macro(d):
            if not d: return None
            n = len(d)
            return {k: round(sum(v[k] for v in d.values())/n, 4)
                    for k in ["precision","recall","f1","fpr"]}

        ms = macro(ok_static)
        ma = macro(ok_aligned)
        if ms:
            print(f"splitStatic  ({len(ok_static):2} protocols): "
                  f"P={ms['precision']:.4f}  R={ms['recall']:.4f}  "
                  f"F1={ms['f1']:.4f}  FPR={ms['fpr']:.4f}")
        if ma:
            print(f"splitAligned ({len(ok_aligned):2} protocols): "
                  f"P={ma['precision']:.4f}  R={ma['recall']:.4f}  "
                  f"F1={ma['f1']:.4f}  FPR={ma['fpr']:.4f}")


if __name__ == "__main__":
    main()
