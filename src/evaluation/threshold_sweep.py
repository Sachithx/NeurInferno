"""
Threshold sweep on BI benchmark data.

Runs model inference exactly once per protocol, then evaluates at every
threshold in the grid — no redundant forward passes.

Outputs
-------
results/threshold_sweep.csv   — protocol × threshold × P/R/FPR/F1
results/threshold_best.csv    — best-F1 threshold per protocol + clean-6 avg

Usage
-----
    python -m src.evaluation.threshold_sweep \\
        --main_ckpt checkpoints/main/best.ckpt \\
        --bench_dir data/tier3_benchmark/top-level/1000 \\
        --results_dir results \\
        [--n_thresholds 19]       # 0.05 … 0.95 in equal steps (default)
        [--thresholds 0.3 0.4 0.5 0.6]   # explicit list overrides n_thresholds
        [--clean_only]
"""

from __future__ import annotations

import argparse
import csv
import math
import numpy as np
from pathlib import Path

import torch

from src.evaluation.benchmark_eval import (
    CLEAN_PROTOCOLS, ALL_PROTOCOLS,
    load_benchmark_messages,
    run_inference_scores,
    scores_to_pred_sets,
    evaluate_protocol,
)
from src.training.train_main import FieldInferenceModule


# ── Sweep core ────────────────────────────────────────────────────────────────

def sweep_protocol(
    protocol:   str,
    messages:   list[bytes],
    all_scores: list[list[float]],
    thresholds: list[float],
) -> list[dict]:
    """Return a list of metric dicts, one per threshold."""
    rows = []
    for t in thresholds:
        pred_sets = scores_to_pred_sets(all_scores, t)
        m = evaluate_protocol(protocol, messages, pred_sets)
        rows.append({
            "protocol":  protocol,
            "threshold": round(t, 4),
            "precision": m["precision"],
            "recall":    m["recall"],
            "fpr":       m["fpr"],
            "f1":        m["f1"],
        })
    return rows


# ── I/O ───────────────────────────────────────────────────────────────────────

def _write_sweep_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["protocol", "threshold", "precision", "recall", "fpr", "f1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames,
                           extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                        for k, v in row.items()})
    print(f"Saved → {path}")


def _write_best_csv(best: dict[str, dict], path: Path) -> None:
    fieldnames = ["protocol", "leakage_free", "best_threshold",
                  "precision", "recall", "fpr", "f1"]
    rows = []
    for proto, m in best.items():
        rows.append({
            "protocol":       proto,
            "leakage_free":   "yes" if proto in CLEAN_PROTOCOLS else "no",
            "best_threshold": f"{m['threshold']:.2f}",
            "precision":      f"{m['precision']:.4f}",
            "recall":         f"{m['recall']:.4f}",
            "fpr":            f"{m['fpr']:.4f}",
            "f1":             f"{m['f1']:.4f}",
        })

    clean = [m for p, m in best.items() if p in CLEAN_PROTOCOLS]
    if clean:
        rows.append({
            "protocol":       "AVERAGE (clean-6)",
            "leakage_free":   "yes",
            "best_threshold": "—",
            "precision":      f"{sum(m['precision'] for m in clean)/len(clean):.4f}",
            "recall":         f"{sum(m['recall']    for m in clean)/len(clean):.4f}",
            "fpr":            f"{sum(m['fpr']       for m in clean)/len(clean):.4f}",
            "f1":             f"{sum(m['f1']        for m in clean)/len(clean):.4f}",
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {path}")


def _print_best_table(best: dict[str, dict]) -> None:
    print(f"\n{'Protocol':<12} {'Leak-free':>9} {'T*':>5} "
          f"{'P':>7} {'R':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 57)
    for proto, m in best.items():
        tag = "yes" if proto in CLEAN_PROTOCOLS else "no "
        print(f"{proto:<12} {tag:>9} {m['threshold']:5.2f} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} "
              f"{m['fpr']:7.3f} {m['f1']:7.3f}")

    clean = [m for p, m in best.items() if p in CLEAN_PROTOCOLS]
    if clean:
        print("-" * 57)
        n = len(clean)
        print(f"{'avg (clean-6)':<12} {'yes':>9} {'—':>5} "
              f"{sum(m['precision'] for m in clean)/n:7.3f} "
              f"{sum(m['recall']    for m in clean)/n:7.3f} "
              f"{sum(m['fpr']       for m in clean)/n:7.3f} "
              f"{sum(m['f1']        for m in clean)/n:7.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_threshold_sweep(
    main_ckpt:    str,
    bench_dir:    str   = "data/tier3_benchmark/top-level/1000",
    results_dir:  str   = "results",
    clean_only:   bool  = False,
    n_thresholds: int   = 19,
    thresholds:   list[float] | None = None,
    n_msgs_batch: int   = 32,
    max_len:      int   = 512,
) -> None:
    bench_dir   = Path(bench_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        # evenly spaced from 0.05 to 0.95
        thresholds = [round(v, 4) for v in
                      np.linspace(0.05, 0.95, n_thresholds).tolist()]

    protocols = CLEAN_PROTOCOLS if clean_only else ALL_PROTOCOLS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {main_ckpt}")
    model = FieldInferenceModule.load_from_checkpoint(str(main_ckpt), strict=True)
    model.eval()
    model.to(device)

    all_rows:  list[dict]       = []
    best_per_proto: dict[str, dict] = {}

    for proto in protocols:
        print(f"\n[{proto}]", end="  ")
        try:
            messages = load_benchmark_messages(bench_dir, proto)
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            continue

        print(f"{len(messages)} msgs  ", end="", flush=True)

        scores = run_inference_scores(
            model, messages,
            n_msgs=n_msgs_batch, max_len=max_len, device=device,
        )

        rows = sweep_protocol(proto, messages, scores, thresholds)
        all_rows.extend(rows)

        best = max(rows, key=lambda r: r["f1"])
        best_per_proto[proto] = best
        print(f"best T={best['threshold']:.2f}  "
              f"P={best['precision']:.3f}  R={best['recall']:.3f}  "
              f"F1={best['f1']:.3f}")

    _write_sweep_csv(all_rows, results_dir / "threshold_sweep.csv")
    _write_best_csv(best_per_proto, results_dir / "threshold_best.csv")
    _print_best_table(best_per_proto)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_ckpt",    required=True)
    p.add_argument("--bench_dir",    default="data/tier3_benchmark/top-level/1000")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--clean_only",   action="store_true")
    p.add_argument("--n_thresholds", type=int, default=19,
                   help="Number of thresholds evenly spaced in [0.05, 0.95]")
    p.add_argument("--thresholds",   type=float, nargs="+", default=None,
                   help="Explicit threshold list (overrides --n_thresholds)")
    p.add_argument("--n_msgs_batch", type=int, default=32)
    p.add_argument("--max_len",      type=int, default=512)
    args = p.parse_args()
    run_threshold_sweep(**vars(args))
