"""
Baseline comparison: run BinaryInferno on each LOPO protocol's test messages
and compute the same metrics as our model, using the same ground truth.

All metrics use OUR metric implementation (not BI's) for apples-to-apples
comparison, as required by the brief.
"""

from __future__ import annotations

import random
from pathlib import Path

from src.training.dataset import LOPO_PROTOCOLS
from src.evaluation.ground_truth import load_labeled_messages, messages_to_hex_lines
from src.evaluation.bi_runner import run_bi, bi_boundaries_to_metrics_input
from src.evaluation.metrics import (
    BoundaryMetrics, compute_boundary_metrics, aggregate_metrics,
)

# BI is slow; limit per-protocol messages for the baseline run
BI_MAX_MSGS = 100


def run_bi_baseline(
    protocol:  str,
    tier2_dir: Path,
    max_msgs:  int  = BI_MAX_MSGS,
    endian:    str  = "BE",
    timeout:   int  = 300,
    seed:      int  = 42,
) -> BoundaryMetrics | None:
    """
    Run BI on one protocol's test messages and return aggregated metrics.
    Returns None if BI fails or produces no parseable output.
    """
    try:
        msgs = load_labeled_messages(tier2_dir, protocol, max_msgs=max_msgs, seed=seed)
    except FileNotFoundError:
        print(f"  [{protocol}] no Tier-2 data, skipping BI baseline")
        return None

    hex_lines = messages_to_hex_lines(msgs)
    gt_list   = [m.boundary_per_gap for m in msgs]

    print(f"  [{protocol}] running BI on {len(hex_lines)} messages ...")
    bi_preds = run_bi(hex_lines, endian=endian, timeout=timeout)

    preds, gt = bi_boundaries_to_metrics_input(bi_preds, gt_list)
    if not preds:
        print(f"  [{protocol}] BI produced no parseable predictions")
        return None

    per_msg = [compute_boundary_metrics(p, g) for p, g in zip(preds, gt)]
    agg = aggregate_metrics(per_msg)
    coverage = len(preds) / len(msgs) * 100
    print(f"  [{protocol}] BI  P={agg.precision:.3f}  R={agg.recall:.3f}"
          f"  FPR={agg.fpr:.3f}  F1={agg.f1:.3f}  (coverage {coverage:.0f}%)")
    return agg


def run_bi_baselines(
    tier2_dir: Path,
    protocols: list[str] | None = None,
    max_msgs:  int  = BI_MAX_MSGS,
    endian:    str  = "BE",
    timeout:   int  = 300,
) -> dict[str, BoundaryMetrics | None]:
    """Run BI baseline for all LOPO protocols."""
    if protocols is None:
        protocols = LOPO_PROTOCOLS

    results: dict[str, BoundaryMetrics | None] = {}
    for proto in protocols:
        print(f"\n=== BI baseline: {proto} ===")
        results[proto] = run_bi_baseline(
            proto, tier2_dir, max_msgs=max_msgs, endian=endian, timeout=timeout)
    return results
