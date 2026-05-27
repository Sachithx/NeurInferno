"""
Full Phase-5 evaluation pipeline.

Runs LOPO evaluation for our model + BI baseline, writes:
  results/comparison_table.csv
  results/pr_curves/{protocol}.npz
  results/ablations.csv         (placeholder — filled by ablation runs)
  results/REPORT.md             (via report.py)

Usage:
    python -m src.evaluation.run_eval \
        --lopo_ckpt_dir checkpoints/lopo \
        --tier2_dir     data/tier2_scapy \
        --results_dir   results \
        [--skip_bi]     # skip slow BI baseline (useful for quick eval)
        [--fast_dev]    # 50 messages per protocol, skip BI
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.training.dataset import LOPO_PROTOCOLS
from src.evaluation.lopo import run_lopo_evaluation
from src.evaluation.baseline_comparison import run_bi_baselines
from src.evaluation.metrics import BoundaryMetrics


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _m_row(m: BoundaryMetrics | None, prefix: str = "") -> dict[str, str]:
    if m is None:
        return {f"{prefix}P": "N/A", f"{prefix}R": "N/A",
                f"{prefix}FPR": "N/A", f"{prefix}F1": "N/A"}
    return {
        f"{prefix}P":   f"{m.precision:.4f}",
        f"{prefix}R":   f"{m.recall:.4f}",
        f"{prefix}FPR": f"{m.fpr:.4f}",
        f"{prefix}F1":  f"{m.f1:.4f}",
    }


def write_comparison_csv(
    ours:     dict[str, BoundaryMetrics],
    bi:       dict[str, BoundaryMetrics | None],
    out_path: Path,
) -> None:
    protocols = LOPO_PROTOCOLS
    fieldnames = ["protocol",
                  "Ours P", "Ours R", "Ours FPR", "Ours F1",
                  "BI P",   "BI R",   "BI FPR",   "BI F1"]
    rows = []
    sum_ours = {"P": 0.0, "R": 0.0, "FPR": 0.0, "F1": 0.0}
    sum_bi   = {"P": 0.0, "R": 0.0, "FPR": 0.0, "F1": 0.0}
    n_ours = 0; n_bi = 0

    for proto in protocols:
        m_o = ours.get(proto)
        m_b = bi.get(proto)
        row = {"protocol": proto}
        row.update(_m_row(m_o, "Ours "))
        row.update(_m_row(m_b, "BI "))
        rows.append(row)
        if m_o:
            sum_ours["P"]   += m_o.precision; sum_ours["R"]   += m_o.recall
            sum_ours["FPR"] += m_o.fpr;       sum_ours["F1"]  += m_o.f1
            n_ours += 1
        if m_b:
            sum_bi["P"]   += m_b.precision; sum_bi["R"]   += m_b.recall
            sum_bi["FPR"] += m_b.fpr;       sum_bi["F1"]  += m_b.f1
            n_bi += 1

    # Average row
    def _avg(d, n):
        return {k: d[k] / n if n > 0 else float("nan") for k in d}

    avg_o = _avg(sum_ours, n_ours)
    avg_b = _avg(sum_bi, n_bi)
    avg_row = {"protocol": "AVERAGE"}
    avg_row.update({f"Ours {k}": f"{v:.4f}" for k, v in avg_o.items()})
    avg_row.update({f"BI {k}":   f"{v:.4f}" if not np.isnan(v) else "N/A"
                    for k, v in avg_b.items()})
    rows.append(avg_row)

    # Delta row
    delta_row = {"protocol": "DELTA (Ours - BI)"}
    for k in ["P", "R", "FPR", "F1"]:
        if n_bi > 0:
            delta_row[f"Ours {k}"] = f"{avg_o[k] - avg_b[k]:+.4f}"
        else:
            delta_row[f"Ours {k}"] = "N/A"
        delta_row[f"BI {k}"] = ""
    rows.append(delta_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved comparison table → {out_path}")


def save_pr_curves(pr_curves: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for proto, data in pr_curves.items():
        np.savez(
            out_dir / f"{proto}.npz",
            precisions=data["precisions"],
            recalls=data["recalls"],
            thresholds=data["thresholds"],
            auprc=np.array([data["auprc"]]),
        )
    print(f"Saved PR curves → {out_dir}")


def write_ablations_placeholder(out_path: Path) -> None:
    """Write empty ablation CSV with correct headers (filled by ablation runs)."""
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ablation", "avg_precision", "avg_recall", "avg_fpr", "avg_f1"]
    ablations = [
        "full_model",
        "no_cross_msg",
        "no_consistency_loss",
        "no_entropy_feat",
        "tier1_only",
        "tier2_only",
        "bio_tagging_baseline",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name in ablations:
            w.writerow({k: ("" if k != "ablation" else name) for k in fieldnames})
    print(f"Saved ablation placeholder → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_eval(
    lopo_ckpt_dir: str = "checkpoints/lopo",
    tier2_dir:     str = "data/tier2_scapy",
    results_dir:   str = "results",
    max_msgs:      int = 1000,
    bi_max_msgs:   int = 100,
    skip_bi:       bool = False,
    fast_dev:      bool = False,
) -> None:
    lopo_ckpt_dir = Path(lopo_ckpt_dir)
    tier2_dir     = Path(tier2_dir)
    results_dir   = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if fast_dev:
        max_msgs    = 50
        bi_max_msgs = 20
        skip_bi     = True

    # ── Stage 5.2: LOPO evaluation ─────────────────────────────────────────
    print("\n" + "="*60)
    print("LOPO EVALUATION (our model)")
    print("="*60)
    ours, pr_curves = run_lopo_evaluation(
        lopo_ckpt_dir=lopo_ckpt_dir,
        tier2_dir=tier2_dir,
        max_msgs=max_msgs,
        n_msgs_batch=32,
    )

    save_pr_curves(pr_curves, results_dir / "pr_curves")

    # ── Stage 5.3: BI baseline ─────────────────────────────────────────────
    bi: dict[str, BoundaryMetrics | None] = {}
    if not skip_bi:
        print("\n" + "="*60)
        print("BI BASELINE")
        print("="*60)
        bi = run_bi_baselines(tier2_dir=tier2_dir, max_msgs=bi_max_msgs)
    else:
        print("\nSkipping BI baseline (--skip_bi)")
        bi = {p: None for p in LOPO_PROTOCOLS}

    # ── Write outputs ───────────────────────────────────────────────────────
    write_comparison_csv(ours, bi, results_dir / "comparison_table.csv")
    write_ablations_placeholder(results_dir / "ablations.csv")

    # Generate REPORT.md
    from src.evaluation.report import generate_report
    generate_report(results_dir)

    # Summary print
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Protocol':<12} {'Ours F1':>8} {'BI F1':>8} {'Delta':>8}")
    print("-"*40)
    for proto in LOPO_PROTOCOLS:
        o_f1 = f"{ours[proto].f1:.3f}" if proto in ours else "  N/A"
        b_f1 = f"{bi[proto].f1:.3f}"   if bi.get(proto) else "  N/A"
        delta = ""
        if proto in ours and bi.get(proto):
            delta = f"{ours[proto].f1 - bi[proto].f1:+.3f}"
        print(f"{proto:<12} {o_f1:>8} {b_f1:>8} {delta:>8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lopo_ckpt_dir", default="checkpoints/lopo")
    p.add_argument("--tier2_dir",     default="data/tier2_scapy")
    p.add_argument("--results_dir",   default="results")
    p.add_argument("--max_msgs",      type=int, default=1000)
    p.add_argument("--bi_max_msgs",   type=int, default=100)
    p.add_argument("--skip_bi",       action="store_true")
    p.add_argument("--fast_dev",      action="store_true")
    args = p.parse_args()
    run_eval(**vars(args))
