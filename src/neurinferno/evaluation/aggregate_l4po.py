"""
Aggregate L4PO results across multiple seed runs.

Reads all l4po_*.csv files from results_dir, computes:
  - Per-protocol mean ± std across seeds it appeared in
  - Overall mean ± std across all (seed, protocol) pairs
  - Summary CSV + printed table

Usage:
    python -m neurinferno.evaluation.aggregate_l4po --results_dir results/l4po
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def load_seed_results(results_dir: Path) -> list[dict]:
    """
    Read every l4po_*.csv in results_dir.
    Returns a flat list of dicts with keys:
      seed_label, protocol, precision, recall, fpr, f1
    """
    records = []
    pattern = re.compile(r"^l4po_(.+)\.csv$")
    for csv_path in sorted(results_dir.glob("l4po_*.csv")):
        m = pattern.match(csv_path.name)
        if not m:
            continue
        label = m.group(1)  # e.g. "arp_igmp_ip_tcp"
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["protocol"].startswith("AVERAGE"):
                    continue
                try:
                    records.append(
                        {
                            "seed_label": label,
                            "protocol": row["protocol"],
                            "precision": float(row["precision"]),
                            "recall": float(row["recall"]),
                            "fpr": float(row["fpr"]),
                            "f1": float(row["f1"]),
                            "n_messages": int(row["n_messages"]),
                        }
                    )
                except (ValueError, KeyError):
                    continue
    return records


def aggregate(records: list[dict]) -> dict:
    """
    Returns:
      per_proto : {proto: {metric: (mean, std, n)}}
      overall   : {metric: (mean, std, n)}
    """
    per_proto: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    all_vals: dict[str, list] = defaultdict(list)

    for r in records:
        p = r["protocol"]
        for metric in ("precision", "recall", "fpr", "f1"):
            per_proto[p][metric].append(r[metric])
            all_vals[metric].append(r[metric])

    def _stats(vals):
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        return mean, std, n

    per_proto_stats = {
        p: {m: _stats(vals) for m, vals in metrics.items()} for p, metrics in per_proto.items()
    }
    overall_stats = {m: _stats(vals) for m, vals in all_vals.items()}
    return per_proto_stats, overall_stats


def write_summary_csv(
    per_proto_stats: dict,
    overall_stats: dict,
    out_path: Path,
) -> None:
    fieldnames = [
        "protocol",
        "n_seeds",
        "P_mean",
        "P_std",
        "R_mean",
        "R_std",
        "FPR_mean",
        "FPR_std",
        "F1_mean",
        "F1_std",
    ]
    rows = []
    for proto in sorted(per_proto_stats):
        s = per_proto_stats[proto]
        rows.append(
            {
                "protocol": proto,
                "n_seeds": s["f1"][2],
                "P_mean": f"{s['precision'][0]:.4f}",
                "P_std": f"{s['precision'][1]:.4f}",
                "R_mean": f"{s['recall'][0]:.4f}",
                "R_std": f"{s['recall'][1]:.4f}",
                "FPR_mean": f"{s['fpr'][0]:.4f}",
                "FPR_std": f"{s['fpr'][1]:.4f}",
                "F1_mean": f"{s['f1'][0]:.4f}",
                "F1_std": f"{s['f1'][1]:.4f}",
            }
        )
    # Overall row
    o = overall_stats
    rows.append(
        {
            "protocol": "OVERALL",
            "n_seeds": o["f1"][2],
            "P_mean": f"{o['precision'][0]:.4f}",
            "P_std": f"{o['precision'][1]:.4f}",
            "R_mean": f"{o['recall'][0]:.4f}",
            "R_std": f"{o['recall'][1]:.4f}",
            "FPR_mean": f"{o['fpr'][0]:.4f}",
            "FPR_std": f"{o['fpr'][1]:.4f}",
            "F1_mean": f"{o['f1'][0]:.4f}",
            "F1_std": f"{o['f1'][1]:.4f}",
        }
    )
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {out_path}")


def print_table(per_proto_stats: dict, overall_stats: dict) -> None:
    print(f"\n{'Protocol':<12} {'Seeds':>5}  {'P':>13}  {'R':>13}  {'FPR':>13}  {'F1':>13}")
    print("-" * 72)
    for proto in sorted(per_proto_stats):
        stats = per_proto_stats[proto]
        n = stats["f1"][2]
        formatted = {
            metric: f"{stats[metric][0]:.3f}±{stats[metric][1]:.3f}"
            for metric in ("precision", "recall", "fpr", "f1")
        }
        print(
            f"{proto:<12} {n:>5}  "
            f"{formatted['precision']:>13}  {formatted['recall']:>13}  "
            f"{formatted['fpr']:>13}  {formatted['f1']:>13}"
        )
    print("-" * 72)
    o = overall_stats

    def ofmt(m):
        return f"{o[m][0]:.3f}±{o[m][1]:.3f}"

    print(
        f"{'OVERALL':<12} {o['f1'][2]:>5}  "
        f"{ofmt('precision'):>13}  {ofmt('recall'):>13}  "
        f"{ofmt('fpr'):>13}  {ofmt('f1'):>13}"
    )


def print_seed_breakdown(records: list[dict]) -> None:
    print(f"\n{'Seed label':<35} {'Proto':<10} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 65)
    current = None
    for r in records:
        if r["seed_label"] != current:
            current = r["seed_label"]
            print(f"  {current}")
        print(
            f"    {'':31} {r['protocol']:<10} "
            f"{r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/l4po")
    p.add_argument(
        "--out_csv", default=None, help="Output CSV path (default: results_dir/l4po_aggregate.csv)"
    )
    p.add_argument("--verbose", action="store_true", help="Print per-seed breakdown")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_csv = Path(args.out_csv) if args.out_csv else results_dir / "l4po_aggregate.csv"

    records = load_seed_results(results_dir)
    if not records:
        print(f"No l4po_*.csv files found in {results_dir}")
        raise SystemExit(1)

    print(f"Loaded {len(records)} protocol results from {results_dir}")

    if args.verbose:
        print_seed_breakdown(records)

    per_proto_stats, overall_stats = aggregate(records)
    print_table(per_proto_stats, overall_stats)
    write_summary_csv(per_proto_stats, overall_stats, out_csv)
