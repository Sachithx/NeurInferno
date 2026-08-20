"""
Unified evaluation pipeline: LOPO + L4PO.

Scans checkpoint directories and evaluates all found checkpoints:
  LOPO : checkpoints/seed789/lopo/<protocol>/  →  results/seed789/lopo_results.csv
  L4PO : checkpoints/seed789/l4po/<label>/     →  results/seed789/l4po/l4po_<label>.csv
                                               →  results/seed789/l4po/l4po_aggregate.csv

Usage:
    python -m neurinferno.evaluation.run_eval \
        --ckpt_root   checkpoints/seed789 \
        --tier2_dir   data/protocols \
        --results_dir results/seed789 \
        --threshold   0.75 \
        --max_msgs    1000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from neurinferno.training.dataset import LOPO_PROTOCOLS
from neurinferno.training.train_main import FieldInferenceModule
from neurinferno.evaluation.lopo import run_lopo_evaluation, run_model_inference, _find_best_ckpt
from neurinferno.evaluation.ground_truth import load_labeled_messages
from neurinferno.evaluation.metrics import (
    BoundaryMetrics, compute_boundary_metrics, aggregate_metrics,
)

# Threshold values used for the sensitivity sweep (saved alongside main results)
_TAU_SWEEP = np.round(np.linspace(0.05, 0.95, 50), 3).tolist()


def _preferred_ckpt_name(label: str) -> str | None:
    """Filename recorded in results/reference/l4po/l4po_<label>.csv, if any."""
    ref = Path("results/reference/l4po") / f"l4po_{label}.csv"
    if not ref.exists():
        return None
    with ref.open() as f:
        for row in csv.DictReader(f):
            ck = (row.get("checkpoint") or "").strip()
            if ck:
                return Path(ck).name
    return None


def _label_to_held_out(label: str) -> list[str]:
    """Recover held-out protocol list from a sorted-joined label string.

    e.g. 'arp_bgp_raw_igmp_ip' → ['arp', 'bgp_raw', 'igmp', 'ip']
    """
    remaining = label
    found = []
    for proto in sorted(LOPO_PROTOCOLS):
        if remaining == proto or remaining.startswith(proto + "_"):
            found.append(proto)
            remaining = remaining[len(proto):]
            if remaining.startswith("_"):
                remaining = remaining[1:]
    if remaining:
        raise ValueError(
            f"Could not fully parse label '{label}' — leftover: '{remaining}'. "
            f"Known protocols: {sorted(LOPO_PROTOCOLS)}"
        )
    return found


# ── LOPO evaluation ───────────────────────────────────────────────────────────

def _sweep_from_pr_curves(pr_curves: dict) -> list[dict]:
    """Compute mean P/R/F1 at each tau from in-memory per-protocol PR curve data."""
    curve = []
    for tau in _TAU_SWEEP:
        ps, rs, f1s = [], [], []
        for proto, data in pr_curves.items():
            thresholds = np.asarray(data["thresholds"])
            precisions = np.asarray(data["precisions"])
            recalls    = np.asarray(data["recalls"])
            idx = int(np.argmin(np.abs(thresholds - tau)))
            p  = float(precisions[idx])
            r  = float(recalls[idx])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            ps.append(p); rs.append(r); f1s.append(f1)
        if ps:
            curve.append({
                "tau": round(float(tau), 3),
                "mean_P":  round(float(np.mean(ps)),  4),
                "mean_R":  round(float(np.mean(rs)),  4),
                "mean_F1": round(float(np.mean(f1s)), 4),
            })
    return curve


def eval_lopo(
    lopo_ckpt_dir: Path,
    tier2_dir:     Path,
    results_dir:   Path,
    threshold:     float,
    max_msgs:      int,
    device:        str,
) -> dict[str, BoundaryMetrics]:
    print("\n" + "=" * 60)
    print("LOPO EVALUATION")
    print("=" * 60)

    ours, pr_curves = run_lopo_evaluation(
        lopo_ckpt_dir=lopo_ckpt_dir,
        tier2_dir=tier2_dir,
        max_msgs=max_msgs,
        n_msgs_batch=32,
        threshold=threshold,
        device=device,
    )

    # Save PR curves
    pr_dir = results_dir / "pr_curves"
    pr_dir.mkdir(parents=True, exist_ok=True)
    for proto, data in pr_curves.items():
        np.savez(
            pr_dir / f"{proto}.npz",
            precisions=data["precisions"],
            recalls=data["recalls"],
            thresholds=data["thresholds"],
            auprc=np.array([data["auprc"]]),
        )

    # Compute and save tau sweep (no extra inference — derived from PR curves)
    tau_sweep = _sweep_from_pr_curves(pr_curves)
    if tau_sweep:
        with open(results_dir / "lopo_tau_sweep.json", "w") as f:
            json.dump({"curve": tau_sweep}, f, indent=2)

    # Write results CSV
    fieldnames = ["protocol", "P", "R", "FPR", "F1"]
    rows = []
    total = {"P": 0.0, "R": 0.0, "FPR": 0.0, "F1": 0.0}
    n = 0
    for proto in LOPO_PROTOCOLS:
        m = ours.get(proto)
        if m is None:
            rows.append({"protocol": proto, "P": "N/A", "R": "N/A",
                         "FPR": "N/A", "F1": "N/A"})
            continue
        rows.append({"protocol": proto,
                     "P": f"{m.precision:.4f}", "R": f"{m.recall:.4f}",
                     "FPR": f"{m.fpr:.4f}",     "F1": f"{m.f1:.4f}"})
        total["P"] += m.precision; total["R"] += m.recall
        total["FPR"] += m.fpr;     total["F1"] += m.f1
        n += 1
    if n:
        rows.append({"protocol": "AVERAGE",
                     "P": f"{total['P']/n:.4f}", "R": f"{total['R']/n:.4f}",
                     "FPR": f"{total['FPR']/n:.4f}", "F1": f"{total['F1']/n:.4f}"})

    out_path = results_dir / "lopo_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved LOPO results → {out_path}")

    return ours


# ── L4PO evaluation ───────────────────────────────────────────────────────────

def _eval_one_l4po_split(
    ckpt_path:   Path,
    held_out:    list[str],
    tier2_dir:   Path,
    results_dir: Path,
    threshold:   float,
    max_msgs:    int,
    device:      str,
) -> tuple[dict[str, dict], list[dict]]:
    """Evaluate one L4PO checkpoint. Returns (per-proto results, tau sweep)."""
    model = FieldInferenceModule.load_from_checkpoint(str(ckpt_path), strict=True)
    model.eval()
    model.to(device)

    # Collect raw scores for all held-out protocols (used for tau sweep)
    proto_scores: dict[str, tuple[list, list]] = {}

    all_results: dict[str, dict] = {}
    for proto in held_out:
        print(f"  [{proto}]", end="  ")
        try:
            msgs = load_labeled_messages(tier2_dir, proto, max_msgs=max_msgs)
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            continue
        print(f"{len(msgs)} msgs", end="  ")
        scores_list, gt_list = run_model_inference(
            model, msgs, n_msgs=32, max_len=512, device=device,
        )
        proto_scores[proto] = (scores_list, gt_list)
        per_msg = [
            compute_boundary_metrics(
                [1 if s >= threshold else 0 for s in sc], gt
            )
            for sc, gt in zip(scores_list, gt_list)
        ]
        agg = aggregate_metrics(per_msg)
        all_results[proto] = {
            "precision": agg.precision, "recall": agg.recall,
            "fpr": agg.fpr,             "f1": agg.f1,
            "n_messages": len(msgs),
        }
        print(f"P={agg.precision:.3f}  R={agg.recall:.3f}  "
              f"FPR={agg.fpr:.3f}  F1={agg.f1:.3f}")

    # Tau sweep: re-threshold in-memory (no extra model loading)
    tau_sweep: list[dict] = []
    if proto_scores:
        for tau in _TAU_SWEEP:
            ps, rs, f1s = [], [], []
            for proto, (sc_list, gt_list) in proto_scores.items():
                per_msg = [
                    compute_boundary_metrics(
                        [1 if s >= tau else 0 for s in sc], gt
                    )
                    for sc, gt in zip(sc_list, gt_list)
                ]
                agg = aggregate_metrics(per_msg)
                ps.append(agg.precision); rs.append(agg.recall); f1s.append(agg.f1)
            tau_sweep.append({
                "tau":    round(float(tau), 3),
                "mean_P":  round(float(np.mean(ps)),  4),
                "mean_R":  round(float(np.mean(rs)),  4),
                "mean_F1": round(float(np.mean(f1s)), 4),
            })

    # Write per-split results CSV
    run_label = "_".join(sorted(held_out))
    csv_path  = results_dir / f"l4po_{run_label}.csv"
    data_rows = [
        {"protocol": proto,
         "precision": f"{m['precision']:.4f}", "recall": f"{m['recall']:.4f}",
         "fpr": f"{m['fpr']:.4f}", "f1": f"{m['f1']:.4f}",
         "n_messages": int(m["n_messages"]), "checkpoint": Path(ckpt_path).name}
        for proto, m in all_results.items()
    ]
    if data_rows:
        avgs = {k: sum(float(r[k]) for r in data_rows) / len(data_rows)
                for k in ("precision", "recall", "fpr", "f1")}
        data_rows.append({
            "protocol": f"AVERAGE ({len(all_results)} protocols)",
            "precision": f"{avgs['precision']:.4f}", "recall": f"{avgs['recall']:.4f}",
            "fpr": f"{avgs['fpr']:.4f}", "f1": f"{avgs['f1']:.4f}",
            "n_messages": sum(int(r["n_messages"]) for r in data_rows[:-1]),
            "checkpoint": "",
        })
    fieldnames = ["protocol", "precision", "recall", "fpr", "f1",
                  "n_messages", "checkpoint"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(data_rows)
    print(f"  Saved → {csv_path}")

    return all_results, tau_sweep


def eval_l4po(
    l4po_ckpt_dir: Path,
    tier2_dir:     Path,
    results_dir:   Path,
    threshold:     float,
    max_msgs:      int,
    device:        str,
) -> None:
    print("\n" + "=" * 60)
    print("L4PO EVALUATION")
    print("=" * 60)

    if not l4po_ckpt_dir.is_dir():
        print(f"No L4PO checkpoint directory {l4po_ckpt_dir} — skipping.")
        return

    split_dirs = sorted(
        d for d in l4po_ckpt_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not split_dirs:
        print("No L4PO checkpoint directories found — skipping.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    all_tau_sweeps: list[list[dict]] = []

    for split_dir in split_dirs:
        label = split_dir.name

        try:
            held_out = _label_to_held_out(label)
        except ValueError:
            continue  # not a split label (e.g. logs/, mains/)

        ckpt = _find_best_ckpt(
            split_dir, preferred_name=_preferred_ckpt_name(label),
        )
        if ckpt is None:
            print(f"\n[{label}] No checkpoint found — skipping.")
            continue

        print(f"\n[{label}]  held-out: {held_out}")
        _, tau_sweep = _eval_one_l4po_split(
            ckpt_path=ckpt,
            held_out=held_out,
            tier2_dir=tier2_dir,
            results_dir=results_dir,
            threshold=threshold,
            max_msgs=max_msgs,
            device=device,
        )
        if tau_sweep:
            all_tau_sweeps.append(tau_sweep)

    # Aggregate L4PO tau sweep across splits
    if all_tau_sweeps:
        agg_sweep = []
        for i, tau_val in enumerate(_TAU_SWEEP):
            ps  = [s[i]["mean_P"]  for s in all_tau_sweeps if i < len(s)]
            rs  = [s[i]["mean_R"]  for s in all_tau_sweeps if i < len(s)]
            f1s = [s[i]["mean_F1"] for s in all_tau_sweeps if i < len(s)]
            agg_sweep.append({
                "tau":    round(float(tau_val), 3),
                "mean_P":  round(float(np.mean(ps)),  4) if ps  else 0.0,
                "mean_R":  round(float(np.mean(rs)),  4) if rs  else 0.0,
                "mean_F1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            })
        with open(results_dir / "l4po_tau_sweep.json", "w") as f:
            json.dump({"curve": agg_sweep}, f, indent=2)

    # Aggregate per-protocol metrics across splits
    print("\n" + "=" * 60)
    print("L4PO AGGREGATE")
    print("=" * 60)
    from neurinferno.evaluation.aggregate_l4po import (
        load_seed_results, aggregate, print_table, write_summary_csv,
    )
    records = load_seed_results(results_dir)
    if records:
        per_proto, overall = aggregate(records)
        print_table(per_proto, overall)
        write_summary_csv(per_proto, overall, results_dir / "l4po_aggregate.csv")


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(lopo: dict[str, BoundaryMetrics]) -> None:
    print("\n" + "=" * 60)
    print("LOPO SUMMARY")
    print("=" * 60)
    print(f"{'Protocol':<12} {'P':>7} {'R':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 44)
    total_f1 = 0.0; n = 0
    for proto in LOPO_PROTOCOLS:
        m = lopo.get(proto)
        if m:
            print(f"{proto:<12} {m.precision:>7.4f} {m.recall:>7.4f} "
                  f"{m.fpr:>7.4f} {m.f1:>7.4f}")
            total_f1 += m.f1; n += 1
        else:
            print(f"{proto:<12}     N/A")
    if n:
        print("-" * 44)
        print(f"{'AVERAGE':<12} {'':>7} {'':>7} {'':>7} {total_f1/n:>7.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_eval(
    ckpt_root:    str   = "checkpoints/seed789",
    tier2_dir:    str   = "data/protocols",
    results_dir:  str   = "results/seed789",
    threshold:    float = 0.75,
    max_msgs:     int   = 1000,
    device:       str   = "auto",
    skip_lopo:    bool  = False,
    skip_l4po:    bool  = False,
) -> None:
    import torch as _torch
    if device == "auto":
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  threshold: {threshold}  |  max_msgs: {max_msgs}")

    ckpt_root   = Path(ckpt_root)
    tier2_dir   = Path(tier2_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lopo_results: dict[str, BoundaryMetrics] = {}

    if not skip_lopo:
        lopo_results = eval_lopo(
            lopo_ckpt_dir=ckpt_root / "lopo",
            tier2_dir=tier2_dir,
            results_dir=results_dir,
            threshold=threshold,
            max_msgs=max_msgs,
            device=device,
        )

    if not skip_l4po:
        eval_l4po(
            l4po_ckpt_dir=ckpt_root / "l4po",
            tier2_dir=tier2_dir,
            results_dir=results_dir / "l4po",
            threshold=threshold,
            max_msgs=max_msgs,
            device=device,
        )

    if lopo_results:
        _print_summary(lopo_results)

    print("\n" + "=" * 60)
    print("ALL DONE")
    print(f"  LOPO results : {results_dir / 'lopo_results.csv'}")
    print(f"  L4PO results : {results_dir / 'l4po' / 'l4po_aggregate.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate LOPO and L4PO checkpoints."
    )
    p.add_argument("--ckpt_root",    default="checkpoints/seed789")
    p.add_argument("--tier2_dir",    default="data/protocols")
    p.add_argument("--results_dir",  default="results/seed789")
    p.add_argument("--threshold",    type=float, default=0.75)
    p.add_argument("--max_msgs",     type=int,   default=1000)
    p.add_argument("--device",       default="auto")
    p.add_argument("--skip_lopo",    action="store_true")
    p.add_argument("--skip_l4po",    action="store_true")
    args = p.parse_args()
    run_eval(
        ckpt_root=args.ckpt_root,
        tier2_dir=args.tier2_dir,
        results_dir=args.results_dir,
        threshold=args.threshold,
        max_msgs=args.max_msgs,
        device=args.device,
        skip_lopo=args.skip_lopo,
        skip_l4po=args.skip_l4po,
    )
