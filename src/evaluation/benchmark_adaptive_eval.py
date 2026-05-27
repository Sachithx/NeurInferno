"""
GT-free adaptive threshold selection on BI benchmark data.

For each protocol, the threshold is chosen automatically using the
ByteLM surprise F-ratio — no ground truth required.

Mechanism
---------
1. Run model once per protocol → per-gap sigmoid scores + per-byte ByteLM entropy.
2. Sweep thresholds; for each threshold score every message with surprise_fratio:
       fratio(entropy[i, :len(msg)], pred_set[i])
   and take the mean across messages.
3. Pick the threshold with the highest mean fratio.
4. Evaluate at that threshold using eval_harness ground truth.

The fratio measures how well the predicted boundaries partition the entropy
signal: boundaries that align with entropy peaks / transitions score higher
than boundaries that cut through flat, predictable field interiors.

Assumption: fields must have *uniformly* different entropy levels across their
bytes (e.g. opaque payload → high entropy throughout; fixed enum → low entropy
throughout).  If only the first byte of each new field is surprising (narrow
spike at transition) the criterion is weak — consider --method mdl instead.

Outputs
-------
results/benchmark_adaptive_eval.csv — per-protocol metrics at adaptive threshold
                                       + comparison column vs fixed t=0.5
results/benchmark_adaptive_thresholds.csv — chosen threshold per protocol

Usage
-----
    python -m src.evaluation.benchmark_adaptive_eval \\
        --main_ckpt   checkpoints/main/best.ckpt \\
        --bench_dir   data/tier3_benchmark/top-level/1000 \\
        --results_dir results \\
        [--n_thresholds 19] [--lam 0.5] [--method fratio|mdl]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.evaluation.benchmark_eval import (
    CLEAN_PROTOCOLS, ALL_PROTOCOLS,
    load_benchmark_messages,
    run_inference_scores_with_entropy,
    run_inference_scores,
    scores_to_pred_sets,
    evaluate_protocol,
)
from src.evaluation.field_mdl import surprise_fratio, field_mdl_score
from src.training.train_main import FieldInferenceModule


# ── GT-free threshold selection ───────────────────────────────────────────────

def _fratio_score_for_threshold(
    all_entropy: list[np.ndarray],
    pred_sets:   list[set[int]],
    lam:         float,
) -> float:
    """Average per-message surprise_fratio across all messages."""
    scores = []
    for ent, bounds in zip(all_entropy, pred_sets):
        if len(ent) < 2:
            continue
        scores.append(surprise_fratio(ent, bounds, lam=lam))
    return float(np.mean(scores)) if scores else -float("inf")


def _majority_vote_bounds(pred_sets: list[set[int]], min_frac: float = 0.5) -> set[int]:
    """Boundaries present in at least min_frac of the predicted sets."""
    if not pred_sets:
        return set()
    counts: dict[int, int] = defaultdict(int)
    for ps in pred_sets:
        for b in ps:
            counts[b] += 1
    n = len(pred_sets)
    return {pos for pos, cnt in counts.items() if cnt / n >= min_frac}


def _mdl_score_for_threshold(
    messages:  list[bytes],
    pred_sets: list[set[int]],
    lam:       float,
) -> float:
    """MDL score of the majority-vote consensus boundary set (negated so higher=better)."""
    consensus = _majority_vote_bounds(pred_sets)
    return -field_mdl_score(messages, consensus, lam=lam)


def select_threshold_gt_free(
    all_scores:  list[list[float]],
    all_entropy: list[np.ndarray],
    messages:    list[bytes],
    thresholds:  list[float],
    method:      str   = "fratio",
    lam:         float = 0.5,
) -> tuple[float, list[float]]:
    """
    Pick the threshold that maximises the GT-free criterion.

    Returns
    -------
    best_threshold : float
    criterion_scores : list[float] — one score per threshold (for diagnostics)
    """
    criterion_scores = []
    for t in thresholds:
        pred_sets = scores_to_pred_sets(all_scores, t)
        if method == "fratio":
            score = _fratio_score_for_threshold(all_entropy, pred_sets, lam)
        elif method == "mdl":
            score = _mdl_score_for_threshold(messages, pred_sets, lam)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'fratio' or 'mdl'.")
        criterion_scores.append(score)

    best_idx = int(np.argmax(criterion_scores))
    return thresholds[best_idx], criterion_scores


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_benchmark_adaptive_eval(
    main_ckpt:    str,
    bench_dir:    str   = "data/tier3_benchmark/top-level/1000",
    results_dir:  str   = "results",
    clean_only:   bool  = False,
    n_thresholds: int   = 19,
    thresholds:   list[float] | None = None,
    method:       str   = "fratio",
    lam:          float = 0.5,
    n_msgs_batch: int   = 32,
    max_len:      int   = 512,
) -> dict[str, dict]:
    bench_dir   = Path(bench_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        thresholds = [round(v, 4) for v in
                      np.linspace(0.05, 0.95, n_thresholds).tolist()]

    protocols = CLEAN_PROTOCOLS if clean_only else ALL_PROTOCOLS
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {main_ckpt}")
    model = FieldInferenceModule.load_from_checkpoint(str(main_ckpt), strict=True)
    model.eval()
    model.to(device)

    all_results:    dict[str, dict] = {}
    chosen_thresholds: list[dict]   = []

    for proto in protocols:
        print(f"\n[{proto}]", end="  ")
        try:
            messages = load_benchmark_messages(bench_dir, proto)
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            continue

        print(f"{len(messages)} msgs", end="  ")

        # ── One forward pass → scores + entropy ──────────────────────────────
        all_scores, all_entropy = run_inference_scores_with_entropy(
            model, messages,
            n_msgs=n_msgs_batch, max_len=max_len, device=device,
        )

        # ── GT-free threshold selection ───────────────────────────────────────
        best_t, crit_scores = select_threshold_gt_free(
            all_scores, all_entropy, messages,
            thresholds, method=method, lam=lam,
        )

        # ── Evaluate at adaptive threshold ────────────────────────────────────
        pred_sets_adaptive = scores_to_pred_sets(all_scores, best_t)
        m_adaptive = evaluate_protocol(proto, messages, pred_sets_adaptive)

        # ── Also evaluate at fixed t=0.5 for comparison ───────────────────────
        pred_sets_fixed = scores_to_pred_sets(all_scores, 0.5)
        m_fixed = evaluate_protocol(proto, messages, pred_sets_fixed)

        all_results[proto] = {
            "adaptive": m_adaptive,
            "fixed":    m_fixed,
            "best_t":   best_t,
            "crit_scores": crit_scores,
        }

        delta_f1 = m_adaptive["f1"] - m_fixed["f1"]
        print(
            f"t*={best_t:.2f}  "
            f"F1(adaptive)={m_adaptive['f1']:.3f}  "
            f"F1(fixed-0.5)={m_fixed['f1']:.3f}  "
            f"ΔF1={delta_f1:+.3f}"
        )
        chosen_thresholds.append({
            "protocol":    proto,
            "best_t":      best_t,
            "crit_score":  max(crit_scores),
        })

    _write_results_csv(all_results, results_dir / "benchmark_adaptive_eval.csv")
    _write_threshold_csv(chosen_thresholds, results_dir / "benchmark_adaptive_thresholds.csv")
    _print_summary(all_results)
    return all_results


# ── Output ────────────────────────────────────────────────────────────────────

def _write_results_csv(results: dict, path: Path) -> None:
    fieldnames = [
        "protocol", "leakage_free",
        "adaptive_t", "adaptive_P", "adaptive_R", "adaptive_FPR", "adaptive_F1",
        "fixed_P",    "fixed_R",    "fixed_FPR",    "fixed_F1",
        "delta_F1",
    ]
    rows = []
    for proto, r in results.items():
        a, f = r["adaptive"], r["fixed"]
        rows.append({
            "protocol":     proto,
            "leakage_free": "yes" if proto in CLEAN_PROTOCOLS else "no",
            "adaptive_t":   f"{r['best_t']:.2f}",
            "adaptive_P":   f"{a['precision']:.4f}",
            "adaptive_R":   f"{a['recall']:.4f}",
            "adaptive_FPR": f"{a['fpr']:.4f}",
            "adaptive_F1":  f"{a['f1']:.4f}",
            "fixed_P":      f"{f['precision']:.4f}",
            "fixed_R":      f"{f['recall']:.4f}",
            "fixed_FPR":    f"{f['fpr']:.4f}",
            "fixed_F1":     f"{f['f1']:.4f}",
            "delta_F1":     f"{a['f1'] - f['f1']:+.4f}",
        })

    # Clean-6 average rows
    def _avg(key, metric, rows_subset):
        return sum(r[f"{key}_{metric}"] for r in rows_subset) / len(rows_subset)

    clean_res = {p: r for p, r in results.items() if p in CLEAN_PROTOCOLS}
    if clean_res:
        ca = [r["adaptive"] for r in clean_res.values()]
        cf = [r["fixed"]    for r in clean_res.values()]
        rows.append({
            "protocol":     "AVERAGE (clean-6)",
            "leakage_free": "yes",
            "adaptive_t":   "—",
            "adaptive_P":   f"{sum(m['precision'] for m in ca)/len(ca):.4f}",
            "adaptive_R":   f"{sum(m['recall']    for m in ca)/len(ca):.4f}",
            "adaptive_FPR": f"{sum(m['fpr']       for m in ca)/len(ca):.4f}",
            "adaptive_F1":  f"{sum(m['f1']        for m in ca)/len(ca):.4f}",
            "fixed_P":      f"{sum(m['precision'] for m in cf)/len(cf):.4f}",
            "fixed_R":      f"{sum(m['recall']    for m in cf)/len(cf):.4f}",
            "fixed_FPR":    f"{sum(m['fpr']       for m in cf)/len(cf):.4f}",
            "fixed_F1":     f"{sum(m['f1']        for m in cf)/len(cf):.4f}",
            "delta_F1":     f"{sum(m['f1'] for m in ca)/len(ca) - sum(m['f1'] for m in cf)/len(cf):+.4f}",
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {path}")


def _write_threshold_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["protocol", "best_t", "crit_score"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({
                "protocol":   row["protocol"],
                "best_t":     f"{row['best_t']:.2f}",
                "crit_score": f"{row['crit_score']:.4f}",
            })
    print(f"Saved → {path}")


def _print_summary(results: dict) -> None:
    print(f"\n{'Protocol':<12} {'t*':>5} "
          f"{'Adapt-F1':>9} {'Fixed-F1':>9} {'ΔF1':>7}")
    print("-" * 47)
    for proto, r in results.items():
        a, f = r["adaptive"], r["fixed"]
        print(f"{proto:<12} {r['best_t']:5.2f} "
              f"{a['f1']:9.3f} {f['f1']:9.3f} {a['f1']-f['f1']:+7.3f}")

    clean_res = [r for p, r in results.items() if p in CLEAN_PROTOCOLS]
    if clean_res:
        avg_a = sum(r["adaptive"]["f1"] for r in clean_res) / len(clean_res)
        avg_f = sum(r["fixed"]["f1"]    for r in clean_res) / len(clean_res)
        print("-" * 47)
        print(f"{'avg (clean-6)':<12} {'—':>5} "
              f"{avg_a:9.3f} {avg_f:9.3f} {avg_a-avg_f:+7.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_ckpt",    required=True)
    p.add_argument("--bench_dir",    default="data/tier3_benchmark/top-level/1000")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--clean_only",   action="store_true")
    p.add_argument("--n_thresholds", type=int,   default=19,
                   help="Grid size in [0.05, 0.95] (ignored if --thresholds given)")
    p.add_argument("--thresholds",   type=float, nargs="+", default=None)
    p.add_argument("--method",       choices=["fratio", "mdl"], default="fratio")
    p.add_argument("--lam",          type=float, default=0.5,
                   help="Penalty weight for the fratio/mdl criterion")
    p.add_argument("--n_msgs_batch", type=int,   default=32)
    p.add_argument("--max_len",      type=int,   default=512)
    args = p.parse_args()
    run_benchmark_adaptive_eval(**vars(args))
