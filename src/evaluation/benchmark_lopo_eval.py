"""
Benchmark LOPO evaluation — proper per-protocol checkpoints.

For each of the 10 BI benchmark protocols the model is evaluated using the
checkpoint that was fine-tuned with that protocol's training data excluded:

  overlapping-4  (bgp, dhcp, modbus, ntp48)
    Loaded from checkpoints/lopo/<tier2_name>/ — fine-tuned from Stage-2
    with the Tier-2 equivalent excluded.

  clean-6  (dnp3, mavlink, mirai, smb, smb2, tutorial)
    No Tier-2 equivalent exists.  Stage-2 is already the zero-shot
    checkpoint — these protocol families were never in any training stage.

Ground truth: eval_harness.ground_truth_for_sample() matching §IX-A.

Usage
-----
    python -m src.evaluation.benchmark_lopo_eval \\
        --main_ckpt   checkpoints/main/best.ckpt \\
        --lopo_dir    checkpoints/lopo \\
        --bench_dir   data/tier3_benchmark/top-level/1000 \\
        --results_dir results
"""

from __future__ import annotations

import argparse
import csv
import math
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

# Benchmark protocol → Tier-2 LOPO directory name.
# None means the protocol has no Tier-2 equivalent; use Stage-2 checkpoint.
BENCHMARK_TO_LOPO: dict[str, str | None] = {
    "bgp":      "bgp_raw",
    "dhcp":     "dhcp",
    "modbus":   "modbus",
    "ntp48":    "ntp",
    "dnp3":     None,
    "mavlink":  None,
    "mirai":    None,
    "smb":      None,
    "smb2":     None,
    "tutorial": None,
}


# ── Checkpoint selection ──────────────────────────────────────────────────────

def _best_ckpt_in(lopo_proto_dir: Path) -> Path:
    """Return the best (lowest val-loss) checkpoint inside a LOPO protocol dir.

    PL writes checkpoints into subdirs when the filename contains '/' from
    metric names (e.g. val/loss_total=…), so we use rglob.
    """
    ckpts = list(lopo_proto_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {lopo_proto_dir}")

    def _loss(p: Path) -> float:
        # filename contains val/loss_total=<float>
        for part in p.parts:
            if "loss_total=" in part:
                try:
                    return float(part.split("loss_total=")[-1].replace(".ckpt", ""))
                except ValueError:
                    pass
        return float("inf")

    return min(ckpts, key=_loss)


def select_checkpoint(
    protocol:  str,
    main_ckpt: Path,
    lopo_dir:  Path,
) -> tuple[Path, str]:
    """Return (checkpoint_path, reason_tag) for a given benchmark protocol.

    Looks for a per-protocol checkpoint in this priority order:
      1. lopo_dir/{protocol}/          — BI-mode training (bi_mode=True)
      2. lopo_dir/{tier2_name}/        — legacy Tier-2 named training
      3. main_ckpt                     — zero-shot fallback
    """
    # Priority 1: BI-mode checkpoint (dir named by BI protocol name)
    bi_dir = lopo_dir / protocol
    if bi_dir.exists():
        try:
            ckpt = _best_ckpt_in(bi_dir)
            return ckpt, f"lopo-bi-{protocol}"
        except FileNotFoundError:
            pass

    # Priority 2: legacy Tier-2 named checkpoint
    tier2_name = BENCHMARK_TO_LOPO.get(protocol)
    if tier2_name is not None:
        t2_dir = lopo_dir / tier2_name
        if t2_dir.exists():
            try:
                ckpt = _best_ckpt_in(t2_dir)
                return ckpt, f"lopo-{tier2_name}"
            except FileNotFoundError:
                pass

    # Priority 3: zero-shot fallback
    tag = "stage2-zero-shot" if BENCHMARK_TO_LOPO.get(protocol) is None \
          else "stage2-fallback (lopo ckpt missing)"
    return main_ckpt, tag


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_benchmark_lopo_eval(
    main_ckpt:    str,
    lopo_dir:     str  = "checkpoints/lopo",
    bench_dir:    str  = "data/tier3_benchmark/top-level/1000",
    results_dir:  str  = "results",
    clean_only:   bool = False,
    n_msgs_batch: int  = 32,
    max_len:      int  = 512,
    threshold:    float = 0.5,
) -> dict[str, dict]:
    main_ckpt   = Path(main_ckpt)
    lopo_dir    = Path(lopo_dir)
    bench_dir   = Path(bench_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    protocols = CLEAN_PROTOCOLS if clean_only else ALL_PROTOCOLS
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    # Cache loaded models to avoid reloading the same checkpoint multiple times
    _model_cache: dict[Path, FieldInferenceModule] = {}

    def _load(ckpt: Path) -> FieldInferenceModule:
        if ckpt not in _model_cache:
            print(f"  Loading {ckpt.name}")
            m = FieldInferenceModule.load_from_checkpoint(str(ckpt), strict=True)
            m.eval()
            m.to(device)
            _model_cache[ckpt] = m
        return _model_cache[ckpt]

    all_results: dict[str, dict] = {}

    for proto in protocols:
        ckpt, tag = select_checkpoint(proto, main_ckpt, lopo_dir)
        print(f"\n[{proto}]  ckpt={tag}", end="  ")

        try:
            messages = load_benchmark_messages(bench_dir, proto)
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            continue

        print(f"{len(messages)} msgs", end="  ")

        model     = _load(ckpt)
        scores    = run_inference_scores(model, messages,
                                         n_msgs=n_msgs_batch,
                                         max_len=max_len,
                                         device=device)
        pred_sets = scores_to_pred_sets(scores, threshold)
        metrics   = evaluate_protocol(proto, messages, pred_sets)
        metrics["ckpt_tag"] = tag
        all_results[proto]  = metrics

        print(f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"FPR={metrics['fpr']:.3f}  F1={metrics['f1']:.3f}")

    _write_csv(all_results, results_dir / "benchmark_lopo_eval.csv")
    _print_summary(all_results)
    return all_results


# ── Output ────────────────────────────────────────────────────────────────────

def _write_csv(results: dict, path: Path) -> None:
    fieldnames = ["protocol", "leakage_free", "ckpt_used",
                  "precision", "recall", "fpr", "f1", "n_messages"]
    rows = []
    for proto, m in results.items():
        rows.append({
            "protocol":     proto,
            "leakage_free": "yes" if proto in CLEAN_PROTOCOLS else "no",
            "ckpt_used":    m.get("ckpt_tag", ""),
            "precision":    f"{m['precision']:.4f}",
            "recall":       f"{m['recall']:.4f}",
            "fpr":          f"{m['fpr']:.4f}",
            "f1":           f"{m['f1']:.4f}",
            "n_messages":   int(m["n_messages"]),
        })

    clean_rows = [m for p, m in results.items() if p in CLEAN_PROTOCOLS]
    if clean_rows:
        rows.append({
            "protocol":     "AVERAGE (clean-6)",
            "leakage_free": "yes",
            "ckpt_used":    "stage2-zero-shot",
            "precision":    f"{sum(r['precision'] for r in clean_rows)/len(clean_rows):.4f}",
            "recall":       f"{sum(r['recall']    for r in clean_rows)/len(clean_rows):.4f}",
            "fpr":          f"{sum(r['fpr']       for r in clean_rows)/len(clean_rows):.4f}",
            "f1":           f"{sum(r['f1']        for r in clean_rows)/len(clean_rows):.4f}",
            "n_messages":   sum(int(r["n_messages"]) for r in clean_rows),
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {path}")


def _print_summary(results: dict) -> None:
    print(f"\n{'Protocol':<12} {'Leak-free':>9} {'Checkpoint':>22} "
          f"{'P':>7} {'R':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 72)
    for proto, m in results.items():
        tag = "yes" if proto in CLEAN_PROTOCOLS else "no "
        ckpt = m.get("ckpt_tag", "")[:22]
        print(f"{proto:<12} {tag:>9} {ckpt:>22} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} "
              f"{m['fpr']:7.3f} {m['f1']:7.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_ckpt",    required=True)
    p.add_argument("--lopo_dir",     default="checkpoints/lopo")
    p.add_argument("--bench_dir",    default="data/tier3_benchmark/top-level/1000")
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--clean_only",   action="store_true")
    p.add_argument("--n_msgs_batch", type=int,   default=32)
    p.add_argument("--max_len",      type=int,   default=512)
    p.add_argument("--threshold",    type=float, default=0.5)
    args = p.parse_args()
    run_benchmark_lopo_eval(**vars(args))
