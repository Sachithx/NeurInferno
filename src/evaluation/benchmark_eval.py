"""
Benchmark evaluation on BI top-level protocols using eval_harness.py.

Clean-6 protocols (never seen during any training stage):
    dnp3, mavlink, mirai, smb, smb2, tutorial

These are evaluated using the Stage-2 main checkpoint directly.
No LOPO fine-tuning was done for them — the Stage-2 model is tested
zero-shot on protocol families it has never seen.

The other 4 (bgp, dhcp, modbus, ntp48) have Tier-2 equivalents in
training data; they are excluded from the leakage-free evaluation but
can be run optionally with --include_overlapping.

Ground truth: eval_harness.ground_truth_for_sample() — deterministic
per-protocol parsers + sample-level constant-field merging, matching
BinaryInferno's §IX-A methodology exactly.

Usage:
    python -m src.evaluation.benchmark_eval \\
        --main_ckpt  checkpoints/main/best.ckpt \\
        --bench_dir  data/tier3_benchmark/top-level/1000 \\
        --results_dir results \\
        [--include_overlapping]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import numpy as np

from src.evaluation.eval_harness import (
    PARSERS, ground_truth_for_sample, _boundary_metrics, PROTOCOL_ENDIAN,
)
from src.training.train_main import FieldInferenceModule
from src.model.encoder import PAD_IDX

# Protocols with no Tier-2 training equivalent — zero-shot, leakage-free
CLEAN_PROTOCOLS = ["dnp3", "mavlink", "mirai", "smb", "smb2", "tutorial"]

# Protocols whose Tier-2 equivalent was in Stage-2 training.
# Evaluated with current Stage-2 checkpoint; leakage_free=no is noted in
# the CSV and the paper. The specific benchmark messages were never seen
# during training (only same-protocol Scapy-generated messages were).
OVERLAPPING_PROTOCOLS = ["bgp", "dhcp", "modbus", "ntp48"]

ALL_PROTOCOLS = CLEAN_PROTOCOLS + OVERLAPPING_PROTOCOLS


# ── Data loading ──────────────────────────────────────────────────────────────

def load_benchmark_messages(bench_dir: Path, protocol: str) -> list[bytes]:
    """Read hex-per-line benchmark file → list of bytes objects."""
    path = bench_dir / f"{protocol}.txt.input"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    messages.append(bytes.fromhex(line))
                except ValueError:
                    pass
    return messages


# ── Boundary format conversion ────────────────────────────────────────────────

def boundary_per_gap_to_set(boundary_per_gap: list[int]) -> set[int]:
    """
    Convert our binary gap list to eval_harness boundary-position set.

    boundary_per_gap[i] = 1  → boundary between byte i and byte i+1
                            → harness position i+1  (right-endpoint of field)
    """
    return {i + 1 for i, v in enumerate(boundary_per_gap) if v == 1}


# ── Model inference ───────────────────────────────────────────────────────────

def _pack_batch(msgs: list[bytes], max_len: int = 512) -> torch.Tensor:
    """Pack a list of bytes into (1, N, L) long tensor with PAD_IDX."""
    seqs = [
        torch.tensor(list(m[:max_len]), dtype=torch.long)
        for m in msgs
    ]
    L = min(max(s.size(0) for s in seqs), max_len)
    x = torch.full((1, len(seqs), L), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        l = min(s.size(0), L)
        x[0, i, :l] = s[:l]
    return x


@torch.no_grad()
def run_inference_scores(
    model:   FieldInferenceModule,
    messages: list[bytes],
    n_msgs:  int = 32,
    max_len: int = 512,
    device:  str = "cpu",
) -> list[list[float]]:
    """
    Run model and return raw sigmoid scores per gap, one list per message.
    len(result[i]) == max(0, min(len(messages[i]), max_len) - 1).
    """
    model.eval()
    model.to(device)

    all_scores: list[list[float]] = []
    n_batches = math.ceil(len(messages) / n_msgs)

    for b in range(n_batches):
        batch = messages[b * n_msgs: (b + 1) * n_msgs]
        x = _pack_batch(batch, max_len=max_len).to(device)
        out = model.model(x)

        bd_scores = out.boundary_logits.squeeze(-1).squeeze(0).sigmoid()  # (N, L-1)

        for n, msg in enumerate(batch):
            raw_len = min(len(msg), max_len)
            n_gaps  = max(raw_len - 1, 0)
            if n_gaps == 0:
                all_scores.append([])
                continue
            all_scores.append(bd_scores[n, :n_gaps].cpu().tolist())

    return all_scores


@torch.no_grad()
def run_inference_scores_with_entropy(
    model:   FieldInferenceModule,
    messages: list[bytes],
    n_msgs:  int = 32,
    max_len: int = 512,
    device:  str = "cpu",
) -> tuple[list[list[float]], list[np.ndarray]]:
    """
    Run model and return (boundary scores, ByteLM entropy) per message.

    Returns
    -------
    all_scores  : list of per-gap sigmoid scores, one list per message
    all_entropy : list of per-byte entropy arrays (float32), one per message,
                  length = min(len(msg), max_len)
    """
    model.eval()
    model.to(device)

    all_scores:  list[list[float]]  = []
    all_entropy: list[np.ndarray]   = []
    n_batches = math.ceil(len(messages) / n_msgs)

    for b in range(n_batches):
        batch = messages[b * n_msgs: (b + 1) * n_msgs]
        x = _pack_batch(batch, max_len=max_len).to(device)
        out = model.model(x)

        bd_scores = out.boundary_logits.squeeze(-1).squeeze(0).sigmoid()  # (N, L-1)
        entropy   = out.entropy.squeeze(0).cpu().numpy()                   # (N, L)

        for n, msg in enumerate(batch):
            raw_len = min(len(msg), max_len)
            n_gaps  = max(raw_len - 1, 0)
            if n_gaps == 0:
                all_scores.append([])
                all_entropy.append(np.zeros(raw_len, dtype=np.float32))
                continue
            all_scores.append(bd_scores[n, :n_gaps].cpu().tolist())
            all_entropy.append(entropy[n, :raw_len].astype(np.float32))

    return all_scores, all_entropy


def scores_to_pred_sets(
    all_scores: list[list[float]],
    threshold:  float,
) -> list[set[int]]:
    """Apply a threshold to raw scores → list of boundary-position sets."""
    return [
        boundary_per_gap_to_set([1 if s >= threshold else 0 for s in scores])
        for scores in all_scores
    ]


def run_inference(
    model:     FieldInferenceModule,
    messages:  list[bytes],
    n_msgs:    int   = 32,
    max_len:   int   = 512,
    threshold: float = 0.5,
    device:    str   = "cpu",
) -> list[set[int]]:
    """Convenience wrapper: run_inference_scores + threshold."""
    scores = run_inference_scores(model, messages, n_msgs=n_msgs,
                                  max_len=max_len, device=device)
    return scores_to_pred_sets(scores, threshold)


# ── Per-protocol evaluation ───────────────────────────────────────────────────

def evaluate_protocol(
    protocol:  str,
    messages:  list[bytes],
    pred_sets: list[set[int]],
) -> dict[str, float]:
    """
    Evaluate predicted boundary sets against eval_harness ground truth.

    Uses ground_truth_for_sample (sample-level constant-field merging)
    matching BinaryInferno's methodology.
    """
    gt_sets = ground_truth_for_sample(protocol, messages)

    per_msg = [
        _boundary_metrics(pred, gt, len(msg))
        for pred, gt, msg in zip(pred_sets, gt_sets, messages)
    ]
    n = len(per_msg)
    result = {
        k: sum(m[k] for m in per_msg) / n
        for k in ("precision", "recall", "fpr", "f1", "tp", "fp", "fn", "tn")
    }
    result["n_messages"] = float(n)
    return result


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_benchmark_eval(
    main_ckpt:           str,
    bench_dir:           str  = "data/tier3_benchmark/top-level/1000",
    results_dir:         str  = "results",
    clean_only: bool = False,
    n_msgs_batch:        int  = 32,
    max_len:             int  = 512,
    threshold:           float = 0.5,
) -> dict[str, dict]:
    bench_dir   = Path(bench_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    protocols = CLEAN_PROTOCOLS if clean_only else ALL_PROTOCOLS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {main_ckpt}")
    model = FieldInferenceModule.load_from_checkpoint(str(main_ckpt), strict=True)
    model.eval()
    model.to(device)

    all_results: dict[str, dict] = {}

    for proto in protocols:
        print(f"\n[{proto}]", end="  ")
        try:
            messages = load_benchmark_messages(bench_dir, proto)
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            continue

        print(f"{len(messages)} messages", end="  ")

        pred_sets = run_inference(
            model, messages,
            n_msgs=n_msgs_batch, max_len=max_len,
            threshold=threshold, device=device,
        )

        metrics = evaluate_protocol(proto, messages, pred_sets)
        all_results[proto] = metrics

        print(f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"FPR={metrics['fpr']:.3f}  F1={metrics['f1']:.3f}")

    _write_csv(all_results, results_dir / "benchmark_eval.csv")
    _print_summary(all_results)
    return all_results


def _write_csv(results: dict, path: Path) -> None:
    fieldnames = ["protocol", "leakage_free", "precision", "recall", "fpr", "f1", "n_messages"]
    rows = []
    for proto, m in results.items():
        rows.append({
            "protocol":     proto,
            "leakage_free": "yes" if proto in CLEAN_PROTOCOLS else "no",
            "precision":    f"{m['precision']:.4f}",
            "recall":       f"{m['recall']:.4f}",
            "fpr":          f"{m['fpr']:.4f}",
            "f1":           f"{m['f1']:.4f}",
            "n_messages":   int(m["n_messages"]),
        })
    # Average over clean protocols
    clean_rows = [m for p, m in results.items() if p in CLEAN_PROTOCOLS]
    if clean_rows:
        rows.append({
            "protocol":     "AVERAGE (clean-6)",
            "leakage_free": "yes",
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
    print(f"\n{'Protocol':<12} {'Leak-free':>9} {'P':>7} {'R':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 50)
    for proto, m in results.items():
        tag = "yes" if proto in CLEAN_PROTOCOLS else "no "
        print(f"{proto:<12} {tag:>9} {m['precision']:7.3f} {m['recall']:7.3f} "
              f"{m['fpr']:7.3f} {m['f1']:7.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_ckpt",           required=True)
    p.add_argument("--bench_dir",           default="data/tier3_benchmark/top-level/1000")
    p.add_argument("--results_dir",         default="results")
    p.add_argument("--clean_only", action="store_true")
    p.add_argument("--n_msgs_batch",        type=int,   default=32)
    p.add_argument("--max_len",             type=int,   default=512)
    p.add_argument("--threshold",           type=float, default=0.5)
    args = p.parse_args()
    run_benchmark_eval(**vars(args))
