"""
Failure-mode analysis: sample false positives and false negatives for
protocols where our model loses to BI, and summarize the byte patterns.

Writes results/failure_analysis.json.

Usage:
    python -m src.evaluation.failure_analysis \
        --lopo_ckpt_dir checkpoints/lopo \
        --tier2_dir     data/tier2_scapy \
        --results_dir   results \
        [--protocols    ip dns]   # analyse specific protocols
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from src.training.dataset import LOPO_PROTOCOLS
from src.evaluation.lopo import evaluate_protocol, run_model_inference, _to_tensor
from src.evaluation.ground_truth import load_labeled_messages, LabeledMessage
from src.evaluation.metrics import compute_boundary_metrics
from src.training.train_main import FieldInferenceModule
from src.model.encoder import PAD_IDX


def _describe_gap(msg: LabeledMessage, gap_idx: int, context: int = 2) -> str:
    """Human-readable description of the byte context around a gap."""
    raw   = bytes.fromhex(msg.bytes_hex)
    n     = len(raw)
    left  = max(0, gap_idx - context + 1)
    right = min(n, gap_idx + context + 2)
    snippet = raw[left:right].hex().upper()
    gaps_shown = "|".join(
        snippet[i*2:(i+1)*2] for i in range(len(snippet)//2)
    )
    return (f"gap@{gap_idx}  bytes[{left}:{right}]={gaps_shown}"
            f"  msg_len={n}  format={msg.format_id}")


def analyse_protocol(
    protocol:   str,
    lopo_ckpt:  Path,
    tier2_dir:  Path,
    max_msgs:   int   = 200,
    n_fp:       int   = 5,
    n_fn:       int   = 5,
    threshold:  float = 0.5,
) -> dict:
    """Return FP/FN samples for one protocol."""
    msgs   = load_labeled_messages(tier2_dir, protocol, max_msgs=max_msgs)
    model  = FieldInferenceModule.load_from_checkpoint(str(lopo_ckpt), strict=True)
    scores_list, gt_list = run_model_inference(model, msgs, n_msgs=32)

    fp_examples, fn_examples = [], []

    for msg, scores, gt in zip(msgs, scores_list, gt_list):
        pred = [1 if s >= threshold else 0 for s in scores]
        for i, (p, g) in enumerate(zip(pred, gt)):
            if p == 1 and g == 0:  # FP
                fp_examples.append(_describe_gap(msg, i))
            elif p == 0 and g == 1:  # FN
                fn_examples.append(_describe_gap(msg, i))

    import random
    rng = random.Random(0)
    rng.shuffle(fp_examples); rng.shuffle(fn_examples)

    return {
        "n_fp": len(fp_examples),
        "n_fn": len(fn_examples),
        "false_positives": fp_examples[:n_fp],
        "false_negatives": fn_examples[:n_fn],
    }


def run_failure_analysis(
    lopo_ckpt_dir: Path,
    tier2_dir:     Path,
    results_dir:   Path,
    protocols:     list[str] | None = None,
) -> None:
    if protocols is None:
        protocols = LOPO_PROTOCOLS

    output: dict[str, dict] = {}
    for proto in protocols:
        ckpt_candidates = list((lopo_ckpt_dir / proto).rglob("*.ckpt"))
        if not ckpt_candidates:
            print(f"  [{proto}] no checkpoint, skipping")
            continue

        def _loss(p):
            try:
                return float(p.stem.split("loss_total=")[1])
            except Exception:
                try:
                    return float(p.parent.name.split("loss_total=")[1])
                except Exception:
                    return float("inf")

        ckpt = min(ckpt_candidates, key=_loss)
        print(f"  Analysing {proto} ...")
        try:
            result = analyse_protocol(proto, ckpt, tier2_dir)
            output[proto] = result
            print(f"    FP={result['n_fp']}  FN={result['n_fn']}")
        except Exception as e:
            print(f"    ERROR: {e}")

    out_path = results_dir / "failure_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lopo_ckpt_dir", default="checkpoints/lopo")
    p.add_argument("--tier2_dir",     default="data/tier2_scapy")
    p.add_argument("--results_dir",   default="results")
    p.add_argument("--protocols",     nargs="*", default=None)
    args = p.parse_args()
    run_failure_analysis(
        Path(args.lopo_ckpt_dir), Path(args.tier2_dir),
        Path(args.results_dir), args.protocols,
    )
