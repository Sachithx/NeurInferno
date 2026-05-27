"""
LOPO evaluation: run our trained model on held-out Tier-2 protocol messages.

For each of the 10 LOPO protocols:
  1. Load the LOPO fine-tuned checkpoint.
  2. Load held-out Tier-2 test messages (with ground truth).
  3. Run inference in batches of n_msgs=32.
  4. Compute Precision / Recall / FPR / F1 against Scapy ground truth.

Returns a dict {protocol: BoundaryMetrics} and per-protocol PR-curve data.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import numpy as np

from src.training.dataset import LOPO_PROTOCOLS
from src.training.train_main import FieldInferenceModule
from src.evaluation.ground_truth import load_labeled_messages, LabeledMessage
from src.evaluation.metrics import (
    BoundaryMetrics, compute_boundary_metrics, aggregate_metrics,
    compute_pr_curve, auprc,
)
from src.model.encoder import PAD_IDX


# ── Batched inference ─────────────────────────────────────────────────────────

def _to_tensor(msgs: list[LabeledMessage], max_len: int = 512) -> torch.Tensor:
    """Pack a list of messages into a (1, N, L) tensor with padding."""
    N = len(msgs)
    seqs = []
    for m in msgs:
        raw = bytes.fromhex(m.bytes_hex)
        n   = min(len(raw), max_len)
        t   = torch.tensor(list(raw[:n]), dtype=torch.long)
        seqs.append(t)

    L = min(max(s.size(0) for s in seqs), max_len)

    x = torch.full((1, N, L), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        l = min(s.size(0), L)
        x[0, i, :l] = s[:l]
    return x  # (1, N, L)


@torch.no_grad()
def run_model_inference(
    model:   FieldInferenceModule,
    msgs:    list[LabeledMessage],
    n_msgs:  int   = 32,
    max_len: int   = 512,
    device:  str   = "cpu",
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Run model on messages in batches of n_msgs.

    Returns
    -------
    scores_per_msg  : list of per-gap sigmoid scores  (variable length)
    gt_per_msg      : list of per-gap ground-truth labels
    """
    model.eval()
    model.to(device)

    scores_all: list[list[float]] = []
    gt_all:     list[list[int]]   = []

    # Process in batches; keep a running index over gt too
    n_batches = math.ceil(len(msgs) / n_msgs)
    for b in range(n_batches):
        batch_msgs = msgs[b * n_msgs: (b + 1) * n_msgs]

        x = _to_tensor(batch_msgs, max_len=max_len).to(device)  # (1, N, L)
        out = model.model(x)                                      # ModelOutput

        # boundary_logits: (1, N, L-1, 1)
        bd_scores = out.boundary_logits.squeeze(-1).squeeze(0).sigmoid()  # (N, L-1)
        pad_mask  = out.padding_mask.squeeze(0)                            # (N, L)

        for n, m in enumerate(batch_msgs):
            raw_len = min(len(bytes.fromhex(m.bytes_hex)), max_len)
            n_gaps  = max(raw_len - 1, 0)
            if n_gaps == 0:
                continue
            # Scores for this message's real gaps (non-pad)
            scores = bd_scores[n, :n_gaps].cpu().tolist()
            gt     = m.boundary_per_gap[:n_gaps]
            scores_all.append(scores)
            gt_all.append(gt)

    return scores_all, gt_all


# ── Per-protocol evaluation ───────────────────────────────────────────────────

def evaluate_protocol(
    protocol:    str,
    lopo_ckpt:   Path,
    tier2_dir:   Path,
    max_msgs:    int   = 1000,
    n_msgs_batch: int  = 32,
    max_len:     int   = 512,
    threshold:   float = 0.5,
    device:      str   = "cpu",
) -> tuple[BoundaryMetrics, dict]:
    """
    Evaluate one LOPO protocol.

    Returns (metrics, extra) where extra contains PR-curve data and AUPRC.
    """
    msgs = load_labeled_messages(tier2_dir, protocol, max_msgs=max_msgs)
    print(f"  [{protocol}] {len(msgs)} test messages")

    model = FieldInferenceModule.load_from_checkpoint(str(lopo_ckpt), strict=True)
    scores_list, gt_list = run_model_inference(
        model, msgs, n_msgs=n_msgs_batch, max_len=max_len, device=device)

    # Per-message metrics at threshold 0.5
    per_msg = [
        compute_boundary_metrics(
            [1 if s >= threshold else 0 for s in sc], gt
        )
        for sc, gt in zip(scores_list, gt_list)
    ]
    agg = aggregate_metrics(per_msg)

    # PR curve (micro over all gaps)
    all_scores = [s for sc in scores_list for s in sc]
    all_gt     = [g for gt in gt_list     for g in gt]
    prec, rec, thr = compute_pr_curve(all_scores, all_gt)
    auc = auprc(prec, rec)

    extra = {"precisions": prec, "recalls": rec, "thresholds": thr, "auprc": auc}
    return agg, extra


# ── Full LOPO loop ─────────────────────────────────────────────────────────────

def run_lopo_evaluation(
    lopo_ckpt_dir: Path,
    tier2_dir:     Path,
    protocols:     list[str] | None = None,
    max_msgs:      int   = 1000,
    n_msgs_batch:  int   = 32,
    max_len:       int   = 512,
    threshold:     float = 0.5,
    device:        str   = "cpu",
) -> tuple[dict[str, BoundaryMetrics], dict[str, dict]]:
    """
    Run LOPO evaluation for all protocols.

    Returns
    -------
    results   : {protocol: BoundaryMetrics}
    pr_curves : {protocol: {precisions, recalls, thresholds, auprc}}
    """
    if protocols is None:
        protocols = LOPO_PROTOCOLS

    results: dict[str, BoundaryMetrics] = {}
    pr_curves: dict[str, dict] = {}

    for proto in protocols:
        # Find checkpoint — PL may create subdirs from filename slashes
        ckpt_candidates = list((lopo_ckpt_dir / proto).rglob("*.ckpt"))
        if not ckpt_candidates:
            print(f"  [{proto}] WARNING: no checkpoint found, skipping")
            continue

        def _loss_from_path(p: Path) -> float:
            # Path looks like .../val/loss_total=1.8716.ckpt
            try:
                return float(p.stem.split("loss_total=")[1])
            except Exception:
                try:
                    return float(p.parent.name.split("loss_total=")[1])
                except Exception:
                    return float("inf")

        ckpt = min(ckpt_candidates, key=_loss_from_path)
        print(f"  [{proto}] using {ckpt.name}")

        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        try:
            metrics, extra = evaluate_protocol(
                protocol=proto, lopo_ckpt=ckpt, tier2_dir=tier2_dir,
                max_msgs=max_msgs, n_msgs_batch=n_msgs_batch,
                max_len=max_len, threshold=threshold, device=device,
            )
            results[proto]   = metrics
            pr_curves[proto] = extra
            print(f"  [{proto}] P={metrics.precision:.3f}  R={metrics.recall:.3f}"
                  f"  FPR={metrics.fpr:.3f}  F1={metrics.f1:.3f}"
                  f"  AUPRC={extra['auprc']:.3f}")
        except Exception as e:
            print(f"  [{proto}] ERROR: {e}")

    return results, pr_curves
