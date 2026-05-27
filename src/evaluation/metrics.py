"""
Boundary-inference metrics matching BinaryInferno paper Section IX-A (eqs 4-6).

TP = inferred boundary coincides with a ground-truth boundary
FP = inferred boundary with no matching ground truth
FN = ground-truth boundary not inferred
Negatives = non-boundary gaps (for FPR denominator)

Metrics are computed per-message and then macro-averaged across messages.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class BoundaryMetrics:
    precision:  float
    recall:     float
    fpr:        float
    f1:         float
    tp:         int
    fp:         int
    fn:         int
    tn:         int

    @property
    def negatives(self) -> int:
        return self.tn + self.fp


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_boundary_metrics(
    pred:  Sequence[int],   # binary per-gap predictions  (0/1)
    truth: Sequence[int],   # binary per-gap ground truth (0/1)
) -> BoundaryMetrics:
    """Per-message boundary metrics."""
    pred  = np.asarray(pred,  dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    assert len(pred) == len(truth), f"length mismatch: {len(pred)} vs {len(truth)}"

    tp = int(( pred &  truth).sum())
    fp = int(( pred & ~truth).sum())
    fn = int((~pred &  truth).sum())
    tn = int((~pred & ~truth).sum())

    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    fpr       = _safe_div(fp, fp + tn)   # FP / negatives
    f1        = _safe_div(2 * precision * recall, precision + recall)

    return BoundaryMetrics(
        precision=precision, recall=recall, fpr=fpr, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def aggregate_metrics(per_msg: list[BoundaryMetrics]) -> BoundaryMetrics:
    """Macro-average over a list of per-message metrics."""
    if not per_msg:
        return BoundaryMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    tp = sum(m.tp for m in per_msg)
    fp = sum(m.fp for m in per_msg)
    fn = sum(m.fn for m in per_msg)
    tn = sum(m.tn for m in per_msg)

    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    fpr       = _safe_div(fp, fp + tn)
    f1        = _safe_div(2 * precision * recall, precision + recall)

    return BoundaryMetrics(
        precision=precision, recall=recall, fpr=fpr, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def compute_pr_curve(
    scores: Sequence[float],  # per-gap boundary score (e.g. sigmoid output)
    truth:  Sequence[int],    # per-gap ground truth (0/1)
    n_thresholds: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (precisions, recalls, thresholds) arrays for a PR curve.
    Aggregated micro over all gaps from all messages (pass flattened).
    """
    scores = np.asarray(scores, dtype=float)
    truth  = np.asarray(truth,  dtype=bool)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    precisions = []
    recalls    = []
    for t in thresholds:
        pred = scores >= t
        tp = ( pred &  truth).sum()
        fp = ( pred & ~truth).sum()
        fn = (~pred &  truth).sum()
        precisions.append(_safe_div(tp, tp + fp))
        recalls.append(_safe_div(tp, tp + fn))
    return np.array(precisions), np.array(recalls), thresholds


def auprc(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Area under PR curve using the trapezoidal rule (sorted by recall)."""
    order = np.argsort(recalls)
    r = recalls[order]
    p = precisions[order]
    # numpy.trapezoid added in 2.0; numpy.trapz removed in 2.0
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(p, r))
