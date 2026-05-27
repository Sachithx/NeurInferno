"""
field_mdl.py — GT-free criterion for selecting the best boundary set.

Two complementary scores:

1. surprise_fratio(mean_s, bounds)
   Fast O(L) criterion operating on the precomputed mean-surprise signal.
   An ANOVA F-ratio measuring how well the segmentation explains the
   surprise signal: high between-field surprise variation + low within-field
   variation → good boundaries.  Penalised for over-segmentation.

2. field_mdl_score(messages, bounds, lam)
   Exact MDL/BIC score operating on raw message bytes.
   score(B) = -Σ_{m,f,p} log P̂(byte | field=f, pos=p)
              + lam·|B|·log N
   Lower is better.  The first term rewards field-position distributions that
   are concentrated (predictable); the penalty prevents over-segmentation.
   Bayes-consistent under i.i.d. field-byte model.

   NOTE: for fixed-length messages, the per-(field,position) distributions
   are identical to per-position distributions regardless of segmentation,
   so the code-length term is constant and the penalty always favours fewer
   boundaries.  field_mdl_score is most discriminative on variable-length
   message corpora where the same relative positions appear in different field
   slots across messages.  For fixed-format binary protocols, prefer
   surprise_fratio which operates on the learned surprise signal.

3. select_by_fratio(mean_s, std_s, candidate_bounds_list)
   Picks the best boundary set from a list using surprise_fratio.

4. select_by_mdl(messages, candidate_bounds_list, lam)
   Picks the best boundary set from a list using field_mdl_score.
"""

import numpy as np
from collections import defaultdict


# ── Surprise F-ratio ──────────────────────────────────────────────────────────

def surprise_fratio(
    mean_s: np.ndarray,
    bounds: set[int],
    lam: float = 1.0,
) -> float:
    """
    Adjusted-R² score of the surprise signal under the given segmentation.
    Higher is better.

    R² = SS_between / SS_total  (fraction of surprise variance explained by
    the segmentation — bounded in [0, 1]).

    Penalised: score = R² − lam · k / N
    where k = number of segments and N = number of valid positions.
    The penalty discourages over-segmentation without the F-ratio's blow-up
    when segments shrink to size 1.

    lam=1.0 means a fully over-segmented signal (k=N singletons, R²=1) scores
    exactly 0, ensuring any meaningful segmentation beats it as long as it
    explains even a fraction of variance with fewer than N segments.

    Returns 0 for degenerate inputs (empty signal, no valid positions,
    no variance in the signal).
    """
    valid_mask = ~np.isnan(mean_s)
    valid_vals = mean_s[valid_mask]
    N = len(valid_vals)
    if N < 2:
        return 0.0

    ss_total = float(np.sum((valid_vals - valid_vals.mean()) ** 2))
    if ss_total < 1e-12:
        return 0.0

    # Boundary positions clipped to signal length, position 0 excluded
    split_pts = sorted(b for b in bounds if 0 < b < len(mean_s))
    edges = [0] + split_pts + [len(mean_s)]
    n_segs = len(edges) - 1

    if n_segs < 2:
        penalty = lam * 1 / N
        return 0.0 - penalty   # no segmentation: R²=0, still penalise 1 segment

    global_mean = valid_vals.mean()
    ss_between = 0.0

    for i in range(n_segs):
        seg = mean_s[edges[i]:edges[i + 1]]
        seg_valid = seg[~np.isnan(seg)]
        if len(seg_valid) == 0:
            continue
        seg_mean   = seg_valid.mean()
        ss_between += len(seg_valid) * (seg_mean - global_mean) ** 2

    r_squared = ss_between / ss_total          # ∈ [0, 1]
    penalty   = lam * n_segs / N              # proportional to segments per position
    return float(r_squared - penalty)


# ── MDL / BIC field score ─────────────────────────────────────────────────────

def field_mdl_score(
    messages: list[bytes],
    bounds: set[int],
    lam: float = 1.0,
) -> float:
    """
    MDL/BIC score for a candidate boundary set — lower is better.

    Splits each message at the given boundaries, treating each resulting
    segment as a named 'field'.  For each (field_index, within-field position)
    slot, accumulates an empirical byte distribution and computes the
    negative log-likelihood under that distribution.

    BIC-style penalty: lam * |B| * log(N) discourages over-segmentation.

    Complexity: O(N · L) per call (N messages, L max length).
    """
    N = len(messages)
    if N == 0:
        return float('inf')

    splits = sorted(b for b in bounds if b > 0)

    # Accumulate per-(field_idx, within_field_pos) byte counts
    # Using flat numpy arrays keyed by (fi, pi) → index into a 256-bucket table
    # For speed: collect all byte values per slot, then count once.
    slot_bytes: dict[tuple[int, int], list[int]] = defaultdict(list)

    for msg in messages:
        prev = 0
        for fi, bp in enumerate(splits + [len(msg)]):
            end = min(bp, len(msg))
            for pi in range(end - prev):
                slot_bytes[(fi, pi)].append(msg[prev + pi])
            prev = end
            if prev >= len(msg):
                break

    # Build count arrays and compute NLL
    slot_counts: dict[tuple[int, int], np.ndarray] = {}
    slot_totals: dict[tuple[int, int], int] = {}
    for key, byte_list in slot_bytes.items():
        arr = np.bincount(byte_list, minlength=256).astype(np.float64)
        slot_counts[key] = arr
        slot_totals[key] = int(arr.sum())

    # NLL: −Σ count_b · log(count_b / total)
    nll = 0.0
    for key, arr in slot_counts.items():
        total = slot_totals[key]
        if total == 0:
            continue
        probs = arr / total
        # Only non-zero entries contribute
        mask = arr > 0
        nll -= float(np.dot(arr[mask], np.log(probs[mask])))

    penalty = lam * len(bounds) * np.log(N)
    return nll + penalty


# ── Batch selectors ───────────────────────────────────────────────────────────

def select_by_fratio(
    mean_s: np.ndarray,
    candidate_bounds_list: list[set[int]],
    lam: float = 0.5,
) -> tuple[set[int], float]:
    """
    Choose the best boundary set from a pool using surprise_fratio.
    Returns (best_bounds, best_score).  Empty set returned if pool is empty.
    """
    best_bounds: set[int] = set()
    best_score = -float('inf')
    for bounds in candidate_bounds_list:
        score = surprise_fratio(mean_s, bounds, lam=lam)
        if score > best_score:
            best_score, best_bounds = score, bounds
    return best_bounds, best_score


def select_by_mdl(
    messages: list[bytes],
    candidate_bounds_list: list[set[int]],
    lam: float = 1.0,
) -> tuple[set[int], float]:
    """
    Choose the best boundary set from a pool using field_mdl_score.
    Returns (best_bounds, best_score).  Empty set returned if pool is empty.
    """
    best_bounds: set[int] = set()
    best_score = float('inf')
    for bounds in candidate_bounds_list:
        score = field_mdl_score(messages, bounds, lam=lam)
        if score < best_score:
            best_score, best_bounds = score, bounds
    return best_bounds, best_score
