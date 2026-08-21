"""Compare produced CSVs against results/reference/ (4 decimal places)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

METRIC_KEYS_LOPO = ("P", "R", "FPR", "F1")
METRIC_KEYS_L4PO = ("precision", "recall", "fpr", "f1")
METRIC_KEYS_AGG = ("P_mean", "R_mean", "FPR_mean", "F1_mean")


def _load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _close(a: str, b: str, tol: float = 5e-5) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _compare_rows(
    name: str,
    got_rows: list[dict],
    ref_rows: list[dict],
    key: str,
    metrics: tuple[str, ...],
) -> list[str]:
    mismatches: list[str] = []
    ref_by = {r[key]: r for r in ref_rows}
    got_by = {r[key]: r for r in got_rows}
    if set(ref_by) != set(got_by):
        mismatches.append(
            f"{name}: protocol set mismatch  "
            f"missing={sorted(set(ref_by) - set(got_by))}  "
            f"extra={sorted(set(got_by) - set(ref_by))}"
        )
    for proto in sorted(set(ref_by) & set(got_by)):
        for m in metrics:
            if m not in ref_by[proto] or m not in got_by[proto]:
                continue
            if not _close(got_by[proto][m], ref_by[proto][m]):
                mismatches.append(
                    f"{name}/{proto}/{m}: got {got_by[proto][m]}  ref {ref_by[proto][m]}"
                )
    return mismatches


def compare(got_dir: Path, ref_dir: Path) -> int:
    if not ref_dir.is_dir():
        print(f"No reference tables at {ref_dir} — skip comparison.")
        return 0

    mismatches: list[str] = []

    lopo_got = got_dir / "lopo_results.csv"
    lopo_ref = ref_dir / "lopo_results.csv"
    if lopo_got.exists() and lopo_ref.exists():
        mismatches.extend(
            _compare_rows(
                "LOPO",
                _load(lopo_got),
                _load(lopo_ref),
                "protocol",
                METRIC_KEYS_LOPO,
            )
        )
    else:
        mismatches.append(f"missing LOPO csv (got={lopo_got.exists()} ref={lopo_ref.exists()})")

    l4po_got = got_dir / "l4po"
    l4po_ref = ref_dir / "l4po"
    ref_csvs = sorted(p for p in l4po_ref.glob("l4po_*.csv") if p.name != "l4po_aggregate.csv")
    for ref_p in ref_csvs:
        got_p = l4po_got / ref_p.name
        if not got_p.exists():
            mismatches.append(f"missing {got_p}")
            continue
        mismatches.extend(
            _compare_rows(
                ref_p.stem,
                _load(got_p),
                _load(ref_p),
                "protocol",
                METRIC_KEYS_L4PO,
            )
        )

    agg_got = l4po_got / "l4po_aggregate.csv"
    agg_ref = l4po_ref / "l4po_aggregate.csv"
    if agg_got.exists() and agg_ref.exists():
        mismatches.extend(
            _compare_rows(
                "L4PO-agg",
                _load(agg_got),
                _load(agg_ref),
                "protocol",
                METRIC_KEYS_AGG,
            )
        )

    if mismatches:
        print(f"MISMATCH ({len(mismatches)}):")
        for line in mismatches:
            print("  ", line)
        return 1

    print("OK — results match results/reference/ (4 d.p.).")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--got", default="results/seed789")
    p.add_argument("--ref", default="results/reference")
    args = p.parse_args()
    sys.exit(compare(Path(args.got), Path(args.ref)))
