"""
Leave-N-Protocol-Out (L4PO) training and evaluation on Tier-2 protocols.

Randomly selects N protocols from LOPO_PROTOCOLS (or accepts an explicit list),
trains ONE model on the remaining protocols, then evaluates on each held-out
protocol using its Tier-2 test split.

Usage — random selection (seed fixes which 4 are held out, reproducible):
    python -m neurinferno.training.train_l4po \
        --lm_ckpt   checkpoints/lm/bylm-epoch=00-val/loss=4.8214.ckpt \
        --tier1_dir data/grammar \
        --tier2_dir data/protocols \
        --ckpt_dir  checkpoints/l4po \
        --results_dir results/l4po \
        --seed 42 \
        --main_epochs 50 --max_steps 2000 --lr 1e-4 --n_msg 100

Usage — explicit held-out set:
    python -m neurinferno.training.train_l4po \
        --lm_ckpt   checkpoints/lm/<label>/bylm-....ckpt \
        --held_out  bgp_raw modbus tcp ip \
        --tier1_dir data/grammar \
        --tier2_dir data/protocols \
        --ckpt_dir  checkpoints/l4po \
        --results_dir results/l4po
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pytorch_lightning as pl
import torch

from neurinferno.evaluation.ground_truth import load_labeled_messages
from neurinferno.evaluation.lopo import run_model_inference
from neurinferno.evaluation.metrics import aggregate_metrics, compute_boundary_metrics
from neurinferno.training.dataset import LOPO_PROTOCOLS
from neurinferno.training.train_lopo import _train_one_lopo
from neurinferno.training.train_main import FieldInferenceModule

# ── Protocol selection ────────────────────────────────────────────────────────


def select_held_out(
    held_out: list[str] | None,
    seed: int | None,
    n: int = 4,
) -> list[str]:
    """Return the list of held-out Tier-2 protocol names.

    If held_out is given explicitly, validate against LOPO_PROTOCOLS.
    Otherwise randomly draw n from LOPO_PROTOCOLS using seed.
    """
    if held_out:
        unknown = set(held_out) - set(LOPO_PROTOCOLS)
        if unknown:
            raise ValueError(
                f"Unknown protocol(s): {unknown}.\nMust be from LOPO_PROTOCOLS: {LOPO_PROTOCOLS}"
            )
        return list(held_out)

    rng = random.Random(seed)
    chosen = rng.sample(LOPO_PROTOCOLS, n)
    print(f"Randomly selected {n} held-out protocols (seed={seed}): {chosen}")
    return chosen


# ── Training ──────────────────────────────────────────────────────────────────


def train_l4po(
    lm_ckpt: str,
    held_out: list[str],
    tier1_dir: str = "data/grammar",
    tier2_dir: str = "data/protocols",
    ckpt_dir: str = "checkpoints/l4po",
    main_epochs: int = 50,
    max_steps: int = 2_000,
    n_msgs: int = 32,
    max_len: int = 512,
    lr: float = 1e-4,
    num_workers: int = 0,
    fast_dev: bool = False,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 512,
    n_layers: int = 4,
    lm_d_model: int = 64,
    lm_n_heads: int = 4,
    lm_d_ff: int = 256,
    lm_n_layers: int = 2,
    use_dynamic_val: bool = False,
    n_val_protocols: int = 3,
    seed: int = 42,
) -> Path:
    """
    Train one model on all LOPO protocols except held_out.
    Returns the path to the best fine-tuned checkpoint.
    """
    pl.seed_everything(seed, workers=True)
    run_label = "_".join(sorted(held_out))

    print("\n=== L4PO training ===")
    print(f"  held-out  : {held_out}")
    print(f"  run label : {run_label}")

    best = _train_one_lopo(
        held_out=run_label,
        tier1_dir=Path(tier1_dir),
        tier2_dir=Path(tier2_dir),
        ckpt_dir=Path(ckpt_dir),
        lm_ckpt=lm_ckpt,
        main_epochs=main_epochs,
        max_steps=max_steps,
        n_msgs=n_msgs,
        max_len=max_len,
        lr=lr,
        num_workers=num_workers,
        fast_dev=fast_dev,
        tier2_exclude=held_out,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        lm_d_model=lm_d_model,
        lm_n_heads=lm_n_heads,
        lm_d_ff=lm_d_ff,
        lm_n_layers=lm_n_layers,
        use_dynamic_val=use_dynamic_val,
        n_val_protocols=n_val_protocols,
    )
    print(f"\nL4PO training complete. Checkpoint: {best}")
    return Path(best)


# ── Evaluation ────────────────────────────────────────────────────────────────


def eval_l4po(
    ckpt_path: Path,
    held_out: list[str],
    tier2_dir: str = "data/protocols",
    results_dir: str = "results/l4po",
    max_msgs: int = 1000,
    n_msgs_batch: int = 32,
    max_len: int = 512,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Evaluate the L4PO checkpoint on each held-out Tier-2 protocol."""
    tier2_dir = Path(tier2_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading checkpoint: {ckpt_path}")
    model = FieldInferenceModule.load_from_checkpoint(str(ckpt_path), strict=True)
    model.eval()
    model.to(device)

    all_results: dict[str, dict] = {}

    print(f"\n=== L4PO evaluation on {len(held_out)} held-out protocols ===")
    for proto in held_out:
        print(f"\n[{proto}]", end="  ")
        try:
            msgs = load_labeled_messages(tier2_dir, proto, max_msgs=max_msgs)
        except FileNotFoundError as e:
            print(f"SKIP: {e}")
            continue

        print(f"{len(msgs)} msgs", end="  ")

        scores_list, gt_list = run_model_inference(
            model,
            msgs,
            n_msgs=n_msgs_batch,
            max_len=max_len,
            device=device,
        )
        per_msg = [
            compute_boundary_metrics([1 if s >= threshold else 0 for s in sc], gt)
            for sc, gt in zip(scores_list, gt_list, strict=True)
        ]
        agg = aggregate_metrics(per_msg)
        all_results[proto] = {
            "precision": agg.precision,
            "recall": agg.recall,
            "fpr": agg.fpr,
            "f1": agg.f1,
            "n_messages": len(msgs),
        }
        print(f"P={agg.precision:.3f}  R={agg.recall:.3f}  FPR={agg.fpr:.3f}  F1={agg.f1:.3f}")

    _write_results(all_results, held_out, results_dir, ckpt_path)
    return all_results


def _write_results(
    results: dict[str, dict],
    held_out: list[str],
    results_dir: Path,
    ckpt_path: Path,
) -> None:
    run_label = "_".join(sorted(held_out))
    csv_path = results_dir / f"l4po_{run_label}.csv"

    data_rows = [
        {
            "protocol": proto,
            "precision": f"{m['precision']:.4f}",
            "recall": f"{m['recall']:.4f}",
            "fpr": f"{m['fpr']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "n_messages": int(m["n_messages"]),
            "checkpoint": str(ckpt_path),
        }
        for proto, m in results.items()
    ]
    rows = list(data_rows)

    if data_rows:
        avgs = {
            k: sum(float(r[k]) for r in data_rows) / len(data_rows)
            for k in ("precision", "recall", "fpr", "f1")
        }
        rows.append(
            {
                "protocol": f"AVERAGE ({len(data_rows)} protocols)",
                "precision": f"{avgs['precision']:.4f}",
                "recall": f"{avgs['recall']:.4f}",
                "fpr": f"{avgs['fpr']:.4f}",
                "f1": f"{avgs['f1']:.4f}",
                "n_messages": sum(int(r["n_messages"]) for r in data_rows),
                "checkpoint": "",
            }
        )

    fieldnames = ["protocol", "precision", "recall", "fpr", "f1", "n_messages", "checkpoint"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {csv_path}")

    print(f"\n{'Protocol':<14} {'P':>7} {'R':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 44)
    for r in rows:
        print(
            f"{r['protocol']:<14} {r['precision']:>7} {r['recall']:>7} {r['fpr']:>7} {r['f1']:>7}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Leave-N-Protocol-Out training + Tier-2 evaluation. "
        "Trains one model on remaining protocols, evaluates on held-out.",
    )
    p.add_argument(
        "--lm_ckpt",
        required=True,
        metavar="PATH",
        help="Stage-1 LM checkpoint (leakage-free).",
    )
    held_group = p.add_mutually_exclusive_group()
    held_group.add_argument(
        "--held_out",
        nargs="+",
        metavar="PROTO",
        help=f"Explicit held-out Tier-2 protocol names. Choose from: {LOPO_PROTOCOLS}",
    )
    held_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting n_held_out protocols (default: 42).",
    )
    p.add_argument(
        "--n_held_out",
        type=int,
        default=4,
        help="Number of protocols to hold out when using --seed (default: 4).",
    )
    p.add_argument("--tier1_dir", default="data/grammar")
    p.add_argument("--tier2_dir", default="data/protocols")
    p.add_argument("--ckpt_dir", default="checkpoints/l4po")
    p.add_argument("--results_dir", default="results/l4po")
    p.add_argument("--main_epochs", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=2_000)
    p.add_argument("--n_msg", type=int, default=32, dest="n_msgs")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--fast_dev", action="store_true")
    p.add_argument("--eval_only", metavar="CKPT", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n_msgs_batch", type=int, default=32)
    p.add_argument("--max_msgs", type=int, default=1000)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--lm_d_model", type=int, default=64)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_d_ff", type=int, default=256)
    p.add_argument("--lm_n_layers", type=int, default=2)
    p.add_argument(
        "--use_dynamic_val",
        action="store_true",
        help="Randomly split T2 into train/val instead of fixed coap/ospf.",
    )
    p.add_argument(
        "--n_val_protocols",
        type=int,
        default=3,
        help="T2 val protocols for L4PO (13 remain after 4 excluded → 3 val).",
    )
    p.add_argument("--use_relational_type_head", action="store_true")
    p.add_argument("--use_focal_type", action="store_true")
    p.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training (use run_eval.py instead).",
    )
    p.add_argument(
        "--train_seed",
        type=int,
        default=42,
        help="Global random seed for training reproducibility.",
    )
    args = p.parse_args()

    held_out = select_held_out(args.held_out, args.seed, args.n_held_out)

    if args.eval_only:
        ckpt_path = Path(args.eval_only)
    else:
        ckpt_path = train_l4po(
            lm_ckpt=args.lm_ckpt,
            held_out=held_out,
            tier1_dir=args.tier1_dir,
            tier2_dir=args.tier2_dir,
            ckpt_dir=args.ckpt_dir,
            main_epochs=args.main_epochs,
            max_steps=args.max_steps,
            n_msgs=args.n_msgs,
            max_len=args.max_len,
            lr=args.lr,
            num_workers=args.num_workers,
            fast_dev=args.fast_dev,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            lm_d_model=args.lm_d_model,
            lm_n_heads=args.lm_n_heads,
            lm_d_ff=args.lm_d_ff,
            lm_n_layers=args.lm_n_layers,
            use_dynamic_val=args.use_dynamic_val,
            n_val_protocols=args.n_val_protocols,
            seed=args.train_seed,
        )

    if not args.skip_eval:
        eval_l4po(
            ckpt_path=ckpt_path,
            held_out=held_out,
            tier2_dir=args.tier2_dir,
            results_dir=args.results_dir,
            max_msgs=args.max_msgs,
            n_msgs_batch=args.n_msgs_batch,
            max_len=args.max_len,
            threshold=args.threshold,
        )
