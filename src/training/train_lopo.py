"""
Stage 3 — LOPO (Leave-One-Protocol-Out) fine-tuning.

Two modes:

  Leakage-free (recommended) — pass --lm_ckpt
    For each held-out protocol P, a fresh Stage-2 main model is trained on
    {Tier1 + Tier2 \\ {P}} before fine-tuning starts.  Validation during
    fine-tuning also never touches P — only Tier-1 val groups are used.
    This is the correct LOPO setup: P is completely unknown to the model
    at every stage (Stage 2 and Stage 3).

  Legacy / fast (leaky) — pass --main_ckpt
    Fine-tuning starts from a single Stage-2 checkpoint that was trained on
    ALL protocols.  The model already saw P during Stage 2.  A UserWarning
    is printed to flag this.

Usage (leakage-free, recommended):
    python -m src.training.train_lopo \\
        --lm_ckpt   checkpoints/lm/best.ckpt \\
        --tier1_dir data/tier1_grammar \\
        --tier2_dir data/tier2_scapy \\
        --ckpt_dir  checkpoints/lopo \\
        --main_epochs 50 --max_steps 2000 --lr 1e-4

Usage (legacy, leaky):
    python -m src.training.train_lopo \\
        --main_ckpt checkpoints/main/best.ckpt \\
        --tier1_dir data/tier1_grammar \\
        --tier2_dir data/tier2_scapy \\
        --ckpt_dir  checkpoints/lopo

Smoke-test a single protocol (leakage-free):
    python -m src.training.train_lopo --fast_dev --protocol dns \\
        --lm_ckpt checkpoints/lm/best.ckpt
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from src.training.dataset import (
    load_tier1_groups, load_tier2_groups, load_tier3_groups,
    make_format_dataloader, LOPO_PROTOCOLS,
)
from src.training.train_main import FieldInferenceModule, train_main

# ── BI benchmark protocol list and Tier-2 name mapping ───────────────────────
# The 10 BI benchmark protocols and their Tier-2 equivalents (if any).
# When holding out a BI protocol we must also exclude its Tier-2 counterpart
# so the model never saw similar data at Stage-2 either.
BI_LOPO_PROTOCOLS: list[str] = [
    "bgp", "dhcp", "dnp3", "mavlink", "mirai",
    "modbus", "ntp48", "smb", "smb2", "tutorial",
]

BI_TO_TIER2: dict[str, str | None] = {
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


# ── Per-protocol Stage-2 helper ───────────────────────────────────────────────

def _best_ckpt_in(directory: Path) -> Path | None:
    """Return the checkpoint with the lowest val/loss_total encoded in its name."""
    candidates = list(directory.rglob("*.ckpt"))
    if not candidates:
        return None

    def _loss(p: Path) -> float:
        # Filename pattern: main-NNN-VAL.ckpt  (val is the raw float)
        try:
            return float(p.stem.rsplit("-", 1)[-1])
        except ValueError:
            return float("inf")

    return min(candidates, key=_loss)


def _train_per_protocol_main(
    held_out:      str,
    lm_ckpt:       str | Path,
    tier1_dir:     Path,
    tier2_dir:     Path,
    ckpt_dir:      Path,
    epochs:        int        = 50,
    n_msgs:        int        = 64,
    max_len:       int        = 512,
    num_workers:   int        = 0,
    fast_dev:      bool       = False,
    tier3_dir:     Path | None = None,
    tier2_exclude: list[str] | None = None,
    tier3_exclude: list[str] | None = None,
) -> Path:
    """
    Train a Stage-2 main model excluding the held-out protocol.
    tier2_exclude: protocol names to drop from Tier-2 (may differ from held_out in BI mode).
    tier3_exclude: protocol names to drop from Tier-3 (the BI benchmark name).
    Saves checkpoints under ckpt_dir/mains/{held_out}/.
    Skips training if a checkpoint already exists there.
    """
    out_dir = ckpt_dir / "mains" / held_out
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _best_ckpt_in(out_dir)
    if existing:
        print(f"  [{held_out}] reusing existing per-protocol main ckpt: {existing}")
        return existing

    t2_excl = tier2_exclude if tier2_exclude is not None else [held_out]
    t3_excl = tier3_exclude if tier3_exclude is not None else [held_out]
    print(f"  [{held_out}] training Stage-2 main (excl T2={t2_excl}, T3={t3_excl}) ...")
    best = train_main(
        lm_ckpt_path=str(lm_ckpt),
        tier1_dir=str(tier1_dir),
        tier2_dir=str(tier2_dir),
        ckpt_dir=str(out_dir),
        epochs=epochs,
        n_msgs=n_msgs,
        max_len=max_len,
        lr=3e-4,
        num_workers=num_workers,
        fast_dev=fast_dev,
        exclude_tier2=t2_excl,
        exclude_tier3=t3_excl,
        tier3_dir=str(tier3_dir) if tier3_dir else None,
    )
    return best


# ── Single-protocol LOPO fine-tune ────────────────────────────────────────────

def _train_one_lopo(
    held_out:      str,
    tier1_dir:     Path,
    tier2_dir:     Path,
    ckpt_dir:      Path,
    # exactly one of these two must be set:
    main_ckpt:     str | Path | None = None,
    lm_ckpt:       str | Path | None = None,
    main_epochs:   int        = 50,
    max_steps:     int        = 2_000,
    n_msgs:        int        = 32,
    max_len:       int        = 512,
    lr:            float      = 1e-4,
    num_workers:   int        = 0,
    fast_dev:      bool       = False,
    tier3_dir:     Path | None = None,
    tier2_exclude: list[str] | None = None,  # what to drop from Tier-2 (BI mode aware)
    tier3_exclude: list[str] | None = None,  # what to drop from Tier-3 (BI protocol name)
) -> Path:
    """Fine-tune for one held-out protocol. Returns best checkpoint path."""

    # Resolve effective exclusion lists (fall back to held_out for standard mode)
    t2_excl = tier2_exclude if tier2_exclude is not None else [held_out]
    t3_excl = tier3_exclude if tier3_exclude is not None else [held_out]

    # ── Resolve Stage-2 starting checkpoint ───────────────────────────────
    if main_ckpt is None:
        main_ckpt = _train_per_protocol_main(
            held_out=held_out,
            lm_ckpt=lm_ckpt,
            tier1_dir=tier1_dir,
            tier2_dir=tier2_dir,
            ckpt_dir=ckpt_dir,
            epochs=main_epochs,
            n_msgs=max(n_msgs, 64),
            max_len=max_len,
            num_workers=num_workers,
            fast_dev=fast_dev,
            tier3_dir=tier3_dir,
            tier2_exclude=t2_excl,
            tier3_exclude=t3_excl,
        )

    # ── Build training set: Tier1 + Tier2 + Tier3 \ {held_out} ───────────
    train_t1, val_t1 = load_tier1_groups(tier1_dir, val_fraction=0.10)
    train_t2, _ = load_tier2_groups(
        tier2_dir,
        exclude_protocols=t2_excl,
    )
    train_t3: list = []
    if tier3_dir:
        train_t3 = load_tier3_groups(tier3_dir, exclude_protocols=t3_excl)
    all_train = train_t1 + train_t2 + train_t3
    print(f"  [{held_out}] train groups: {len(all_train):,}")

    # Validation: Tier-1 val only — held-out protocol never appears here.
    # This ensures checkpoint selection is not guided by held-out performance.
    val_groups = val_t1
    if not val_groups:
        # Edge case: very small tier1; fall back to a small training slice
        val_groups = all_train[: max(1, len(all_train) // 10)]
    print(f"  [{held_out}] val  groups: {len(val_groups):,} (tier-1 only, no leakage)")

    train_dl = make_format_dataloader(
        all_train, n_msgs=n_msgs, max_len=max_len,
        shuffle=True, num_workers=num_workers,
    )
    val_dl = make_format_dataloader(
        val_groups, n_msgs=min(n_msgs, 16), max_len=max_len,
        shuffle=False, num_workers=num_workers,
    )

    # ── Load Stage-2 model and override fine-tuning hparams ───────────────
    model = FieldInferenceModule.load_from_checkpoint(
        str(main_ckpt),
        strict=True,
    )
    model.hparams.lr           = lr
    model.hparams.warmup_steps = min(100, max_steps // 10)
    model.hparams.max_steps    = max(max_steps, 1)
    model.model.unfreeze_lm()

    out_dir = ckpt_dir / held_out
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename=f"lopo-{held_out}-{{step:05d}}-{{val/loss_total:.4f}}",
            monitor="val/loss_total", mode="min", save_top_k=1,
        ),
        LearningRateMonitor(),
    ]
    logger = CSVLogger(
        save_dir=str(ckpt_dir / "logs"),
        name=f"lopo_{held_out}",
    )

    steps_per_epoch = max(1, len(train_dl))
    max_epochs = max(1, (max_steps + steps_per_epoch - 1) // steps_per_epoch)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if fast_dev else max_epochs,
        max_steps=5  if fast_dev else max_steps,
        limit_train_batches=5 if fast_dev else 1.0,
        limit_val_batches=5   if fast_dev else 1.0,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=max(1, min(50, steps_per_epoch // 4)),
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dl, val_dl)

    best = callbacks[0].best_model_path
    if not best:
        best = str(out_dir / "last.ckpt")
        trainer.save_checkpoint(best)
    print(f"  [{held_out}] best ckpt: {best}")
    return Path(best)


# ── Full LOPO loop ────────────────────────────────────────────────────────────

def train_lopo(
    main_ckpt:   str | None = None,
    lm_ckpt:     str | None = None,
    tier1_dir:   str = "data/tier1_grammar",
    tier2_dir:   str = "data/tier2_scapy",
    tier3_dir:   str | None = None,
    ckpt_dir:    str = "checkpoints/lopo",
    main_epochs: int   = 50,
    max_steps:   int   = 2_000,
    n_msgs:      int   = 32,
    max_len:     int   = 512,
    lr:          float = 1e-4,
    num_workers: int   = 0,
    fast_dev:    bool  = False,
    protocol:    str | None = None,
    bi_mode:     bool  = False,
) -> dict[str, Path]:
    """
    Run LOPO fine-tuning.

    Standard mode (bi_mode=False): iterates over LOPO_PROTOCOLS (Tier-2 names).
    BI mode (bi_mode=True): iterates over BI_LOPO_PROTOCOLS (10 BI benchmark
      names), correctly excluding the mapped Tier-2 protocol AND the Tier-3
      protocol by BI name.

    Pass --lm_ckpt for leakage-free LOPO (recommended).
    Pass --main_ckpt for a quick legacy run (UserWarning issued).
    """
    if main_ckpt is None and lm_ckpt is None:
        raise ValueError(
            "Provide either --main_ckpt (legacy, leaky) or "
            "--lm_ckpt (leakage-free, recommended)."
        )
    if main_ckpt is not None and lm_ckpt is not None:
        raise ValueError("Provide --main_ckpt OR --lm_ckpt, not both.")

    if main_ckpt is not None:
        warnings.warn(
            "Using a shared --main_ckpt that was trained on ALL protocols "
            "introduces data leakage into LOPO evaluation: the model has already "
            "seen each held-out protocol during Stage-2 training. "
            "Pass --lm_ckpt instead to train leakage-free per-protocol "
            "Stage-2 models.",
            UserWarning,
            stacklevel=2,
        )

    tier1_dir = Path(tier1_dir)
    tier2_dir = Path(tier2_dir)
    ckpt_dir  = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if bi_mode:
        protocols = [protocol] if protocol else BI_LOPO_PROTOCOLS
    else:
        protocols = [protocol] if protocol else LOPO_PROTOCOLS

    results: dict[str, Path] = {}
    for proto in protocols:
        print(f"\n=== LOPO: held-out = {proto} ===")

        if bi_mode:
            t2_name = BI_TO_TIER2.get(proto)
            t2_excl = [t2_name] if t2_name else []
            t3_excl = [proto]
            print(f"  excl Tier-2={t2_excl}  excl Tier-3={t3_excl}")
        else:
            t2_excl = t3_excl = None  # _train_one_lopo defaults to [held_out]

        best = _train_one_lopo(
            held_out=proto,
            tier1_dir=tier1_dir,
            tier2_dir=tier2_dir,
            ckpt_dir=ckpt_dir,
            main_ckpt=main_ckpt,
            lm_ckpt=lm_ckpt,
            main_epochs=main_epochs,
            max_steps=max_steps,
            n_msgs=n_msgs,
            max_len=max_len,
            lr=lr,
            num_workers=num_workers,
            fast_dev=fast_dev,
            tier3_dir=Path(tier3_dir) if tier3_dir else None,
            tier2_exclude=t2_excl,
            tier3_exclude=t3_excl,
        )
        results[proto] = best

    print(f"\nLOPO complete. {len(results)} checkpoints saved.")
    for proto, ckpt in results.items():
        print(f"  {proto:12s}: {ckpt}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 3: LOPO fine-tuning. "
                    "Use --lm_ckpt for leakage-free evaluation (recommended), "
                    "or --main_ckpt for the legacy (leaky) mode.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--lm_ckpt",
        metavar="PATH",
        help="Stage-1 LM checkpoint. Triggers per-protocol Stage-2 training "
             "(leakage-free, recommended).",
    )
    mode.add_argument(
        "--main_ckpt",
        metavar="PATH",
        help="Single shared Stage-2 checkpoint (legacy, leaky — the model already "
             "saw each held-out protocol during Stage-2 training).",
    )
    p.add_argument("--tier1_dir",   default="data/tier1_grammar")
    p.add_argument("--tier2_dir",   default="data/tier2_scapy")
    p.add_argument("--tier3_dir",   default=None,
                   help="Path to tier3_labeled (BI benchmark labeled data). "
                        "9 protocols used per fold, held-out excluded.")
    p.add_argument("--ckpt_dir",    default="checkpoints/lopo")
    p.add_argument("--main_epochs", type=int,   default=50,
                   help="Epochs for per-protocol Stage-2 training (--lm_ckpt mode only).")
    p.add_argument("--max_steps",   type=int,   default=2_000)
    p.add_argument("--n_msgs",      type=int,   default=32)
    p.add_argument("--max_len",     type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--fast_dev",  action="store_true")
    p.add_argument("--bi_mode",   action="store_true",
                   help="Iterate over the 10 BI benchmark protocols with correct "
                        "Tier-2/Tier-3 name mapping (use with --tier3_dir).")
    p.add_argument("--protocol",  default=None,
                   help="Run only this protocol (debug / smoke test).")
    args = p.parse_args()
    train_lopo(**vars(args))
