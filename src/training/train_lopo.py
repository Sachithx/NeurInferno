"""
Stage 3 — LOPO (Leave-One-Protocol-Out) fine-tuning.

Leakage-free (pass --lm_ckpt): for each held-out protocol P, a fresh Stage-2
main model is trained on {grammar + protocols \\ {P}} before fine-tuning.
Validation during fine-tuning uses grammar val groups only, so P is unknown
at every stage.

Legacy / leaky (pass --main_ckpt): fine-tuning starts from a shared Stage-2
checkpoint trained on all protocols.

Usage:
    python -m src.training.train_lopo \
        --lm_ckpt   checkpoints/lm/<proto>/bylm-....ckpt \
        --protocol  dns \
        --tier1_dir data/grammar \
        --tier2_dir data/protocols \
        --ckpt_dir  checkpoints/lopo
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from src.training.dataset import (
    load_tier1_groups, load_tier2_groups,
    split_tier2_protocols,
    make_format_dataloader, LOPO_PROTOCOLS,
)
from src.training.train_main import FieldInferenceModule, train_main


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
    tier2_exclude: list[str] | None = None,
    d_model:       int        = 128,
    n_heads:       int        = 4,
    d_ff:          int        = 512,
    n_layers:      int        = 4,
    lm_d_model:               int        = 64,
    lm_n_heads:               int        = 4,
    lm_d_ff:                  int        = 256,
    lm_n_layers:              int        = 2,
    use_relational_type_head: bool       = False,
    use_focal_type:           bool       = False,
    tier2_val_protocols:      list[str] | None = None,
    tier2_train_val_fraction: float            = 0.0,
) -> Path:
    """
    Train a Stage-2 main model excluding the held-out protocol.
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
    print(f"  [{held_out}] training Stage-2 main (excl={t2_excl}) ...")
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
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
        lm_d_model=lm_d_model, lm_n_heads=lm_n_heads,
        lm_d_ff=lm_d_ff, lm_n_layers=lm_n_layers,
        use_relational_type_head=use_relational_type_head,
        use_focal_type=use_focal_type,
        tier2_val_protocols=tier2_val_protocols,
        tier2_train_val_fraction=tier2_train_val_fraction,
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
    tier2_exclude: list[str] | None = None,
    d_model:       int        = 128,
    n_heads:       int        = 4,
    d_ff:          int        = 512,
    n_layers:      int        = 4,
    lm_d_model:               int        = 64,
    lm_n_heads:               int        = 4,
    lm_d_ff:                  int        = 256,
    lm_n_layers:              int        = 2,
    use_relational_type_head: bool       = False,
    use_focal_type:           bool       = False,
    use_dynamic_val:          bool       = False,
    n_val_protocols:          int        = 6,
) -> Path:
    """Fine-tune for one held-out protocol. Returns best checkpoint path."""

    t2_excl = tier2_exclude if tier2_exclude is not None else [held_out]

    # ── Compute dynamic T2 val split (consistent across Stage 2 and Stage 3) ─
    t2_val_protos: list[str] | None = None
    t2_train_val_frac: float = 0.0
    if use_dynamic_val:
        _, t2_val_protos = split_tier2_protocols(
            tier2_dir, t2_excl, n_val=n_val_protocols,
        )
        t2_train_val_frac = 0.10
        print(f"  [{held_out}] dynamic val — T2 val protocols: {sorted(t2_val_protos)}")

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
            tier2_exclude=t2_excl,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
            lm_d_model=lm_d_model, lm_n_heads=lm_n_heads,
            lm_d_ff=lm_d_ff, lm_n_layers=lm_n_layers,
            use_relational_type_head=use_relational_type_head,
            use_focal_type=use_focal_type,
            tier2_val_protocols=t2_val_protos,
            tier2_train_val_fraction=t2_train_val_frac,
        )

    # ── Build training set: grammar + protocols \ {held_out} ───────────
    train_t1, val_t1 = load_tier1_groups(tier1_dir, val_fraction=0.10)
    train_t2, _ = load_tier2_groups(
        tier2_dir,
        val_protocols=t2_val_protos,
        exclude_protocols=t2_excl,
        test_fraction=0.0 if t2_val_protos is not None else 0.20,
        train_val_fraction=0.0,   # Stage 3 FT trains on all train-protocol messages
    )
    all_train = train_t1 + train_t2
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
    model.model.unfreeze_lm()   # allow LM entropy features to adapt during fine-tune

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
    tier1_dir:   str = "data/grammar",
    tier2_dir:   str = "data/protocols",
    ckpt_dir:    str = "checkpoints/lopo",
    main_epochs: int   = 50,
    max_steps:   int   = 2_000,
    n_msgs:      int   = 32,
    max_len:     int   = 512,
    lr:          float = 1e-4,
    num_workers: int   = 0,
    fast_dev:    bool  = False,
    protocol:    str | None = None,
    d_model:     int   = 128,
    n_heads:     int   = 4,
    d_ff:        int   = 512,
    n_layers:    int   = 4,
    lm_d_model:               int   = 64,
    lm_n_heads:               int   = 4,
    lm_d_ff:                  int   = 256,
    lm_n_layers:              int   = 2,
    use_relational_type_head: bool  = False,
    use_focal_type:           bool  = False,
    use_dynamic_val:          bool  = False,
    n_val_protocols:          int   = 6,
    seed:                     int   = 42,
) -> dict[str, Path]:
    """
    Run LOPO fine-tuning over LOPO_PROTOCOLS (or a single --protocol).

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

    pl.seed_everything(seed, workers=True)
    tier1_dir = Path(tier1_dir)
    tier2_dir = Path(tier2_dir)
    ckpt_dir  = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    protocols = [protocol] if protocol else LOPO_PROTOCOLS

    results: dict[str, Path] = {}
    for proto in protocols:
        print(f"\n=== LOPO: held-out = {proto} ===")

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
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
            lm_d_model=lm_d_model, lm_n_heads=lm_n_heads,
            lm_d_ff=lm_d_ff, lm_n_layers=lm_n_layers,
            use_relational_type_head=use_relational_type_head,
            use_focal_type=use_focal_type,
            use_dynamic_val=use_dynamic_val,
            n_val_protocols=n_val_protocols,
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
    p.add_argument("--tier1_dir",   default="data/grammar")
    p.add_argument("--tier2_dir",   default="data/protocols")
    p.add_argument("--ckpt_dir",    default="checkpoints/lopo")
    p.add_argument("--main_epochs", type=int,   default=50,
                   help="Epochs for per-protocol Stage-2 training (--lm_ckpt mode only).")
    p.add_argument("--max_steps",   type=int,   default=2_000)
    p.add_argument("--n_msgs",      type=int,   default=32)
    p.add_argument("--max_len",     type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--fast_dev",  action="store_true")
    p.add_argument("--protocol",  default=None)
    p.add_argument("--d_model",    type=int, default=128)
    p.add_argument("--n_heads",    type=int, default=4)
    p.add_argument("--d_ff",       type=int, default=512)
    p.add_argument("--n_layers",   type=int, default=4)
    p.add_argument("--lm_d_model", type=int, default=64)
    p.add_argument("--lm_n_heads", type=int, default=4)
    p.add_argument("--lm_d_ff",    type=int, default=256)
    p.add_argument("--lm_n_layers",              type=int, default=2)
    p.add_argument("--use_relational_type_head", action="store_true")
    p.add_argument("--use_focal_type",           action="store_true")
    p.add_argument("--use_dynamic_val",          action="store_true",
                   help="Randomly split T2 into train/val instead of fixed coap/ospf.")
    p.add_argument("--n_val_protocols",          type=int, default=6,
                   help="T2 val protocols: 6 for LOPO (16 remain), 3 for L4PO (13 remain).")
    p.add_argument("--seed",                     type=int, default=42,
                   help="Global random seed for reproducibility.")
    args = p.parse_args()
    train_lopo(**vars(args))
