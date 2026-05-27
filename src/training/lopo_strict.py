"""
Stage 3 — Strict LOPO (protocol-wise Leave-One-Protocol-Out) training.

For each of the LOPO protocols, this script re-runs Stage 2 from scratch
with that protocol excluded from training data, optionally re-pretrains
the ByteLM with the same exclusion, and optionally does a light final
fine-tune.

Key difference from the previous train_lopo.py:
    The previous version started from a Stage 2 checkpoint that had seen
    all protocols, and only excluded the held-out protocol during a final
    fine-tuning pass.  That measures message-level generalization within
    a known protocol family, not protocol-level generalization.

    This version retrains Stage 2 per holdout so the model never sees the
    held-out protocol at any training stage.

Usage:
    # Full strict LOPO (10 Stage-2 runs, shared LM)
    python -m src.training.train_lopo_strict \
        --lm_ckpt    checkpoints/lm/best.ckpt \
        --tier1_dir  data/tier1_grammar \
        --tier2_dir  data/tier2_scapy \
        --ckpt_dir   checkpoints/lopo_strict \
        --stage2_steps 20000 --finetune_steps 0

    # Strict LOPO + per-holdout LM retrain (20 runs total, slowest)
    python -m src.training.train_lopo_strict \
        --retrain_lm \
        --tier1_dir  data/tier1_grammar \
        --tier2_dir  data/tier2_scapy \
        --ckpt_dir   checkpoints/lopo_strict \
        --stage2_steps 20000 --lm_steps 10000

    # Smoke test on one protocol
    python -m src.training.train_lopo_strict \
        --lm_ckpt    checkpoints/lm/best.ckpt \
        --protocol   dns --fast_dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from src.training.dataset import (
    load_tier1_groups, load_tier2_groups,
    make_format_dataloader, LOPO_PROTOCOLS,
)
from src.training.train_main import FieldInferenceModule
from src.training.train_lm import ByteLMModule, make_lm_dataloader


# ---------------------------------------------------------------------------
# Per-holdout ByteLM pretraining (optional)
# ---------------------------------------------------------------------------

def _pretrain_lm_for_holdout(
    held_out:    str,
    tier1_dir:   Path,
    tier2_dir:   Path,
    ckpt_dir:    Path,
    max_steps:   int  = 10_000,
    batch_size:  int  = 64,
    max_len:     int  = 256,
    lr:          float = 3e-4,
    num_workers: int  = 0,
    fast_dev:    bool = False,
) -> Path:
    """Re-pretrain ByteLM with `held_out` protocol fully excluded.

    Returns path to best checkpoint.
    """
    out_dir = ckpt_dir / held_out / "lm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load flat byte sequences for LM training, with exclusion
    train_dl = make_lm_dataloader(
        tier1_dir=tier1_dir,
        tier2_dir=tier2_dir,
        exclude_protocols=[held_out],
        split="train",
        batch_size=batch_size,
        max_len=max_len,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dl = make_lm_dataloader(
        tier1_dir=tier1_dir,
        tier2_dir=tier2_dir,
        exclude_protocols=[held_out],
        split="val",
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False,
        num_workers=num_workers,
    )

    model = ByteLMModule(lr=lr, max_steps=max_steps,
                         warmup_steps=min(500, max_steps // 20))

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename=f"lm-{held_out}-{{step:06d}}-{{val/loss:.4f}}",
            monitor="val/loss", mode="min", save_top_k=1,
        ),
        LearningRateMonitor(),
    ]
    logger = CSVLogger(save_dir=str(ckpt_dir / "logs"),
                       name=f"lm_{held_out}")

    steps_per_epoch = max(1, len(train_dl))
    max_epochs = max(1, (max_steps + steps_per_epoch - 1) // steps_per_epoch)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if fast_dev else max_epochs,
        max_steps=10 if fast_dev else max_steps,
        limit_train_batches=10 if fast_dev else 1.0,
        limit_val_batches=10   if fast_dev else 1.0,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_dl, val_dl)

    best = callbacks[0].best_model_path or str(out_dir / "last.ckpt")
    if not callbacks[0].best_model_path:
        trainer.save_checkpoint(best)
    print(f"  [{held_out}] LM best ckpt: {best}")
    return Path(best)


# ---------------------------------------------------------------------------
# Stage 2 retraining with one protocol excluded
# ---------------------------------------------------------------------------

def _train_stage2_for_holdout(
    held_out:        str,
    lm_ckpt:         str | Path,
    tier1_dir:       Path,
    tier2_dir:       Path,
    ckpt_dir:        Path,
    max_steps:       int   = 20_000,
    n_msgs:          int   = 32,
    max_len:         int   = 512,
    lr:              float = 3e-4,
    val_protocols:   list[str] | None = None,
    num_workers:     int   = 0,
    fast_dev:        bool  = False,
) -> Path:
    """Train Stage 2 from scratch with `held_out` excluded.

    Returns path to best checkpoint.
    """
    out_dir = ckpt_dir / held_out / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default val protocols: pick two Tier-2 protocols that are NOT held_out
    # to use as protocol-level validation during Stage 2.
    if val_protocols is None:
        default_val = ["coap", "ospf"]
        val_protocols = [p for p in default_val if p != held_out]
        if not val_protocols:
            # held_out was one of the defaults — fall back to others
            val_protocols = ["mqtt", "snmp"]

    # Training set: Tier 1 + Tier 2 excluding held_out AND excluding val_protocols
    excl = [held_out] + val_protocols
    train_t1, _ = load_tier1_groups(tier1_dir, val_fraction=0.0)
    train_t2, _ = load_tier2_groups(
        tier2_dir,
        exclude_protocols=excl,
    )
    all_train = train_t1 + train_t2
    print(f"  [{held_out}] Stage 2 train groups: {len(all_train):,} "
          f"(excluded: {excl})")

    # Validation: the held-out val protocols
    _, val_groups = load_tier2_groups(
        tier2_dir,
        val_protocols=val_protocols,
        exclude_protocols=[],
    )
    print(f"  [{held_out}] Stage 2 val   groups: {len(val_groups):,} "
          f"(from: {val_protocols})")

    train_dl = make_format_dataloader(
        all_train, n_msgs=n_msgs, max_len=max_len,
        shuffle=True, num_workers=num_workers,
    )
    val_dl = make_format_dataloader(
        val_groups, n_msgs=min(n_msgs, 16), max_len=max_len,
        shuffle=False, num_workers=num_workers,
    )

    # Fresh model, load LM weights only (frozen)
    model = FieldInferenceModule(
        lr=lr,
        max_steps=max_steps,
        warmup_steps=min(1_000, max_steps // 20),
        lm_ckpt_path=str(lm_ckpt),
        freeze_lm=True,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename=f"stage2-{held_out}-{{step:06d}}-{{val/loss_total:.4f}}",
            monitor="val/loss_total", mode="min", save_top_k=1,
        ),
        LearningRateMonitor(),
    ]
    logger = CSVLogger(save_dir=str(ckpt_dir / "logs"),
                       name=f"stage2_{held_out}")

    steps_per_epoch = max(1, len(train_dl))
    max_epochs = max(1, (max_steps + steps_per_epoch - 1) // steps_per_epoch)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if fast_dev else max_epochs,
        max_steps=10 if fast_dev else max_steps,
        limit_train_batches=10 if fast_dev else 1.0,
        limit_val_batches=10   if fast_dev else 1.0,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=max(1, min(100, steps_per_epoch // 4)),
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dl, val_dl)

    best = callbacks[0].best_model_path or str(out_dir / "last.ckpt")
    if not callbacks[0].best_model_path:
        trainer.save_checkpoint(best)
    print(f"  [{held_out}] Stage 2 best ckpt: {best}")
    return Path(best)


# ---------------------------------------------------------------------------
# Optional final fine-tune (still no exposure to held-out protocol)
# ---------------------------------------------------------------------------

def _finetune_for_holdout(
    held_out:    str,
    stage2_ckpt: str | Path,
    tier1_dir:   Path,
    tier2_dir:   Path,
    ckpt_dir:    Path,
    max_steps:   int   = 2_000,
    n_msgs:      int   = 32,
    max_len:     int   = 512,
    lr:          float = 1e-4,
    num_workers: int   = 0,
    fast_dev:    bool  = False,
) -> Path:
    """Light fine-tune on Tier 1 + Tier 2 excluding held_out.

    This step is purely for adapting from generic Stage 2 representations;
    the held-out protocol is still NEVER seen.
    """
    out_dir = ckpt_dir / held_out / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_t1, _ = load_tier1_groups(tier1_dir, val_fraction=0.0)
    train_t2, _ = load_tier2_groups(
        tier2_dir, exclude_protocols=[held_out],
    )
    all_train = train_t1 + train_t2

    # Use a small Tier-1 val slice (no protocol-level signal needed here)
    _, val_t1 = load_tier1_groups(tier1_dir, val_fraction=0.05)
    val_groups = val_t1[:max(1, len(val_t1) // 4)]

    train_dl = make_format_dataloader(
        all_train, n_msgs=n_msgs, max_len=max_len,
        shuffle=True, num_workers=num_workers,
    )
    val_dl = make_format_dataloader(
        val_groups, n_msgs=min(n_msgs, 16), max_len=max_len,
        shuffle=False, num_workers=num_workers,
    )

    model = FieldInferenceModule.load_from_checkpoint(
        str(stage2_ckpt), strict=True,
    )
    model.hparams.lr           = lr
    model.hparams.warmup_steps = min(100, max_steps // 10)
    model.hparams.max_steps    = max(max_steps, 1)
    model.model.unfreeze_lm()

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir,
            filename=f"ft-{held_out}-{{step:05d}}-{{val/loss_total:.4f}}",
            monitor="val/loss_total", mode="min", save_top_k=1,
        ),
        LearningRateMonitor(),
    ]
    logger = CSVLogger(save_dir=str(ckpt_dir / "logs"),
                       name=f"ft_{held_out}")

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
    )
    trainer.fit(model, train_dl, val_dl)

    best = callbacks[0].best_model_path or str(out_dir / "last.ckpt")
    if not callbacks[0].best_model_path:
        trainer.save_checkpoint(best)
    print(f"  [{held_out}] Finetune best ckpt: {best}")
    return Path(best)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def train_lopo_strict(
    lm_ckpt:        str | None = None,
    tier1_dir:      str  = "data/tier1_grammar",
    tier2_dir:      str  = "data/tier2_scapy",
    ckpt_dir:       str  = "checkpoints/lopo_strict",
    retrain_lm:     bool = False,
    lm_steps:       int  = 10_000,
    stage2_steps:   int  = 20_000,
    finetune_steps: int  = 0,
    n_msgs:         int  = 32,
    max_len:        int  = 512,
    stage2_lr:      float = 3e-4,
    finetune_lr:    float = 1e-4,
    num_workers:    int  = 0,
    fast_dev:       bool = False,
    protocol:       str | None = None,
) -> dict[str, dict[str, str]]:
    """Run strict LOPO for all (or one) BI benchmark protocols.

    Returns mapping protocol → {"lm": path, "stage2": path, "finetune": path}.
    """
    tier1_dir = Path(tier1_dir)
    tier2_dir = Path(tier2_dir)
    ckpt_dir  = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not retrain_lm and lm_ckpt is None:
        raise ValueError(
            "Either pass --lm_ckpt to share a pretrained LM across all "
            "holdouts, or pass --retrain_lm to retrain the LM per holdout."
        )

    protocols = [protocol] if protocol else LOPO_PROTOCOLS

    manifest: dict[str, dict[str, str]] = {}
    for proto in protocols:
        print(f"\n{'=' * 60}\n=== STRICT LOPO: held-out = {proto} ===\n"
              f"{'=' * 60}")
        entry: dict[str, str] = {}

        # ---- Stage 1: ByteLM (shared or per-holdout) ----
        if retrain_lm:
            lm_path = _pretrain_lm_for_holdout(
                held_out=proto,
                tier1_dir=tier1_dir,
                tier2_dir=tier2_dir,
                ckpt_dir=ckpt_dir,
                max_steps=lm_steps,
                num_workers=num_workers,
                fast_dev=fast_dev,
            )
        else:
            lm_path = Path(lm_ckpt)  # type: ignore[arg-type]
        entry["lm"] = str(lm_path)

        # ---- Stage 2: full retrain with holdout excluded ----
        stage2_path = _train_stage2_for_holdout(
            held_out=proto,
            lm_ckpt=lm_path,
            tier1_dir=tier1_dir,
            tier2_dir=tier2_dir,
            ckpt_dir=ckpt_dir,
            max_steps=stage2_steps,
            n_msgs=n_msgs,
            max_len=max_len,
            lr=stage2_lr,
            num_workers=num_workers,
            fast_dev=fast_dev,
        )
        entry["stage2"] = str(stage2_path)

        # ---- Stage 3 (optional): light fine-tune ----
        if finetune_steps > 0:
            ft_path = _finetune_for_holdout(
                held_out=proto,
                stage2_ckpt=stage2_path,
                tier1_dir=tier1_dir,
                tier2_dir=tier2_dir,
                ckpt_dir=ckpt_dir,
                max_steps=finetune_steps,
                n_msgs=n_msgs,
                max_len=max_len,
                lr=finetune_lr,
                num_workers=num_workers,
                fast_dev=fast_dev,
            )
            entry["finetune"] = str(ft_path)
        else:
            entry["finetune"] = ""  # not run

        manifest[proto] = entry

        # Write manifest incrementally so a crashed run leaves partial state
        with open(ckpt_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}\nSTRICT LOPO complete. {len(manifest)} holdouts.\n"
          f"{'=' * 60}")
    for proto, entry in manifest.items():
        print(f"  {proto:12s}:")
        for stage, path in entry.items():
            print(f"      {stage:10s} = {path}")

    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lm_ckpt", default=None,
                   help="Shared LM checkpoint (skip if --retrain_lm)")
    p.add_argument("--tier1_dir", default="data/tier1_grammar")
    p.add_argument("--tier2_dir", default="data/tier2_scapy")
    p.add_argument("--ckpt_dir",  default="checkpoints/lopo_strict")
    p.add_argument("--retrain_lm",   action="store_true",
                   help="Re-pretrain ByteLM per holdout (strictest setting)")
    p.add_argument("--lm_steps",     type=int, default=10_000)
    p.add_argument("--stage2_steps", type=int, default=20_000)
    p.add_argument("--finetune_steps", type=int, default=0,
                   help="0 disables the optional fine-tuning stage")
    p.add_argument("--n_msgs",  type=int, default=32)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--stage2_lr",   type=float, default=3e-4)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--fast_dev", action="store_true")
    p.add_argument("--protocol", default=None,
                   help="Run only this protocol (smoke test)")
    args = p.parse_args()
    train_lopo_strict(**vars(args))