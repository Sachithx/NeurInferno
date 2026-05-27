"""
Stage 1 — ByteLM pretraining.

Train the auxiliary byte language model on all Tier 1 + Tier 2 data
with standard next-byte cross-entropy loss.  ~10 epochs.

Usage:
    python -m src.training.train_lm \
        --tier1_dir data/tier1_grammar \
        --tier2_dir data/tier2_scapy \
        --ckpt_dir  checkpoints \
        --epochs 10 --batch_size 128 --lr 3e-4
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from src.model.byte_lm import ByteLM
from src.model.encoder import PAD_IDX
from src.training.dataset import (
    load_tier1_groups, load_tier2_groups,
    make_lm_dataloader,
)


class ByteLMModule(pl.LightningModule):
    """PyTorch Lightning wrapper for ByteLM pretraining."""

    def __init__(
        self,
        lr:         float = 3e-4,
        warmup_steps: int = 500,
        max_steps:    int = 50_000,
        max_len:      int = 256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lm = ByteLM(max_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm(x)

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x       = batch                               # (B, L)
        logits  = self.lm(x)                          # (B, L, 256)
        # Shift: predict byte[i+1] from context byte[0..i]
        target  = x[:, 1:].clone()                    # (B, L-1)
        logits  = logits[:, :-1]                      # (B, L-1, 256)
        # Mask padding positions
        valid   = (target != PAD_IDX)
        loss    = F.cross_entropy(
            logits[valid], target[valid].clamp(0, 255),
        )
        bpb = loss / math.log(2)  # bits-per-byte
        self.log(f"{stage}/loss", loss, prog_bar=(stage=="train"), on_step=(stage=="train"),
                 on_epoch=True, sync_dist=True)
        self.log(f"{stage}/bpb",  bpb,  prog_bar=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr,
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup_steps / self.hparams.max_steps,
            anneal_strategy="cos",
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def train_lm(
    tier1_dir:   str = "data/tier1_grammar",
    tier2_dir:   str = "data/tier2_scapy",
    ckpt_dir:    str = "checkpoints",
    epochs:      int = 10,
    batch_size:  int = 128,
    lr:          float = 3e-4,
    max_len:     int   = 256,
    num_workers: int   = 0,
    fast_dev:    bool  = False,
) -> Path:
    """Run Stage 1 LM pretraining, return path to best checkpoint."""
    tier1_dir = Path(tier1_dir)
    tier2_dir = Path(tier2_dir)
    ckpt_dir  = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_t1, val_t1 = load_tier1_groups(tier1_dir, val_fraction=0.05)
    train_t2, val_t2 = load_tier2_groups(tier2_dir)
    all_train = train_t1 + train_t2
    all_val   = val_t1   + val_t2
    print(f"LM data: {len(all_train)} train groups, {len(all_val)} val groups")
    total_seqs = sum(len(g.messages) for g in all_train)
    print(f"         {total_seqs:,} total sequences")

    train_dl = make_lm_dataloader(all_train, batch_size=batch_size,
                                   max_len=max_len, shuffle=True,
                                   num_workers=num_workers)
    val_dl   = make_lm_dataloader(all_val,   batch_size=batch_size,
                                   max_len=max_len, shuffle=False,
                                   num_workers=num_workers)

    steps_per_epoch = len(train_dl)
    max_steps       = steps_per_epoch * epochs

    model = ByteLMModule(lr=lr, max_steps=max(max_steps, 1), max_len=max_len,
                         warmup_steps=min(500, max_steps // 10))

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir / "lm",
            filename="bylm-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss", mode="min", save_top_k=2,
        ),
        EarlyStopping(monitor="val/loss", patience=5, mode="min"),
        LearningRateMonitor(),
    ]
    logger = CSVLogger(save_dir=str(ckpt_dir / "logs"), name="lm")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if fast_dev else epochs,
        limit_train_batches=5 if fast_dev else 1.0,
        limit_val_batches=5   if fast_dev else 1.0,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=max(1, min(50, steps_per_epoch // 4)),
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dl, val_dl)

    best_ckpt = callbacks[0].best_model_path
    print(f"Best LM checkpoint: {best_ckpt}")
    return Path(best_ckpt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tier1_dir",   default="data/tier1_grammar")
    p.add_argument("--tier2_dir",   default="data/tier2_scapy")
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--max_len",     type=int,   default=256)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--fast_dev",    action="store_true")
    args = p.parse_args()
    train_lm(**vars(args))
