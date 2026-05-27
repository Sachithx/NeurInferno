"""
Stage 2 — Main joint training.

Trains encoder + FieldTypeHead + BoundaryHead on Tier 1 + Tier 2 data.
Each batch contains N=64 messages from one format, enabling cross-message
attention to learn inter-message correlations.

Usage:
    python -m src.training.train_main \
        --lm_ckpt checkpoints/lm/bylm-....ckpt \
        --ckpt_dir checkpoints \
        --epochs 50 --n_msgs 64 --lr 3e-4

Or with fast_dev for CI / smoke testing:
    python -m src.training.train_main --fast_dev
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor)
from pytorch_lightning.loggers import CSVLogger

from src.model.full_model import FullModel
from src.model.byte_lm import ByteLM
from src.training.dataset import (
    load_tier1_groups, load_tier2_groups, load_tier3_groups,
    make_format_dataloader, LOPO_PROTOCOLS,
)
from src.training.losses import compute_total_loss, LossWeights


class FieldInferenceModule(pl.LightningModule):
    """PyTorch Lightning module for Stage 2 joint training."""

    def __init__(
        self,
        lm_ckpt_path: str | None = None,
        d_model:      int   = 128,
        n_heads:      int   = 4,
        d_ff:         int   = 512,
        n_layers:     int   = 4,
        max_len:      int   = 512,
        dropout:      float = 0.1,
        lr:           float = 3e-4,
        warmup_steps: int   = 1_000,
        max_steps:    int   = 200_000,
        use_focal:    bool  = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = FullModel(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_len=max_len, dropout=dropout,
            lm_frozen=True,
        )

        # Load pretrained LM weights if provided
        if lm_ckpt_path and Path(lm_ckpt_path).exists():
            lm_state = torch.load(lm_ckpt_path, map_location="cpu")
            # Strip "lm." prefix from LightningModule checkpoint
            lm_weights_raw = {
                k.replace("lm.", "", 1): v
                for k, v in lm_state["state_dict"].items()
                if k.startswith("lm.")
            }
            # Drop keys whose shapes don't match (e.g. pos_embedding when max_len differs)
            model_sd = self.model.byte_lm.state_dict()
            lm_weights = {
                k: v for k, v in lm_weights_raw.items()
                if k in model_sd and model_sd[k].shape == v.shape
            }
            if lm_weights:
                self.model.byte_lm.load_state_dict(lm_weights, strict=False)
                print(f"Loaded LM weights from {lm_ckpt_path}")
            self.model._freeze_lm()

        self.loss_weights = LossWeights()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x         = batch["x"]           # (B, N, L)
        ft_labels = batch["ft_labels"]   # (B, N, L)
        bg_labels = batch["bg_labels"]   # (B, N, L-1)
        pad_mask  = batch["pad_mask"]    # (B, N, L)

        out = self.model(x)

        loss_info = compute_total_loss(
            boundary_logits=out.boundary_logits,
            field_logits=out.field_logits,
            bg_labels=bg_labels,
            ft_labels=ft_labels,
            pad_mask=pad_mask,
            weights=self.loss_weights,
            use_focal=self.hparams.use_focal,
        )

        # Metrics
        metrics = loss_info.as_dict(prefix=f"{stage}/")
        self.log_dict(metrics, prog_bar=(stage == "train"),
                      on_step=(stage == "train"), on_epoch=True, sync_dist=True)

        # Additional accuracy metric on non-PAD, non-UNKNOWN bytes
        if stage == "val":
            byte_mask = ~pad_mask
            from src.data_generation.label_format import FIELD_TYPE_IDS
            valid = byte_mask & (ft_labels != FIELD_TYPE_IDS["UNKNOWN"])
            if valid.sum() > 0:
                preds = out.field_logits.argmax(-1)
                acc   = (preds[valid] == ft_labels[valid]).float().mean()
                self.log("val/type_acc", acc, on_epoch=True, sync_dist=True)

            # Boundary F1 proxy (precision * recall)
            gap_mask = byte_mask[..., :-1] & byte_mask[..., 1:]
            if gap_mask.sum() > 0:
                bd_pred = (out.boundary_logits.squeeze(-1).sigmoid() > 0.5)[gap_mask]
                bd_tgt  = bg_labels[gap_mask].bool()
                tp = (bd_pred & bd_tgt).sum().float()
                fp = (bd_pred & ~bd_tgt).sum().float()
                fn = (~bd_pred & bd_tgt).sum().float()
                prec = tp / (tp + fp + 1e-8)
                rec  = tp / (tp + fn + 1e-8)
                f1   = 2 * prec * rec / (prec + rec + 1e-8)
                self.log("val/boundary_f1", f1, on_epoch=True, sync_dist=True)

        return loss_info.total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr, weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_steps, eta_min=1e-6,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        combined = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, scheduler],
            milestones=[self.hparams.warmup_steps],
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": combined, "interval": "step"}}


def train_main(
    lm_ckpt_path: str | None       = None,
    tier1_dir:    str              = "data/tier1_grammar",
    tier2_dir:    str              = "data/tier2_scapy",
    ckpt_dir:     str              = "checkpoints",
    epochs:       int              = 50,
    n_msgs:       int              = 64,
    max_len:      int              = 512,
    lr:           float            = 3e-4,
    num_workers:  int              = 0,
    fast_dev:     bool             = False,
    exclude_tier2: list[str] | None = None,
    exclude_tier3: list[str] | None = None,
    tier3_dir:    str | None       = None,
) -> Path:
    """Run Stage 2 main training. Returns path to best checkpoint."""
    tier1_dir = Path(tier1_dir)
    tier2_dir = Path(tier2_dir)
    ckpt_dir  = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data loading — hold out VAL_TIER2_PROTOCOLS from Tier 2;
    # additionally exclude any protocols requested by the caller (e.g. LOPO fold).
    train_t1, val_t1 = load_tier1_groups(tier1_dir, val_fraction=0.10)
    train_t2, val_t2 = load_tier2_groups(
        tier2_dir,
        exclude_protocols=exclude_tier2 or [],
    )
    train_t3: list = []
    if tier3_dir:
        train_t3 = load_tier3_groups(
            Path(tier3_dir),
            exclude_protocols=exclude_tier3 or [],
        )
    all_train = train_t1 + train_t2 + train_t3
    all_val   = val_t1   + val_t2

    print(f"Stage 2 data:")
    print(f"  Train: {len(all_train):,} format-groups")
    print(f"  Val:   {len(all_val):,} format-groups")

    train_dl = make_format_dataloader(
        all_train, n_msgs=n_msgs, max_len=max_len, shuffle=True,
        num_workers=num_workers)
    val_dl   = make_format_dataloader(
        all_val, n_msgs=min(n_msgs, 32), max_len=max_len, shuffle=False,
        num_workers=num_workers)

    steps_per_epoch = len(train_dl)
    max_steps       = steps_per_epoch * epochs

    model = FieldInferenceModule(
        lm_ckpt_path=lm_ckpt_path,
        max_len=max_len, lr=lr,
        warmup_steps=min(1_000, max_steps // 10),
        max_steps=max(max_steps, 1),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir / "main",
            filename="main-{epoch:03d}-{val/loss_total:.4f}",
            monitor="val/loss_total", mode="min", save_top_k=3,
        ),
        EarlyStopping(monitor="val/loss_total", patience=8, mode="min"),
        LearningRateMonitor(),
    ]
    logger = CSVLogger(save_dir=str(ckpt_dir / "logs"), name="main")

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

    best = callbacks[0].best_model_path
    print(f"Best Stage-2 checkpoint: {best}")
    return Path(best) if best else ckpt_dir / "main" / "last.ckpt"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lm_ckpt_path",  default=None)
    p.add_argument("--tier1_dir",     default="data/tier1_grammar")
    p.add_argument("--tier2_dir",     default="data/tier2_scapy")
    p.add_argument("--ckpt_dir",      default="checkpoints")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--n_msgs",        type=int,   default=64)
    p.add_argument("--max_len",       type=int,   default=512)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--fast_dev",      action="store_true")
    p.add_argument("--exclude_tier2", nargs="*", default=None,
                   metavar="PROTO",
                   help="Protocol names to exclude from Tier-2 training")
    p.add_argument("--exclude_tier3", nargs="*", default=None,
                   metavar="PROTO",
                   help="Protocol names to exclude from Tier-3 training")
    p.add_argument("--tier3_dir",     default=None,
                   help="Path to tier3_labeled directory (optional BI benchmark data)")
    args = p.parse_args()
    train_main(**vars(args))
