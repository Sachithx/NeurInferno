"""Public inference API: load a checkpoint and infer field boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from neurinferno.model.encoder import PAD_IDX
from neurinferno.model.full_model import FullModel

DEFAULT_REPO = "sachithabey/neurinferno"
DEFAULT_WEIGHT = "weights/model.ckpt"
MAX_LEN = 512
MAX_MSGS = 64


@dataclass
class Segment:
    start: int
    end: int
    hex: str


@dataclass
class MessageResult:
    """Inference result for one processed message.

    ``n_bytes`` and ``hex`` describe the bytes processed by the model. When an
    input exceeds the configured maximum length, ``original_n_bytes`` records
    its original size and ``truncated`` is true.
    """

    hex: str
    n_bytes: int
    cuts: list[bool]
    scores: list[float]
    segments: list[Segment] = field(default_factory=list)
    original_n_bytes: int | None = None
    truncated: bool = False


def parse_hex(text: str) -> bytes:
    """Parse a hex string with optional whitespace, ``0x``, ``:``, ``-``, or ``_``.

    Unsupported characters are rejected instead of being silently discarded.
    """

    s = text.strip()
    s = re.sub(r"(?i)(?<![0-9a-f])0x", "", s)
    invalid = re.search(r"[^0-9a-fA-F\s:_-]", s)
    if invalid:
        raise ValueError(
            f"invalid hex: unsupported character {invalid.group()!r} "
            f"at position {invalid.start() + 1}"
        )
    s = re.sub(r"[\s:_-]", "", s)
    if not s:
        raise ValueError("invalid hex: no bytes found")
    if len(s) % 2:
        raise ValueError(f"invalid hex: expected an even number of digits, got {len(s)}")
    return bytes.fromhex(s)


def _hp(hp: dict, key: str, default):
    return hp[key] if key in hp else default


def load_full_model(path: str | Path, device: str = "cpu") -> FullModel:
    # Checkpoints are treated as data, not executable pickle payloads.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    hp = ckpt.get("hyper_parameters") or {}
    model = FullModel(
        d_model=_hp(hp, "d_model", 128),
        n_heads=_hp(hp, "n_heads", 4),
        d_ff=_hp(hp, "d_ff", 512),
        n_layers=_hp(hp, "n_layers", 4),
        max_len=_hp(hp, "max_len", 512),
        dropout=0.0,
        lm_frozen=True,
        disable_cross_msg=_hp(hp, "disable_cross_msg", False),
        disable_entropy=_hp(hp, "disable_entropy", False),
        use_bio_head=_hp(hp, "use_bio_head", False),
        use_relational_type_head=_hp(hp, "use_relational_type_head", False),
        lm_d_model=_hp(hp, "lm_d_model", 64),
        lm_n_heads=_hp(hp, "lm_n_heads", 4),
        lm_d_ff=_hp(hp, "lm_d_ff", 256),
        lm_n_layers=_hp(hp, "lm_n_layers", 2),
    )
    sd = ckpt["state_dict"]
    stripped = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in sd.items()
        if not k.startswith("loss_weights")
    }
    model.load_state_dict(stripped, strict=False)
    model.eval()
    model.to(device)
    return model


class FieldBoundaryModel:
    """Byte-level field-boundary model. Predicts cuts, not field names."""

    def __init__(self, model: FullModel, device: str = "cpu", max_len: int = MAX_LEN):
        self.model = model
        self.device = device
        self.max_len = max_len

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> FieldBoundaryModel:
        model = load_full_model(path, device=device)
        return cls(model, device=device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_WEIGHT,
        device: str = "cpu",
        revision: str | None = None,
    ) -> FieldBoundaryModel:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return cls.from_checkpoint(path, device=device)

    def infer(
        self,
        messages: list[str | bytes],
        threshold: float = 0.75,
        max_msgs: int = MAX_MSGS,
    ) -> list[MessageResult]:
        if not messages:
            raise ValueError("need at least one message")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if max_msgs < 1:
            raise ValueError("max_msgs must be at least 1")
        if len(messages) > max_msgs:
            raise ValueError(f"received {len(messages)} messages; maximum is {max_msgs}")

        raw: list[bytes] = []
        for index, message in enumerate(messages, start=1):
            parsed = (
                bytes(message)
                if isinstance(message, (bytes, bytearray))
                else parse_hex(str(message))
            )
            if not parsed:
                raise ValueError(f"message {index} is empty")
            raw.append(parsed)

        n = len(raw)
        lens = [min(len(m), self.max_len) for m in raw]
        L = max(lens)
        x = torch.full((1, n, L), PAD_IDX, dtype=torch.long)
        for i, m in enumerate(raw):
            x[0, i, : lens[i]] = torch.tensor(list(m[: lens[i]]), dtype=torch.long)
        x = x.to(self.device)

        with torch.no_grad():
            out = self.model(x)
            bd = out.boundary_logits.squeeze(-1).squeeze(0).sigmoid()

        results: list[MessageResult] = []
        for i, m in enumerate(raw):
            n_bytes = lens[i]
            n_gaps = max(n_bytes - 1, 0)
            scores = bd[i, :n_gaps].detach().cpu().tolist() if n_gaps else []
            cuts = [s >= threshold for s in scores]
            bounds = [0] + [j + 1 for j, cut in enumerate(cuts) if cut] + [n_bytes]
            segs = [
                Segment(a, b, m[a:b].hex())
                for a, b in zip(bounds, bounds[1:], strict=False)
                if b > a
            ]
            results.append(
                MessageResult(
                    hex=m[:n_bytes].hex(),
                    n_bytes=n_bytes,
                    cuts=cuts,
                    scores=scores,
                    segments=segs,
                    original_n_bytes=len(m),
                    truncated=len(m) > n_bytes,
                )
            )
        return results


def infer_boundaries(
    messages: list[str | bytes],
    threshold: float = 0.75,
    repo_id: str = DEFAULT_REPO,
    device: str = "cpu",
) -> list[MessageResult]:
    model = FieldBoundaryModel.from_pretrained(repo_id=repo_id, device=device)
    return model.infer(messages, threshold=threshold)
