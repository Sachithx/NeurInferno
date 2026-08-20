"""CLI: neurinferno infer messages.hex"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurinferno.inference import FieldBoundaryModel, parse_hex


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="neurinferno",
        description="Infer field boundaries in unlabeled binary protocol messages.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    inf = sub.add_parser("infer", help="cut fields in hex messages")
    inf.add_argument("path", nargs="?", help="file of hex lines (default: stdin)")
    inf.add_argument("--threshold", type=float, default=0.75)
    inf.add_argument("--repo", default=None, help="Hugging Face model repo")
    inf.add_argument("--ckpt", default=None, help="local .ckpt path")
    inf.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    if args.cmd != "infer":
        return 1

    raw = Path(args.path).read_text() if args.path else sys.stdin.read()
    messages = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        messages.append(parse_hex(line))
    if not messages:
        print("no hex messages", file=sys.stderr)
        return 1

    if args.ckpt:
        model = FieldBoundaryModel.from_checkpoint(args.ckpt, device=args.device)
    else:
        kw = {"device": args.device}
        if args.repo:
            kw["repo_id"] = args.repo
        model = FieldBoundaryModel.from_pretrained(**kw)

    for i, r in enumerate(model.infer(messages, threshold=args.threshold), start=1):
        parts = [f"[{s.start}:{s.end}] {s.hex}" for s in r.segments]
        print(f"m{i}: " + " | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
