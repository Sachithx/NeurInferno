"""Command-line interface for NeurInferno."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from neurinferno.inference import FieldBoundaryModel, parse_hex


def package_version() -> str:
    """Return the installed distribution version."""

    try:
        return version("neurinferno")
    except PackageNotFoundError:
        return "0.2.0"


def probability(value: str) -> float:
    """Argparse type for values in the inclusive [0, 1] range."""

    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurinferno",
        description="Infer field boundaries in unlabeled binary protocol messages.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {package_version()}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    infer_parser = sub.add_parser("infer", help="infer boundaries in hex messages")
    infer_parser.add_argument("path", nargs="?", help="file of hex lines (default: stdin)")
    infer_parser.add_argument("--threshold", type=probability, default=0.75)
    infer_parser.add_argument("--repo", default=None, help="Hugging Face model repository")
    infer_parser.add_argument("--ckpt", default=None, help="local checkpoint path")
    infer_parser.add_argument("--device", default="cpu")
    infer_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd != "infer":
        return 1

    try:
        raw = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
        messages = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                messages.append(parse_hex(line))
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
        if not messages:
            raise ValueError("no hex messages found")

        if args.ckpt:
            model = FieldBoundaryModel.from_checkpoint(args.ckpt, device=args.device)
        else:
            kwargs = {"device": args.device}
            if args.repo:
                kwargs["repo_id"] = args.repo
            model = FieldBoundaryModel.from_pretrained(**kwargs)

        results = model.infer(messages, threshold=args.threshold)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps([asdict(result) for result in results], indent=2))
        return 0

    for index, result in enumerate(results, start=1):
        parts = [f"[{segment.start}:{segment.end}] {segment.hex}" for segment in result.segments]
        suffix = " (truncated)" if result.truncated else ""
        print(f"m{index}{suffix}: " + " | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
