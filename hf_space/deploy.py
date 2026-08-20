"""Upload hf_space/ to a CPU Gradio Space. Token is read from ../.env, never printed."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
SPACE = Path(__file__).resolve().parent
STAGING = SPACE / ".staging"


def load_token() -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"} and v:
                os.environ.setdefault(k, v)
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )
    if not token.startswith("hf_"):
        sys.exit("No Hugging Face token in .env (HF_TOKEN=hf_...).")
    return token


def copy_src(dest: Path) -> None:
    src = ROOT / "src" / "neurinferno"
    for rel in (
        "__init__.py",
        "inference.py",
        "cli.py",
        "model/__init__.py",
        "model/encoder.py",
        "model/byte_lm.py",
        "model/heads.py",
        "model/full_model.py",
        "data_generation/__init__.py",
        "data_generation/label_format.py",
    ):
        s = src / rel
        d = dest / "neurinferno" / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)


def copy_ckpt(dest: Path) -> None:
    matches = sorted((ROOT / "checkpoints/seed789/lopo/modbus").rglob("*.ckpt"))
    if not matches:
        sys.exit("No LOPO/modbus checkpoint on disk. Run training or download_checkpoints.sh.")
    weights = dest / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], weights / "model.ckpt")
    print(f"Bundled checkpoint {matches[0].name} ({matches[0].stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    token = load_token()
    api = HfApi(token=token)
    me = api.whoami()
    user = me["name"]
    repo_id = f"{user}/neurinferno"
    print(f"Hugging Face user: {user}")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()
    shutil.copy2(SPACE / "app.py", STAGING / "app.py")
    shutil.copy2(SPACE / "requirements.txt", STAGING / "requirements.txt")
    shutil.copy2(SPACE / "README.md", STAGING / "README.md")
    copy_src(STAGING)
    copy_ckpt(STAGING)

    create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
        token=token,
    )
    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=repo_id,
        repo_type="space",
        commit_message="CPU Gradio Space: paste same-format hex messages.",
        ignore_patterns=["**/__pycache__/**"],
    )
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print("First CPU build installs PyTorch and may take a few minutes.")


if __name__ == "__main__":
    main()
