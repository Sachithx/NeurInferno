"""Upload data/protocols and data/grammar to a Hugging Face dataset repo."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(SPACE))
from deploy import ROOT, load_token
STAGING = SPACE / ".dataset_staging"


def main() -> None:
    token = load_token()
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/neurinferno"
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()
    shutil.copy2(SPACE / "DATASET_README.md", STAGING / "README.md")
    shutil.copytree(
        ROOT / "data" / "protocols",
        STAGING / "protocols",
        ignore=shutil.ignore_patterns(".DS_Store"),
    )
    shutil.copytree(
        ROOT / "data" / "grammar",
        STAGING / "grammar",
        ignore=shutil.ignore_patterns(".DS_Store"),
    )

    create_repo(
        repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
        token=token,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(STAGING),
        commit_message="Add protocol traces and synthetic grammar formats.",
    )
    print(f"Done. https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
