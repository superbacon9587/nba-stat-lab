"""Upload ``data/deploy/`` to a Hugging Face dataset repo.

The repo is created **private** by default. The raw source data has its own
license terms, so check them before making a derived copy public. A private
repo needs a read-only ``HF_TOKEN`` in the app's secrets.

Usage (after ``hf auth login``, or with ``HF_TOKEN`` set)::

    python -m nbalab.deploy.slim
    python -m nbalab.deploy.publish --repo yourname/nbalab-data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nbalab.deploy.slim import DEPLOY_DIR, MANIFEST_FILE

DATASET_CARD = """---
license: other
pretty_name: NBA Stat Lab app data
---

Column-trimmed, zstd-compressed parquet tables used by the NBA Stat Lab app.
Derived from public NBA box scores and play-by-play; see `manifest.json` for
row counts, build date, and the first season included.
"""


def publish(repo_id: str, folder: Path = DEPLOY_DIR, private: bool = True) -> str:
    """Create (if needed) and update the dataset repo. Removes tables no longer shipped."""
    from huggingface_hub import HfApi

    if not (folder / MANIFEST_FILE).exists():
        raise FileNotFoundError(f"{folder / MANIFEST_FILE} missing; run `python -m nbalab.deploy.slim` first")
    (folder / "README.md").write_text(DATASET_CARD)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=folder,
        allow_patterns=["*.parquet", MANIFEST_FILE, "README.md"],
        delete_patterns=["*.parquet"],
        commit_message="Update app data",
    )
    return str(info.commit_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="e.g. yourname/nbalab-data")
    parser.add_argument("--folder", type=Path, default=DEPLOY_DIR)
    parser.add_argument("--public", action="store_true", help="create the repo public (default private)")
    args = parser.parse_args()
    print(publish(args.repo, args.folder, private=not args.public))


if __name__ == "__main__":
    main()
