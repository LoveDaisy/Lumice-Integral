"""Content hashing and git commit lookup shared by artifact-provenance writers.

Kept free of any intra-package imports so both :mod:`.prescan` and
:mod:`.strip_io` can depend on it directly without joining the cycle
``strip_io -> strip_pixel -> discovery -> prescan``.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path | None) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


__all__ = ["sha256_of", "git_commit"]
