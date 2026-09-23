"""Read-only access to the writing repository (``现代冰晕研究漫谈``) for tests that compare against its published text and data.

The writing repository is evidence, never a dependency: tests read its
Markdown / CSV files and skip when the checkout is absent.  Its location
defaults to the owner's layout and can be overridden with
``LUMICE_INTEGRAL_WRITING_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

WRITING_ROOT = Path(
    os.environ.get("LUMICE_INTEGRAL_WRITING_ROOT", "~/Codes/Writing-Lab/现代冰晕研究漫谈")
).expanduser()


def writing_file(relative: str) -> Path:
    """``WRITING_ROOT / relative``; skips the calling test when the file is absent."""
    path = WRITING_ROOT / relative
    if not path.is_file():
        pytest.skip(f"writing repository file not available: {path}")
    return path
