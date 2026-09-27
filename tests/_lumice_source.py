"""Read-only access to the Lumice source tree for tests that diff transcribed tables against its text.

The Lumice checkout is evidence, never a dependency: tests read its C++
source files as text (no build, no linking) and skip when the checkout is
absent.  Its location defaults to the owner's layout and can be overridden
with ``LUMICE_INTEGRAL_SOURCE_ROOT``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

LUMICE_SOURCE_ROOT = Path(os.environ.get("LUMICE_INTEGRAL_SOURCE_ROOT", "~/Codes/Ice Halo Simulation")).expanduser()


def lumice_source(relative: str) -> str:
    """Text of ``LUMICE_SOURCE_ROOT / relative``; skips the calling test when the file is absent."""
    path = LUMICE_SOURCE_ROOT / relative
    if not path.is_file():
        pytest.skip(f"Lumice source file not available: {path}")
    return path.read_text(encoding="utf-8")


def float_array(text: str, name: str) -> list[str]:
    """The literals of the C++ array ``name[...] = { ... };`` as source strings, ``f`` suffix and comments stripped."""
    match = re.search(r"\b" + re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\};", text, re.S)
    assert match, f"array {name} not found"
    body = re.sub(r"//[^\n]*", "", match.group(1))
    return [token.strip().removesuffix("f") for token in body.split(",") if token.strip()]


def float_constant(text: str, name: str) -> str:
    """The literal of ``constexpr ... name = <literal>;``, ``f`` suffix stripped."""
    match = re.search(r"\b" + re.escape(name) + r"\s*=\s*([-+0-9.eE]+)f?\s*;", text)
    assert match, f"constant {name} not found"
    return match.group(1)
