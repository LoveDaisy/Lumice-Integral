"""Dependency direction ``symmetry -> geometry``: no module of :mod:`lumice_integral.geometry` imports :mod:`lumice_integral.symmetry`.

Parsed with :mod:`ast` (every ``import`` / ``from ... import`` node, relative
imports resolved against the package), so comments, strings and docstrings
that merely mention the symmetry package do not count.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lumice_integral

PACKAGE_ROOT = Path(lumice_integral.__file__).resolve().parent


def _imported_modules(path: Path, package: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            out.add(module)
            out.update(f"{module}.{alias.name}" for alias in node.names)
    return out


def test_geometry_never_imports_symmetry():
    offenders = {}
    files = sorted((PACKAGE_ROOT / "geometry").glob("*.py"))
    assert len(files) >= 7
    for path in files:
        hits = {m for m in _imported_modules(path, "lumice_integral.geometry") if m.startswith("lumice_integral.symmetry")}
        if hits:
            offenders[path.name] = sorted(hits)
    assert offenders == {}


def test_the_resolver_sees_the_symmetry_to_geometry_edge():
    """Positive control: the same parser finds ``signature``'s relative imports of ``geometry``."""
    imports = _imported_modules(PACKAGE_ROOT / "symmetry" / "signature.py", "lumice_integral.symmetry")
    assert "lumice_integral.geometry.unfold" in imports and "lumice_integral.symmetry.reflection_group" in imports
