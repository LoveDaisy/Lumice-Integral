"""Import graph of the package for the dependency-direction tests, parsed with :mod:`ast`.

Every ``import`` / ``from ... import`` node, relative imports resolved against
the package, so comments, strings and docstrings that merely mention a
module do not count.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lumice_integral

PACKAGE_ROOT = Path(lumice_integral.__file__).resolve().parent


def imported_modules(path: Path, package: str) -> set[str]:
    """Modules (and ``module.name`` for each ``from`` import) that ``path``, a module of ``package``, imports."""
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
