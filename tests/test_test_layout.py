"""Test modules never import each other: shared helpers live in ``tests/_*.py`` (four such imports had crept in)."""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _test_module_imports(source: str) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(source)):
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names = [node.module]
        out.update(name for name in names if name.split(".")[0].startswith("test_") or name.split(".")[0] == "conftest")
    return out


def test_no_test_module_imports_another() -> None:
    offenders = {p.name: sorted(hits) for p in sorted(TESTS.glob("test_*.py")) if (hits := _test_module_imports(p.read_text(encoding="utf-8")))}
    assert offenders == {}


def test_the_check_sees_a_test_import() -> None:
    """Negative control: both import forms are caught, a string mentioning one is not."""
    source = 'import test_optics\nfrom test_weights import helper\n"from test_x import y"\nfrom _numeric import central_difference_jacobian\n'
    assert _test_module_imports(source) == {"test_optics", "test_weights"}
