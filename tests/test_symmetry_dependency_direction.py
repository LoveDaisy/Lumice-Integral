"""Dependency direction ``symmetry -> geometry``: no module of :mod:`lumice_integral.geometry` imports :mod:`lumice_integral.symmetry`.

Parsed with :mod:`ast` (``tests/_dependency_direction.py``), so comments, strings and docstrings
that merely mention the symmetry package do not count.
"""

from __future__ import annotations

from _dependency_direction import PACKAGE_ROOT, imported_modules


def test_geometry_never_imports_symmetry():
    offenders = {}
    files = sorted((PACKAGE_ROOT / "geometry").glob("*.py"))
    assert len(files) >= 7
    for path in files:
        hits = {m for m in imported_modules(path, "lumice_integral.geometry") if m.startswith("lumice_integral.symmetry")}
        if hits:
            offenders[path.name] = sorted(hits)
    assert offenders == {}


def test_the_resolver_sees_the_symmetry_to_geometry_edge():
    """Positive control: the same parser finds ``signature``'s relative imports of ``geometry``."""
    imports = imported_modules(PACKAGE_ROOT / "symmetry" / "signature.py", "lumice_integral.symmetry")
    assert "lumice_integral.geometry.unfold" in imports and "lumice_integral.symmetry.reflection_group" in imports
