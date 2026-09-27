"""``spectrum`` is a leaf: its pure modules import no JAX, numpy or other ``lumice_integral`` module; only ``store`` reaches ``s2_store``."""

from __future__ import annotations

import subprocess
import sys

from test_symmetry_dependency_direction import PACKAGE_ROOT, _imported_modules

PURE = ["__init__.py", "cmf.py", "dispersion.py", "illuminant.py", "wl_pool.py", "data/__init__.py", "data/cie_1931_cmf.py", "data/cie_daylight_basis.py"]


def _spectrum_imports(relative: str) -> set[str]:
    package = "lumice_integral.spectrum" + (".data" if relative.startswith("data/") else "")
    return _imported_modules(PACKAGE_ROOT / "spectrum" / relative, package)


def test_every_spectrum_module_is_classified() -> None:
    files = {str(p.relative_to(PACKAGE_ROOT / "spectrum")) for p in (PACKAGE_ROOT / "spectrum").rglob("*.py")}
    assert files == {*PURE, "store.py"}


def test_pure_modules_import_nothing_outside_spectrum() -> None:
    offenders = {}
    for relative in PURE:
        hits = {
            m
            for m in _spectrum_imports(relative)
            if m.split(".")[0] in {"jax", "numpy"}
            or (m.startswith("lumice_integral") and not m.startswith("lumice_integral.spectrum"))
        }
        if hits:
            offenders[relative] = sorted(hits)
    assert offenders == {}


def test_store_is_the_edge_to_s2_store() -> None:
    """Positive control for the resolver: ``store``'s relative import of ``s2_store`` is seen."""
    assert "lumice_integral.s2_store.build_or_load" in _spectrum_imports("store.py")


def test_importing_the_package_loads_no_jax_or_numpy() -> None:
    code = "import sys, lumice_integral.spectrum; print(sorted({m.split('.')[0] for m in sys.modules} & {'jax', 'numpy'}))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
