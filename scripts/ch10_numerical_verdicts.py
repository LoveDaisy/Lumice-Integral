"""The chapter-10 numerical verdicts as figure data (task ``ch10-numerical-verdicts``; ``docs/phase2.md`` section 10).

Runs :mod:`lumice_integral.ch10_verdicts` and writes one figure-data directory
per verdict (``metadata.json`` + ``arrays.npz``, schema
``lumice-integral.ch10-verdict/v1``, :func:`.figure_data.export_verdict_figure_data`)
and a top-level ``provenance.json`` listing them.  Every verdict's provenance
records this repository's commit, the command, and the last commit of each
module the numbers come from (``git log -1 -- <file>``: the field layer,
the extraction, the quadrature, the focusing labels).

Usage::

    uv run python scripts/ch10_numerical_verdicts.py --output-dir /tmp/ch10-verdicts
    uv run python scripts/ch10_numerical_verdicts.py --verdict inner-edge --output-dir /tmp/ch10-inner-edge

Wall clock on an M2 Max, one process: inner-edge ~30 s, liljequist ~3 min,
parhelic-circle ~6 min, parallel-face ~1 min.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from lumice_integral import ch10_verdicts
from lumice_integral.figure_data import VERDICT_SCHEMA_VERSION, export_verdict_figure_data
from lumice_integral.provenance import git_commit

REPOSITORY = Path(__file__).resolve().parents[1]
# The modules each verdict's numbers come from (their last commits go into the provenance).
SOURCES = (
    "src/lumice_integral/dp_field",
    "src/lumice_integral/contour.py",
    "src/lumice_integral/contour_quadrature.py",
    "src/lumice_integral/focusing.py",
    "src/lumice_integral/ch10_verdicts.py",
    "src/lumice_integral/s2_store.py",
    "src/lumice_integral/optics.py",
)


def last_commit(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=REPOSITORY, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def dirty(paths: tuple[str, ...]) -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=REPOSITORY, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verdict", choices=(*ch10_verdicts.VERDICTS, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    names = ch10_verdicts.VERDICTS if args.verdict == "all" else (args.verdict,)
    common = {
        "repository_commit": git_commit(REPOSITORY),
        "source_commits": {path: last_commit(path) for path in SOURCES},
        "sources_modified_in_worktree": dirty(SOURCES),
        "command": " ".join([Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "task": "ch10-numerical-verdicts (scrum phase2-contour-quadrature)",
    }
    index = {"schema": VERDICT_SCHEMA_VERSION, "provenance": common, "verdicts": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        start = time.perf_counter()
        verdict = ch10_verdicts.RUNNERS[name]()
        seconds = time.perf_counter() - start
        files = export_verdict_figure_data(verdict, args.output_dir / name, provenance={**common, "wall_clock_s": seconds})
        index["verdicts"][name] = {
            "status": verdict.status,
            "statement": verdict.statement,
            "metadata": str(files.metadata.relative_to(args.output_dir)),
            "arrays": str(files.arrays.relative_to(args.output_dir)),
            "wall_clock_s": seconds,
        }
        print(f"[{name}] {verdict.status} ({seconds:.1f} s)\n  {verdict.statement}\n", flush=True)
    (args.output_dir / "provenance.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
