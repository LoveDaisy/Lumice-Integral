"""Recompute the shared canonical production data for theory illustrations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _canonical_data import recompute_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = recompute_data(args.data_dir)
    print(result.directory)
    print(
        "canonical grids: "
        f"presentation={result.metadata['sampling']['presentation_node_count']}, "
        f"quadrature={result.metadata['sampling']['quadrature_node_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
