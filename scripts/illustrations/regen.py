"""Run the implemented theory illustrations from one explicit ordered table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class FigureStep:
    figure_id: str
    script: Path
    implemented_batch_position: str


STEPS = (
    FigureStep(
        figure_id="world-body-frames",
        script=Path(__file__).with_name("world_body_frames.py").resolve(),
        implemented_batch_position="B1 figure 1/6",
    ),
)


def _operations(mode: str) -> tuple[str, ...]:
    if mode == "all":
        return ("recompute", "replot")
    return (mode,)


def commands(
    *,
    mode: str,
    data_root: Path,
    output_directory: Path,
    only: str | None,
) -> list[tuple[FigureStep, str, list[str], str, Path]]:
    selected = [step for step in STEPS if only is None or step.figure_id == only]
    if only is not None and not selected:
        raise ValueError(f"unknown figure ID {only!r}; implemented: {', '.join(step.figure_id for step in STEPS)}")
    result = []
    for step in selected:
        for operation in _operations(mode):
            command = [
                sys.executable,
                str(step.script),
                f"--{operation}",
                "--data-dir",
                str(data_root),
            ]
            if operation == "recompute":
                category = "scientific recompute"
                output = data_root / step.figure_id
            else:
                category = "pure rendering"
                command.extend(["--output-dir", str(output_directory)])
                output = output_directory / f"{step.figure_id}.png"
            result.append((step, operation, command, category, output))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("replot", "recompute", "all"))
    parser.add_argument("--data-dir", type=Path, required=True, help="batch data root")
    parser.add_argument("--output-dir", type=Path, required=True, help="canonical or preview PNG directory")
    parser.add_argument("--only", help="run one implemented stable figure ID")
    parser.add_argument("--dry-run", action="store_true", help="print exact steps without importing or writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        planned = commands(
            mode=args.mode,
            data_root=args.data_dir,
            output_directory=args.output_dir,
            only=args.only,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"Implemented batch: {len(STEPS)}/6 B1 figures")
    for index, (step, operation, command, category, output) in enumerate(planned, start=1):
        print(f"{index}. [{category}] {step.figure_id}:{operation} ({step.implemented_batch_position})")
        print(f"   command: {shlex.join(command)}")
        print(f"   owns: {output}")
        if args.dry_run:
            continue
        completed = subprocess.run(command)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
