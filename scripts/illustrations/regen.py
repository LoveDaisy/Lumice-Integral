"""Run the accepted theory illustrations from one explicit ordered table."""

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
    output_name: str
    script: Path
    implemented_batch_position: str


_DIRECTORY = Path(__file__).resolve().parent
COMMON_STEP = FigureStep(
    figure_id="canonical-data",
    output_name="",
    script=_DIRECTORY / "canonical_data.py",
    implemented_batch_position="shared scientific prerequisite",
)
STEPS = (
    FigureStep("world-body-frames", "world-body-frames.png", _DIRECTORY / "world_body_frames.py", "B1 figure 1/6"),
    FigureStep("deviation-level-set", "deviation-level-set.png", _DIRECTORY / "deviation_level_set.py", "B1 figure 2/6"),
    FigureStep("pose-fiber", "pose-fiber.png", _DIRECTORY / "pose_fiber.py", "B1 figure 3/6"),
    FigureStep("predictor-corrector", "predictor-corrector.png", _DIRECTORY / "predictor_corrector.py", "B1 figure 4/6"),
    FigureStep("fiber-contributions", "fiber-contributions.png", _DIRECTORY / "fiber_contributions.py", "B1 figure 5/6"),
    FigureStep("entry-area", "entry-area.png", _DIRECTORY / "entry_area.py", "B1 figure 6/6"),
)


def commands(
    *,
    mode: str,
    data_root: Path,
    output_directory: Path,
    only: str | None,
) -> list[tuple[FigureStep, str, list[str], str, tuple[Path, ...]]]:
    selected = [step for step in STEPS if only is None or step.figure_id == only]
    if only is not None and not selected:
        raise ValueError(f"unknown figure ID {only!r}; implemented: {', '.join(step.figure_id for step in STEPS)}")
    result: list[tuple[FigureStep, str, list[str], str, tuple[Path, ...]]] = []
    if mode in ("recompute", "all"):
        result.append(
            (
                COMMON_STEP,
                "recompute",
                [
                    sys.executable,
                    str(COMMON_STEP.script),
                    "--recompute",
                    "--data-dir",
                    str(data_root),
                ],
                "shared scientific recompute",
                (data_root / "canonical-fiber", data_root / "canonical-data"),
            )
        )
        for step in selected:
            result.append(
                (
                    step,
                    "recompute",
                    [
                        sys.executable,
                        str(step.script),
                        "--recompute",
                        "--canonical-data",
                        str(data_root / "canonical-data"),
                        "--data-dir",
                        str(data_root),
                    ],
                    "leaf data assembly",
                    (data_root / step.figure_id,),
                )
            )
    if mode in ("replot", "all"):
        for step in selected:
            result.append(
                (
                    step,
                    "replot",
                    [
                        sys.executable,
                        str(step.script),
                        "--replot",
                        "--data-dir",
                        str(data_root),
                        "--output-dir",
                        str(output_directory),
                    ],
                    "pure rendering",
                    (output_directory / step.output_name,),
                )
            )
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
    for index, (step, operation, command, category, outputs) in enumerate(planned, start=1):
        print(f"{index}. [{category}] {step.figure_id}:{operation} ({step.implemented_batch_position})")
        print(f"   command: {shlex.join(command)}")
        print(f"   owns: {', '.join(str(output) for output in outputs)}")
        if args.dry_run:
            continue
        completed = subprocess.run(command)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
