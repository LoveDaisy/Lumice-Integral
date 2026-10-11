"""Recompute or replot the canonical predictor-corrector illustration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _predictor_corrector_data import (
    IllustrationData,
    IllustrationDataError,
    OUTPUT_NAME,
    load_data,
    recompute_data,
)


def _arrow(ax, start, end, color, *, dashed: bool = False, alpha: float = 1.0) -> None:
    ax.annotate(
        "",
        end,
        start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 2,
            "linestyle": "--" if dashed else "-",
            "alpha": alpha,
            "mutation_scale": 13,
            "shrinkA": 4,
            "shrinkB": 5,
        },
    )


def _decorate_pose_axes(ax) -> None:
    ax.set_xlabel(r"$\phi_x$")
    ax.set_ylabel(r"$\phi_y$")
    ax.set_zlabel(r"$\phi_z$")
    ax.grid(True, alpha=0.45)


def _assemble_figure(data: IllustrationData, figure):
    from matplotlib.patches import ConnectionPatch, FancyBboxPatch
    import numpy as np

    from _style import BLUE, GRAY, GREEN, INK, ORANGE

    arrays = data.arrays
    figure.suptitle(
        "3–5: follow the fiber, keep the sky point fixed",
        x=0.035,
        y=0.96,
        ha="left",
        fontsize=21,
        weight="bold",
        color=INK,
    )
    figure.text(0.04, 0.866, "Pose fiber overview", fontsize=14, weight="bold", color=INK)
    figure.text(
        0.49,
        0.866,
        "Local walk: predict → correct → repeat",
        fontsize=14,
        weight="bold",
        color=INK,
    )

    overview = figure.add_axes([0.005, 0.205, 0.425, 0.62], projection="3d")
    overview.plot(*arrays["global_curve"].T, color=BLUE, linewidth=2, alpha=0.65)
    overview.plot(*arrays["global_segment"].T, color=GREEN, linewidth=5)
    overview.scatter(*arrays["global_segment"][0], color=INK, s=48, depthshade=False)
    overview.text(*arrays["global_segment"][0], "  A", color=INK, weight="bold", fontsize=12)
    _decorate_pose_axes(overview)
    overview.tick_params(labelsize=10)
    figure.text(0.04, 0.145, "Highlighted segment → local view", fontsize=11, color=GREEN)

    connector = figure.add_axes([0.425, 0.47, 0.055, 0.10])
    _arrow(connector, (0.05, 0.3), (0.95, 0.3), GRAY)
    connector.text(0.5, 0.78, "zoom in", color=GRAY, fontsize=10, ha="center")
    connector.set_xlim(0, 1)
    connector.set_ylim(0, 1)
    connector.axis("off")

    walk = figure.add_axes([0.49, 0.17, 0.475, 0.65])
    walk.plot(*arrays["local_curve"].T, color=BLUE, linewidth=2.5)
    accepted = arrays["local_accepted"]
    predicted = arrays["local_predicted"]
    for index in range(3):
        _arrow(
            walk,
            accepted[index],
            predicted[index],
            ORANGE,
            dashed=True,
            alpha=1.0 if index == 0 else 0.75,
        )
        _arrow(
            walk,
            predicted[index],
            accepted[index + 1],
            GREEN,
            alpha=1.0 if index == 0 else 0.75,
        )
        walk.scatter(
            *predicted[index],
            color=ORANGE,
            marker="x",
            s=80 if index == 0 else 45,
            zorder=5,
        )
    for index, point in enumerate(accepted):
        color = INK if index == 0 else GREEN
        walk.scatter(
            *point,
            color=color,
            edgecolors="white",
            s=90 if index < 2 else 60,
            linewidths=0.8,
            zorder=6,
        )
        walk.annotate(
            "ABCD"[index],
            point,
            xytext=(-12, 11),
            textcoords="offset points",
            fontsize=13,
            weight="bold",
            color=color,
        )
    walk.annotate(
        "P",
        predicted[0],
        xytext=(0, -21),
        textcoords="offset points",
        ha="center",
        color=ORANGE,
        weight="bold",
        fontsize=13,
    )
    walk.text(0.014, -0.006, "predict", color=ORANGE, fontsize=11)
    walk.annotate(
        "correct",
        (predicted[0] + accepted[1]) / 2.0,
        xytext=(22, -5),
        textcoords="offset points",
        color=GREEN,
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": GREEN},
    )
    walk.text(accepted[2, 0] + 0.003, accepted[2, 1] - 0.008, "repeat", color=GRAY, fontsize=11)
    walk.text(accepted[-1, 0] + 0.008, accepted[-1, 1] + 0.005, "fiber", color=BLUE, fontsize=12)
    walk.set_xlim(-0.015, 0.15)
    walk.set_ylim(-0.012, 0.105)
    walk.set_aspect("equal")
    walk.axis("off")

    panel = FancyBboxPatch(
        (0.025, 0.455),
        0.59,
        0.535,
        boxstyle="round,pad=.01",
        transform=walk.transAxes,
        facecolor="#f7f9fc",
        edgecolor="#bccad6",
        linewidth=1,
        zorder=7,
    )
    walk.add_patch(panel)
    sky = walk.inset_axes([0.12, 0.56, 0.435, 0.37], zorder=10)
    sky.set_facecolor("#f7f9fc")
    sky.set_title("Sky near fixed target", fontsize=12, pad=9)
    offsets = arrays["sky_offsets_deg"]
    sky.axhline(0, color="#c6d4df", linewidth=1)
    sky.axvline(0, color="#c6d4df", linewidth=1)
    sky.grid(alpha=0.45)
    sky.scatter(0, 0, s=170, facecolors="none", edgecolors=GREEN, linewidth=2.3, zorder=6)
    sky.scatter(0, 0, s=36, color=INK, zorder=7)
    sky.scatter(*offsets[1], color=ORANGE, marker="x", s=85, linewidth=2, zorder=7)
    sky.annotate("A, B: target", (0, 0), xytext=(12, 10), textcoords="offset points", fontsize=10.5, color=INK)
    sky.annotate("B: corrected", (0, 0), xytext=(12, -7), textcoords="offset points", fontsize=10.5, color=GREEN)
    sky.annotate(
        f"P: misses by\n{arrays['angular_errors_deg'][1]:.3f}°",
        offsets[1],
        xytext=(-12, 12),
        textcoords="offset points",
        ha="right",
        color=ORANGE,
        fontsize=10.5,
    )
    sky.annotate(
        "",
        (0, 0),
        offsets[1],
        arrowprops={
            "arrowstyle": "-|>",
            "color": GREEN,
            "lw": 1.6,
            "connectionstyle": "arc3,rad=.22",
            "shrinkA": 8,
            "shrinkB": 10,
        },
    )
    sky.set_xlim(-0.042, 0.073)
    sky.set_ylim(-0.061, 0.021)
    sky.set_aspect("equal")
    sky.set_xticks([-0.04, 0, 0.04])
    sky.set_yticks([-0.04, -0.02, 0])
    sky.tick_params(labelsize=9)
    sky.set_xlabel("horizontal offset [°]", fontsize=9, labelpad=2)
    sky.set_ylabel("vertical offset [°]", fontsize=9, labelpad=2)

    for pose_point, sky_point, color, radius in (
        (predicted[0], offsets[1], ORANGE, -0.20),
        (accepted[1], np.zeros(2), GREEN, 0.35),
    ):
        mapping = ConnectionPatch(
            pose_point,
            sky_point,
            coordsA="data",
            coordsB="data",
            axesA=walk,
            axesB=sky,
            arrowstyle="-",
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            color=color,
            alpha=0.45,
            connectionstyle=f"arc3,rad={radius}",
            zorder=8,
            shrinkA=10,
            shrinkB=10,
        )
        figure.add_artist(mapping)

    figure.text(0.49, 0.12, "A: current    P: predicted    B: corrected", fontsize=11, color=GRAY)
    figure.text(
        0.035,
        0.035,
        "Actual production replay, same component as pose-fiber. Pose coordinates and sky angles are separate views.",
        fontsize=10,
        color=GRAY,
    )
    return figure


def render_figure(data: IllustrationData):
    import matplotlib.pyplot as plt
    from _style import apply_style

    apply_style()
    figure = plt.figure(figsize=(15, 8), dpi=150)
    try:
        return _assemble_figure(data, figure)
    except Exception:
        plt.close(figure)
        raise


def replot_data(data_root: Path | str, output_directory: Path | str) -> Path:
    import matplotlib.pyplot as plt

    data = load_data(data_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / OUTPUT_NAME
    figure = render_figure(data)
    try:
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--recompute", action="store_true")
    operation.add_argument("--replot", action="store_true")
    parser.add_argument("--canonical-data", type=Path, help="existing shared canonical-data directory; recompute only")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.recompute:
        if args.output_dir is not None:
            parser.error("--recompute does not accept --output-dir")
        result = recompute_data(args.data_dir, args.canonical_data)
        print(result.directory)
        print(
            "replay residuals: "
            f"outgoing={result.metadata['validation']['accepted_outgoing_direction_max_l2']:.3g}, "
            f"phase={result.metadata['validation']['phase_condition_error_max_rad']:.3g}, "
            f"predicted miss={result.metadata['validation']['predicted_sky_miss_deg']:.6f} deg"
        )
        return 0
    if args.canonical_data is not None:
        parser.error("--canonical-data is valid only with --recompute")
    if args.output_dir is None:
        parser.error("--replot requires --output-dir")
    try:
        output = replot_data(args.data_dir, args.output_dir)
    except IllustrationDataError as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
