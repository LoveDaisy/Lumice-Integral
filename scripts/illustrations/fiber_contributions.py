"""Recompute or replot the canonical fiber-contributions illustration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _fiber_contributions_data import (
    IllustrationData,
    IllustrationDataError,
    OUTPUT_NAME,
    load_data,
    recompute_data,
)


def _assemble_figure(data: IllustrationData, figure):
    import math

    from _style import BLUE, GRAY, GREEN, INK, MARK_COLORS, ORANGE, PURPLE

    arrays = data.arrays
    metadata = data.metadata
    figure.text(
        0.075,
        0.955,
        "Which parts of the fiber carry the light?",
        fontsize=21,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.075,
        0.908,
        "3–5  |  fixed Sun and sky point  |  column density σ = 0.5°",
        color=GRAY,
    )
    epsilon = metadata["quadrature"]["epsilon"]
    epsilon_tex = rf"10^{{{int(round(math.log10(epsilon)))}}}"
    figure.text(
        0.075,
        0.842,
        rf"$g/a^2 = \rho\, (A/a^2)\, T\, K,\qquad K = 1/(J_\perp+{epsilon_tex})$",
        fontsize=16,
        color=INK,
    )
    axis = figure.add_axes([0.075, 0.19, 0.715, 0.58])
    curves = (
        (arrays["rho_pose"], BLUE, "-", "o", r"Pose density $\rho$"),
        (arrays["entry_area_over_a2"], ORANGE, "-.", "s", r"Entry area $A/a^2$"),
        (arrays["transmission"], GREEN, ":", "^", r"Transmission $T$"),
        (arrays["jacobian_factor"], PURPLE, "--", "D", r"Jacobian factor $K$"),
        (arrays["product_over_a2"], INK, "-", None, r"Product $g/a^2$"),
    )
    handles = []
    for index, (values, color, linestyle, marker, label) in enumerate(curves):
        line, = axis.semilogy(
            arrays["s"],
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2.8 if index == 4 else 1.6,
            marker=marker,
            markevery=(12 + index * 8, 66),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1,
            label=label,
            zorder=5 if index == 4 else 3,
        )
        handles.append(line)
    for index, position in enumerate(arrays["selected_s"]):
        axis.axvline(
            position,
            color=MARK_COLORS[index],
            linestyle=":",
            linewidth=0.8,
            alpha=0.25,
            zorder=1,
        )
        axis.text(
            position,
            1.018,
            str(index + 1),
            transform=axis.get_xaxis_transform(),
            ha="center",
            color=MARK_COLORS[index],
            weight="bold",
            fontsize=11,
        )
    axis.set_xlim(arrays["s"][0], arrays["s"][-1])
    axis.set_ylim(5e-5, 2500)
    axis.set_xlabel(r"SO(3) arclength $s$ [rad]", labelpad=8)
    axis.set_ylabel("Factor / product value", labelpad=9)
    axis.grid(which="major", linewidth=0.7, color="#dbe1e6")
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.81, 0.765),
        frameon=False,
        fontsize=11,
        handlelength=3,
        labelspacing=1.3,
        borderaxespad=0,
    )
    figure.text(
        0.813,
        0.365,
        rf"$I/a^2 = {metadata['quadrature']['value_over_a2']:.5f}$",
        fontsize=14,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.813,
        0.315,
        r"$I=\frac{1}{8\pi^2}\int_C g(s)\,ds$",
        fontsize=13,
        color=INK,
    )
    figure.text(
        0.075,
        0.078,
        "Closed fiber: s = 0 and s = L are the same pose. Numbered positions match pose-fiber.",
        fontsize=10,
        color=GRAY,
    )
    figure.text(
        0.075,
        0.04,
        "Area unit: a² (hexagon edge a). No individual peak normalization. "
        f"ε = ${epsilon_tex}$; path validity = 1.",
        fontsize=9,
        color=GRAY,
    )
    return figure


def render_figure(data: IllustrationData):
    import matplotlib.pyplot as plt
    from _style import apply_style

    apply_style()
    figure = plt.figure(figsize=(14, 7.6), dpi=150)
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
            "factor assembly: "
            f"relative={result.metadata['validation']['factor_product_max_relative_error']:.3g}, "
            f"integral={result.metadata['validation']['factor_grid_integral_absolute_error']:.3g}"
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
