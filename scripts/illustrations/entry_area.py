"""Recompute or replot the canonical entry-area illustration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _entry_area_data import IllustrationData, IllustrationDataError, OUTPUT_NAME, load_data, recompute_data


def _assemble_figure(data: IllustrationData, figure):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    from _style import BLUE, GRAY, GREEN, INK, ORANGE, PURPLE

    arrays = data.arrays
    figure.text(
        0.065,
        0.958,
        "Why does the effective entry area have a kink?",
        fontsize=21,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.065,
        0.913,
        "3–5  |  same fixed Sun and sky point  |  overlap of entry and exit face projections",
        color=GRAY,
    )
    curve = figure.add_axes([0.075, 0.66, 0.875, 0.205])
    curve.plot(arrays["curve_s"], arrays["curve_area_over_a2"], color=ORANGE, linewidth=2)
    curve.axvspan(arrays["sample_s"][0], arrays["sample_s"][-1], color="#eef3f5", zorder=0)
    colors = (BLUE, PURPLE, ORANGE)
    labels = ("A", "B", "C")
    for index, (position, area) in enumerate(zip(arrays["sample_s"], arrays["area_over_a2"])):
        curve.scatter(position, area, s=45, color=colors[index], zorder=5)
        curve.annotate(
            labels[index],
            (position, area),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=colors[index],
            weight="bold",
            fontsize=12,
        )
    curve.set_xlim(0, arrays["curve_s"][-1])
    curve.set_ylim(0.72, 1.19)
    curve.set_xlabel(r"SO(3) arclength $s$ [rad]", labelpad=6)
    curve.set_ylabel(r"Effective entry area $A/a^2$")
    curve.grid(linewidth=0.7, color="#dbe1e6")
    handles = [
        Line2D([], [], color=ORANGE, linewidth=2, label="Entry face 3"),
        Line2D([], [], color=BLUE, linewidth=2, linestyle="--", label="Exit face 5"),
        Polygon([[0, 0], [1, 0], [1, 1]], facecolor="#b4d8c7", edgecolor=GREEN, label=r"Overlap $A_\perp$"),
    ]
    figure.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.51, 0.563),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=3,
    )

    titles = ("A · Before", "B · Boundary alignment", "C · After")
    notes = ("Upper limit:\nexit face 5", "Top edges\ncoincide", "Upper limit:\nentry face 3")
    for index in range(3):
        axis = figure.add_axes([0.055 + index * 0.315, 0.155, 0.285, 0.345])
        entry, exit_face = arrays["projected_faces"][index]
        overlap = arrays["overlap"][index]
        axis.add_patch(Polygon(entry, facecolor=ORANGE, alpha=0.07, edgecolor="none"))
        axis.add_patch(Polygon(exit_face, facecolor=BLUE, alpha=0.07, edgecolor="none"))
        axis.add_patch(Polygon(overlap, facecolor="#b4d8c7", edgecolor=GREEN, linewidth=1.2, zorder=2))
        axis.add_patch(Polygon(entry, fill=False, edgecolor=ORANGE, linewidth=1.7, zorder=3))
        axis.add_patch(
            Polygon(exit_face, fill=False, edgecolor=BLUE, linestyle="--", linewidth=1.7, zorder=4)
        )
        edge = arrays["upper_edges"][index]
        edge_color = PURPLE if index == 1 else (BLUE if index == 0 else ORANGE)
        axis.plot(*edge.T, color=edge_color, linewidth=4, zorder=6)
        axis.annotate(
            notes[index],
            edge.mean(axis=0),
            xytext=(0.53, 1.46),
            textcoords="data",
            ha="center",
            fontsize=10,
            color=edge_color,
            arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.9},
        )
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylim(-1.3, 1.7)
        axis.set_aspect("equal")
        axis.axis("off")
        figure.text(
            0.197 + index * 0.315,
            0.518,
            titles[index],
            ha="center",
            fontsize=13,
            weight="bold",
            color=colors[index],
        )
        figure.text(
            0.197 + index * 0.315,
            0.123,
            f"s = {arrays['sample_s'][index]:.3f} rad     A/a² = {arrays['area_over_a2'][index]:.3f}",
            ha="center",
            fontsize=10,
            color=INK,
        )
    figure.text(
        0.065,
        0.071,
        "View along the internal ray: intersect projected faces, then convert to incident-beam area.",
        fontsize=11,
        color=INK,
    )
    figure.text(0.065, 0.028, r"$A = A_\perp\,\cos\theta_i/\cos\theta_t$", fontsize=14, color=INK)
    figure.text(
        0.395,
        0.032,
        "All three projections use the same scale; both upper and lower limiting faces exchange at B. B is not a Jacobian critical point.",
        color=GRAY,
        fontsize=9,
    )
    return figure


def render_figure(data: IllustrationData):
    import matplotlib.pyplot as plt
    from _style import apply_style

    apply_style()
    figure = plt.figure(figsize=(14, 9.4), dpi=150)
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
            "entry-area event: "
            f"s={result.metadata['event']['s_rad']:.9f}, "
            f"J_perp={result.metadata['validation']['normal_jacobian_at_event']:.6g}, "
            f"area error={result.metadata['validation']['max_area_difference_vs_production_a2']:.3g} a^2"
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
