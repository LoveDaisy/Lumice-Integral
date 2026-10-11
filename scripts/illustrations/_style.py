"""Visual constants shared by the accepted theory-illustration batch."""

from __future__ import annotations


INK = "#25354a"
BLUE = "#287da8"
ORANGE = "#c96635"
GREEN = "#008a78"
PURPLE = "#8962a4"
GRAY = "#788797"
LIGHT_GRAY = "#b8c5cf"
ICE = "#d2e5ee"
BACKGROUND = "#ffffff"
MARK_COLORS = (BLUE, ORANGE, GREEN, PURPLE)


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#c1cad2",
            "savefig.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
        }
    )
