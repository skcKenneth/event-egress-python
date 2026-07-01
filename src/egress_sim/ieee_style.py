from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt


IEEE_DOUBLE_COLUMN_IN = 7.16
IEEE_SINGLE_COLUMN_IN = 3.50

INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#D9DCE3"
BLUE = "#2E4780"
BLUE_LIGHT = "#CEDFFE"
ORANGE = "#CC6F47"
ORANGE_LIGHT = "#FFBDA1"
NEUTRAL = "#7A828F"
NEUTRAL_LIGHT = "#E2E5EA"


def use_ieee_style() -> None:
    """Apply compact IEEE-compatible typography and line settings."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_ieee(fig: plt.Figure, output_base: Path) -> None:
    """Save vector PDF and 600-dpi PNG versions of one figure."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
