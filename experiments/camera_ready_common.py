"""Fixed-size camera-ready figure styling; does not alter legacy renderers."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from egress_sim.ieee_style import BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT, INK, GRID, NEUTRAL

# Measured from this paper's IEEEtran class: \textwidth = 516 TeX points.
WIDTH_IN = 516.0 / 72.27
BASELINE_COMMIT = "dc1d9f4b901751af99f760e3c2b0e64add413a24"
MIN_LABEL_PT = 7.2


def use_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.0,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2, "axes.linewidth": .7,
        "lines.linewidth": 1.1, "lines.markersize": 4.3,
        "grid.color": GRID, "grid.linewidth": .45,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": None,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def title(ax: plt.Axes, letter: str, text: str) -> None:
    ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold", pad=6, color=INK)


def despine(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def zero(ax: plt.Axes, *, horizontal: bool = False) -> None:
    method = ax.axhline if horizontal else ax.axvline
    method(0, color=INK, linestyle="--", linewidth=.9, zorder=1)


def effect(ax: plt.Axes, estimate: float, low: float, high: float,
           position: float, *, color: str = BLUE, marker: str = "o",
           horizontal: bool = True, **kwargs) -> None:
    filled = low > 0 or high < 0
    options = dict(fmt=marker, color=color, markeredgecolor=color,
                   markerfacecolor=color if filled else "white",
                   markeredgewidth=1.1, capsize=2.8, linewidth=1.0,
                   markersize=4.8, zorder=3)
    options.update(kwargs)
    if horizontal:
        ax.errorbar(estimate, position, xerr=[[estimate-low], [high-estimate]], **options)
    else:
        ax.errorbar(position, estimate, yerr=[[estimate-low], [high-estimate]], **options)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the exact canvas. Tight export would change effective text size.
    fig.savefig(base.with_suffix(".pdf"), bbox_inches=None,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches=None)
    plt.close(fig)
