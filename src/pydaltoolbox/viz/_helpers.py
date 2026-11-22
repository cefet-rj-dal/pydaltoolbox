"""Shared palette and theme utilities for the ``viz`` plotting functions.

The default categorical palette is the colour-vision-deficiency-safe eight-hue
set recommended by the project's data-visualization guidance: hues are assigned
in a fixed order (never cycled arbitrarily) so adjacent series stay
distinguishable for readers with CVD. A light, unobtrusive theme (white panel,
hairline major grid, no minor grid) mirrors ggplot2's ``theme_bw(base_size=10)``
used throughout the R ``daltoolbox`` graphics.

References:
    Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis (2nd ed.).
        Springer.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import matplotlib.pyplot as plt

# CVD-safe categorical palette (fixed hue order), light surface.
PALETTE: list[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]

# Chart chrome / ink (light surface).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

ColorsArg = Optional[Union[str, Sequence[str]]]


def resolve_colors(colors: ColorsArg, n: int) -> list[str]:
    """Return a list of ``n`` colours honouring the ``colors`` argument.

    ``None`` falls back to the default palette (cycled if ``n`` exceeds its
    length); a single string is broadcast to every element; a sequence is used
    as given (cycled when shorter than ``n``, truncated when longer).
    """
    if colors is None:
        return [PALETTE[i % len(PALETTE)] for i in range(n)]
    if isinstance(colors, str):
        return [colors] * n
    seq = list(colors)
    if not seq:
        return [PALETTE[i % len(PALETTE)] for i in range(n)]
    return [seq[i % len(seq)] for i in range(n)]


def new_axes(figsize: tuple[float, float] = (6.0, 4.0), polar: bool = False):
    """Create a themed ``Axes`` (its figure is ``ax.figure``); ``polar`` for radar."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": polar} if polar else None)
    fig.set_facecolor(SURFACE)
    if not polar:
        apply_theme(ax)
    return ax


def apply_theme(ax, label_x: str = "", label_y: str = "") -> None:
    """Apply the light ``theme_bw``-like styling and axis labels to ``ax``."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(BASELINE)
    ax.tick_params(labelsize=9, colors=TEXT_SECONDARY)
    if label_x:
        ax.set_xlabel(label_x, fontsize=10, color=TEXT_PRIMARY)
    if label_y:
        ax.set_ylabel(label_y, fontsize=10, color=TEXT_PRIMARY)
