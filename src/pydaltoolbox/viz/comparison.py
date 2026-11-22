"""Comparison plots: bar, grouped bar, stacked bar, lollipop, pie, radar.

Thin adapters over matplotlib that reproduce the default behaviour of the
corresponding R ``daltoolbox`` ggplot2 graphics. Every function builds its own
figure and returns the ``matplotlib.axes.Axes`` (``ax.figure`` for the figure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import (
    BASELINE,
    TEXT_SECONDARY,
    ColorsArg,
    apply_theme,
    new_axes,
    resolve_colors,
)


def plot_bar(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None,
             alpha: float = 1):
    """Simple bar chart from a two-column (category, value) table.

    Theory:
        A bar chart encodes magnitude as length along a common baseline -- the
        most accurate visual channel for comparing quantities across a small set
        of discrete categories.

    The first column supplies the categories, the second the values. ``colors``
    sets a constant fill; ``alpha`` its transparency.

    References:
        Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory,
            experimentation, and application to the development of graphical
            methods. JASA, 79(387), 531-554.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_bar.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    cats = frame.iloc[:, 0].astype(str).tolist()
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy()
    fill = resolve_colors(colors, 1)[0]
    ax = new_axes()
    ax.bar(cats, values, color=fill, alpha=alpha, edgecolor="none")
    apply_theme(ax, label_x, label_y)
    return ax


def plot_groupedbar(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None,
                    alpha: float = 1):
    """Grouped (side-by-side) bar chart for several series per category.

    Theory:
        For each category the series are drawn as adjacent bars sharing a common
        baseline, so values are compared both within a category (across series)
        and across categories (within a series).

    The first column is the category; the remaining columns are numeric series.
    ``colors`` provides one fill per series.

    References:
        Cleveland, W. S., & McGill, R. (1984). Graphical perception. JASA.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_groupedbar.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    cats = frame.iloc[:, 0].astype(str).tolist()
    series_cols = list(frame.columns[1:])
    fills = resolve_colors(colors, len(series_cols))
    n = len(series_cols)
    x = np.arange(len(cats))
    width = 0.8 / max(n, 1)
    ax = new_axes()
    for i, (col, fill) in enumerate(zip(series_cols, fills)):
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy()
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, values, width=width, color=fill, alpha=alpha, label=str(col))
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    apply_theme(ax, label_x, label_y)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n,
              frameon=False, fontsize=9)
    return ax


def plot_stackedbar(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None,
                    alpha: float = 1):
    """Stacked bar chart for several series per category.

    Theory:
        Series are stacked within each category so the total height encodes the
        sum while each segment encodes a part -- emphasising composition and
        totals rather than per-series comparison.

    The first column is the category; the remaining columns are numeric series.
    ``colors`` provides one fill per series.

    References:
        Cleveland, W. S., & McGill, R. (1984). Graphical perception. JASA.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_stackedbar.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    cats = frame.iloc[:, 0].astype(str).tolist()
    series_cols = list(frame.columns[1:])
    fills = resolve_colors(colors, len(series_cols))
    ax = new_axes()
    bottom = np.zeros(len(cats))
    for col, fill in zip(series_cols, fills):
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy()
        ax.bar(cats, values, bottom=bottom, color=fill, alpha=alpha,
               edgecolor="white", linewidth=0.5, label=str(col))
        bottom = bottom + values
    apply_theme(ax, label_x, label_y)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(series_cols),
              frameon=False, fontsize=9)
    return ax


def plot_lollipop(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None,
                  color_text: str = "black", size_text: float = 3, size_ball: float = 8,
                  alpha_ball: float = 0.2, min_value: float = 0, max_value_gap: float = 1):
    """Lollipop chart (stem + circle + value label) per category.

    Theory:
        A lollipop chart is a sparser bar chart: a thin stem replaces the bar's
        area, reducing ink while preserving the position-along-a-baseline cue,
        with a circle marking the value and the (rounded) value printed inside.

    Stems run from ``min_value`` to ``value - max_value_gap``. ``colors`` sets the
    stem/circle colour; ``size_ball``/``alpha_ball`` the circle; ``color_text`` /
    ``size_text`` the label.

    References:
        Tufte, E. R. (1983). The Visual Display of Quantitative Information.
            Graphics Press.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_lollipop.R``. Fidelity: approximate.
        - ggplot point/text sizes map to matplotlib marker area / font points.
    """
    frame = pd.DataFrame(data)
    cats = frame.iloc[:, 0].astype(str).tolist()
    values = np.round(pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy())
    stem_color = resolve_colors(colors, 1)[0]
    x = np.arange(len(cats))
    ax = new_axes()
    ax.vlines(x, min_value, values - max_value_gap, color=stem_color, linewidth=1)
    ax.scatter(x, values, s=(size_ball ** 2) * 2, color=stem_color, alpha=alpha_ball,
               zorder=3)
    for xi, vi in zip(x, values):
        ax.text(xi, vi, str(int(vi)), color=color_text, fontsize=size_text * 2.845,
                ha="center", va="center", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    apply_theme(ax, label_x, label_y)
    return ax


def plot_pieplot(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None,
                 textcolor: str = "white", bordercolor: str = "black"):
    """Pie chart from a two-column (label, value) table.

    Theory:
        A pie chart maps each value to a wedge angle proportional to its share of
        the total, reading part-to-whole composition. Angle/area judgements are
        less precise than length, so pies suit a handful of clearly different
        shares.

    ``colors`` provides one fill per slice; ``textcolor`` the percentage labels;
    ``bordercolor`` the wedge edges.

    References:
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_pieplot.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    labels = frame.iloc[:, 0].astype(str).tolist()
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy()
    fills = resolve_colors(colors, len(labels))
    ax = new_axes()
    ax.set_aspect("equal")
    ax.pie(values, labels=labels, colors=fills, autopct=lambda p: f"{round(p)} %",
           startangle=90, counterclock=False,
           wedgeprops={"edgecolor": bordercolor},
           textprops={"color": textcolor})
    # Category labels (not the percentages) should use readable ink.
    for txt in ax.texts:
        if "%" not in txt.get_text():
            txt.set_color(TEXT_SECONDARY)
    if label_x:
        ax.set_xlabel(label_x)
    if label_y:
        ax.set_ylabel(label_y)
    return ax


def plot_radar(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None):
    """Radar (spider) chart for a single profile across variables.

    Theory:
        A radar chart places each variable on its own radial axis emanating from a
        common centre and connects the values into a closed polygon, giving a
        shape-based fingerprint of a multivariate profile. At least three
        variables are required for an enclosed polygon.

    ``data`` is a two-column table: variable name in the first column, value in
    the second. ``colors`` sets the polygon line/fill colour.

    References:
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_radar.R``. Fidelity: approximate.
        - Uses a native matplotlib ``polar`` projection rather than the R
          Cartesian-polygon construction; the figure is conceptually the same.
    """
    frame = pd.DataFrame(data)
    if frame.shape[1] < 2:
        raise ValueError("'data' must have at least two columns: variable name and value")
    names = frame.iloc[:, 0].astype(str).tolist()
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy()
    if np.isnan(values).any():
        raise ValueError("'data' second column must be numeric")
    if len(names) < 3:
        raise ValueError("'data' must contain at least three variables for a radar chart")
    line_color = resolve_colors(colors, 1)[0] if colors is not None else "#2a78d6"

    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    closed_values = np.concatenate([values, values[:1]])

    ax = new_axes(polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(closed_angles, closed_values, color=line_color, linewidth=2)
    ax.fill(closed_angles, closed_values, color=line_color, alpha=0.1)
    ax.scatter(angles, values, color=line_color, s=30, zorder=3)
    ax.set_xticks(angles)
    ax.set_xticklabels(names, color=TEXT_SECONDARY)
    ax.tick_params(axis="y", labelsize=8, colors=TEXT_SECONDARY)
    ax.grid(color=BASELINE, linewidth=0.5)
    if label_y:
        ax.set_ylabel(label_y)
    return ax
