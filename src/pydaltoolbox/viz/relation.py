"""Relation plots: scatter, points, correlation, pair matrix, parallel, pixel.

Thin adapters over matplotlib / pandas.plotting / scipy that reproduce the
default behaviour of the corresponding R ``daltoolbox`` ggplot2 graphics. Unless
noted, each function builds its own figure and returns a
``matplotlib.axes.Axes`` (``ax.figure`` for the figure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pandas.plotting import parallel_coordinates, scatter_matrix

from ._helpers import (
    BASELINE,
    TEXT_PRIMARY,
    ColorsArg,
    apply_theme,
    new_axes,
    resolve_colors,
)

# Diverging blue-white-red ramp for correlation (neutral white at 0).
_CORR_CMAP = LinearSegmentedColormap.from_list(
    "dal_corr", ["#4575b4", "#ffffff", "#d73027"])
# Sequential white -> steelblue ramp for the pixel view.
_PIXEL_CMAP = LinearSegmentedColormap.from_list("dal_pixel", ["#ffffff", "#4682b4"])


def plot_scatter(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None):
    """Scatter plot of a long table coloured by a grouping column.

    Theory:
        A scatter plot maps two quantitative variables to the x and y position
        channels -- the most accurate channels for perceiving value -- exposing
        correlation, clusters, and outliers; a third (grouping) variable is
        carried by colour.

    ``data`` is a long table whose first three columns are the x value, the y
    value, and the grouping variable. ``colors`` overrides the palette; for a
    numeric grouping variable it is read as a (low, high) gradient.

    References:
        Cleveland, W. S., & McGill, R. (1984). Graphical perception. JASA.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_scatter.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    xcol, ycol, gcol = frame.columns[0], frame.columns[1], frame.columns[2]
    x = pd.to_numeric(frame[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(frame[ycol], errors="coerce").to_numpy()
    groups = frame[gcol]
    ax = new_axes()
    if pd.api.types.is_numeric_dtype(groups):
        cmap = None
        if colors is not None:
            seq = [colors] if isinstance(colors, str) else list(colors)
            cmap = LinearSegmentedColormap.from_list("dal_scatter", [seq[0], seq[-1]])
        sc = ax.scatter(x, y, c=groups.to_numpy(), cmap=cmap, s=12)
        ax.figure.colorbar(sc, ax=ax, label=str(gcol))
    else:
        levels = list(pd.unique(groups))
        palette = resolve_colors(colors, len(levels))
        for level, col in zip(levels, palette):
            mask = (groups == level).to_numpy()
            ax.scatter(x[mask], y[mask], color=col, s=12, label=str(level))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(levels),
                  frameon=False, fontsize=9)
    apply_theme(ax, label_x, label_y)
    return ax


def plot_points(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None):
    """Dot chart of several numeric series across a shared x column (points only).

    Theory:
        Like :func:`plot_series` but without connecting lines, a dot chart shows
        each series as points over a common x axis -- suited to discrete or
        unordered x where a line would imply a false continuity.

    The first column is the x index; the remaining columns are numeric series,
    each drawn in its own colour.

    References:
        Cleveland, W. S. (1985). The Elements of Graphing Data. Wadsworth.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_points.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    x = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy()
    series_cols = list(frame.columns[1:])
    palette = resolve_colors(colors, len(series_cols))
    ax = new_axes()
    for col, color in zip(series_cols, palette):
        y = pd.to_numeric(frame[col], errors="coerce").to_numpy()
        ax.scatter(x, y, color=color, s=12, label=str(col))
    apply_theme(ax, label_x, label_y)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(series_cols),
              frameon=False, fontsize=9)
    return ax


def plot_correlation(df, vars=None, method: str = "pearson",
                     triangle: str = "full", reorder: str = "none", digits: int = 2,
                     label_size: float = 3, tile_color: str = "white",
                     show_diag: bool = True, title=None):
    """Annotated correlation heatmap over numeric columns.

    Theory:
        Each cell encodes the pairwise correlation of two variables on a
        diverging blue-white-red scale (white at zero), so sign and strength read
        at a glance. Optional hierarchical reordering groups co-varying variables
        into blocks near the diagonal.

    ``method`` is ``"pearson"``, ``"spearman"`` or ``"kendall"``; ``triangle``
    keeps ``"full"``/``"upper"``/``"lower"``; ``reorder`` is ``"none"``,
    ``"hclust"`` (complete linkage on 1 - corr) or ``"alphabetical"``.

    References:
        Friendly, M. (2002). Corrgrams: Exploratory displays for correlation
            matrices. The American Statistician, 56(4), 316-324.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_correlation.R``. Fidelity: approximate.
        - ``reorder="hclust"`` uses SciPy complete linkage on ``1 - corr``.
    """
    frame = pd.DataFrame(df)
    sub = frame[list(vars)] if vars is not None else frame.select_dtypes("number")
    if sub.shape[1] < 2:
        raise ValueError("plot_correlation needs at least 2 numeric columns.")
    corr = sub.corr(method=method)

    if reorder == "alphabetical":
        order = sorted(corr.columns)
        corr = corr.loc[order, order]
    elif reorder == "hclust":
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        dist = squareform(1 - corr.to_numpy(), checks=False)
        order_idx = leaves_list(linkage(dist, method="complete"))
        names = [corr.columns[i] for i in order_idx]
        corr = corr.loc[names, names]

    names = list(corr.columns)
    n = len(names)
    mat = corr.to_numpy().astype(float).copy()
    ii, jj = np.indices((n, n))
    if triangle == "upper":
        visible = jj > ii
        if show_diag:
            visible = visible | (jj == ii)
    elif triangle == "lower":
        visible = ii > jj
        if show_diag:
            visible = visible | (ii == jj)
    else:  # full
        visible = np.ones((n, n), dtype=bool)
        if not show_diag:
            visible = visible & (ii != jj)
    mask = ~visible
    display = np.ma.array(mat, mask=mask)

    ax = new_axes(figsize=(1.1 * n + 1.5, 1.1 * n + 1.0))
    im = ax.imshow(display, cmap=_CORR_CMAP, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if not mask[i, j] and not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.{digits}f}", ha="center", va="center",
                        fontsize=label_size * 2.845, color=TEXT_PRIMARY)
    # thin separators between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=tile_color, linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                       label=f"Corr ({method})")
    if title:
        ax.set_title(title, fontweight="bold")
    return ax


def plot_pair(data, cnames, title=None, clabel=None, colors: ColorsArg = None):
    """Scatter-plot matrix (pairs) over selected columns.

    Theory:
        A scatter-plot matrix shows every pairwise scatter among several
        variables in a grid, with univariate distributions on the diagonal --
        a compact overview of bivariate structure across many variables at once.

    ``cnames`` selects the columns; ``clabel`` optionally colours points by a
    class column.

    References:
        Cleveland, W. S. (1985). The Elements of Graphing Data. Wadsworth.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_pair.R``. Fidelity: approximate.
        - Uses ``pandas.plotting.scatter_matrix``; returns the 2-D array of Axes
          (figure via ``arr.flat[0].figure``) rather than a single Axes.
    """
    frame = pd.DataFrame(data)
    cols = [c for c in cnames if c in frame.columns]
    if not cols:
        raise ValueError("plot_pair: no valid columns in 'cnames'.")
    sub = frame[cols]
    kwds = {}
    if clabel is not None:
        if clabel not in frame.columns:
            raise ValueError("plot_pair: 'clabel' is not a valid column in 'data'.")
        levels = list(pd.unique(frame[clabel]))
        palette = resolve_colors(colors, len(levels))
        lut = dict(zip(levels, palette))
        kwds["c"] = frame[clabel].map(lut).to_numpy()
    arr = scatter_matrix(sub, diagonal="kde", **kwds)
    fig = arr.flat[0].figure
    if title:
        fig.suptitle(title, fontweight="bold")
    return arr


def plot_parallel(data, columns, group, colors: ColorsArg = None, title=None):
    """Parallel-coordinates plot of several variables, lines coloured by group.

    Theory:
        Each variable becomes a parallel vertical axis and each observation a
        polyline crossing them; colour by group reveals class separation and
        multivariate patterns that pairwise plots miss.

    ``columns`` are the numeric axes; ``group`` is the class column.

    References:
        Inselberg, A. (1985). The plane with parallel coordinates. The Visual
            Computer, 1(2), 69-91.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_parallel.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    col_names = [frame.columns[c] if isinstance(c, int) else c for c in columns]
    group_name = frame.columns[group] if isinstance(group, int) else group
    levels = list(pd.unique(frame[group_name]))
    palette = resolve_colors(colors, len(levels))
    ax = new_axes()
    parallel_coordinates(frame[col_names + [group_name]], group_name, cols=col_names,
                         color=palette, ax=ax)
    apply_theme(ax)
    ax.grid(True, axis="x", color=BASELINE, linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(levels),
              frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontweight="bold")
    return ax


def plot_pixel(data, colors: ColorsArg = None, title=None, label_x: str = "sample",
               label_y: str = "Attributes"):
    """Pixel-oriented heatmap of a numeric matrix or data frame.

    Theory:
        Every value becomes a coloured cell on a sample-by-attribute grid, so an
        entire multivariate table is inspected at once -- bands, gradients and
        anomalies across samples stand out without aggregation.

    ``colors`` overrides the default white -> steelblue gradient.

    References:
        Keim, D. A. (2000). Designing pixel-oriented visualization techniques.
            IEEE TVCG, 6(1), 59-78.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_pixel.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    mat = frame.to_numpy(dtype=float)
    col_labels = list(frame.columns)
    n_samples, n_attrs = mat.shape
    cmap = _PIXEL_CMAP
    if colors is not None:
        seq = [colors] if isinstance(colors, str) else list(colors)
        cmap = LinearSegmentedColormap.from_list("dal_pixel_custom", seq) \
            if len(seq) > 1 else LinearSegmentedColormap.from_list(
                "dal_pixel_custom", ["#ffffff", seq[0]])
    ax = new_axes()
    im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap)
    ax.set_yticks(range(n_attrs))
    ax.set_yticklabels([str(c) for c in col_labels])
    ax.set_xticks(range(0, n_samples, 10))
    ax.grid(False)
    ax.set_xlabel(label_x, fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel(label_y, fontsize=10, color=TEXT_PRIMARY)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title, fontweight="bold")
    return ax
