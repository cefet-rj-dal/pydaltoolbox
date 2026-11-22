"""Distribution plots: histogram, density, density-per-class, box, box-per-class.

Each function wraps a thin adapter over matplotlib (and, for the kernel-density
curves, ``scipy.stats.gaussian_kde``), reproducing the default behaviour of the
corresponding R ``daltoolbox`` ggplot2 graphic. Every function builds its own
figure and returns the ``matplotlib.axes.Axes`` so the caller can compose,
annotate, or save it (``ax.figure``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ._helpers import ColorsArg, apply_theme, new_axes, resolve_colors


def _as_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(data)


def plot_hist(data, label_x: str = "", label_y: str = "", color: str = "white",
              alpha: float = 0.25):
    """Histogram of a single numeric column.

    Theory:
        A histogram estimates a univariate distribution by partitioning the
        range into contiguous bins and counting observations per bin. Bin edges
        follow Sturges' rule (``k = ceil(log2(n)) + 1``), the default used by
        base R's ``graphics::hist`` that the R original mirrors.

    If ``data`` has several columns only the first is used. ``color`` sets the
    bar fill and ``alpha`` its transparency; bars are outlined in black.

    References:
        Sturges, H. A. (1926). The choice of a class interval. Journal of the
            American Statistical Association, 21(153), 65-66.
        Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis.
            Springer.

    Porting:
        Mirrors ``grf_hist.R``. Fidelity: approximate.
        - Bin edges use Sturges' rule (NumPy) to approximate ``graphics::hist``.
    """
    frame = _as_frame(data)
    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().to_numpy()
    edges = np.histogram_bin_edges(values, bins="sturges")
    ax = new_axes()
    ax.hist(values, bins=edges, color=color, alpha=alpha, edgecolor="black")
    apply_theme(ax, label_x, label_y)
    return ax


def _kde_curve(values: np.ndarray, bw):
    kde = gaussian_kde(values, bw_method=bw)
    grid = np.linspace(values.min(), values.max(), 256)
    return grid, kde(grid)


def plot_density(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None,
                 bin=None, alpha: float = 0.25):
    """Kernel density estimate for one or more numeric columns.

    Theory:
        A kernel density estimate smooths the empirical distribution by summing
        a Gaussian kernel centred on each observation; its bandwidth controls
        the degree of smoothing. With several columns the densities are overlaid
        and filled per column so their shapes can be compared on a common axis.

    ``colors`` sets the fill(s); ``alpha`` their transparency. ``bin`` overrides
    the kernel bandwidth (``bw_method``) when provided.

    References:
        Silverman, B. W. (1986). Density Estimation for Statistics and Data
            Analysis. Chapman & Hall.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_density.R``. Fidelity: approximate.
        - ``bin`` maps to the Gaussian KDE bandwidth (``bw_method``).
    """
    frame = _as_frame(data)
    cols = list(frame.columns)
    palette = resolve_colors(colors, len(cols))
    ax = new_axes()
    for col, fill in zip(cols, palette):
        values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy()
        grid, dens = _kde_curve(values, bin)
        ax.fill_between(grid, dens, color=fill, alpha=alpha, label=str(col))
        ax.plot(grid, dens, color=fill, linewidth=1)
    apply_theme(ax, label_x, label_y)
    if len(cols) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(cols),
                  frameon=False, fontsize=9)
    return ax


def plot_density_class(data, class_label: str, label_x: str = "", label_y: str = "",
                       colors: ColorsArg = None, bin=None, alpha: float = 0.5):
    """Kernel density of a numeric column grouped by a class label.

    Theory:
        One kernel density estimate (see :func:`plot_density`) is drawn per level
        of the grouping column and overlaid, so the class-conditional
        distributions of the numeric variable can be compared.

    ``data`` must carry the grouping column named by ``class_label`` plus one
    numeric column.

    References:
        Silverman, B. W. (1986). Density Estimation for Statistics and Data
            Analysis. Chapman & Hall.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_density_class.R``. Fidelity: approximate.
        - ``bin`` maps to the Gaussian KDE bandwidth (``bw_method``).
    """
    frame = pd.DataFrame(data)
    value_col = next(c for c in frame.columns if c != class_label)
    classes = list(pd.unique(frame[class_label]))
    palette = resolve_colors(colors, len(classes))
    ax = new_axes()
    for level, fill in zip(classes, palette):
        values = pd.to_numeric(frame.loc[frame[class_label] == level, value_col],
                               errors="coerce").dropna().to_numpy()
        grid, dens = _kde_curve(values, bin)
        ax.fill_between(grid, dens, color=fill, alpha=alpha, label=str(level))
        ax.plot(grid, dens, color=fill, linewidth=1)
    apply_theme(ax, label_x, label_y)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(classes),
              frameon=False, fontsize=9)
    return ax


def _style_boxes(bp, fills) -> None:
    for patch, fill in zip(bp["boxes"], fills):
        patch.set_facecolor(fill)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color("#333333")


def plot_boxplot(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None,
                 barwidth: float = 0.25):
    """Boxplot per numeric column.

    Theory:
        A box-and-whisker plot summarises a distribution by its median, the
        inter-quartile range (the box), whiskers extending to 1.5x IQR, and
        points beyond as outliers -- a compact five-number summary that makes
        spread and skew comparable across columns.

    ``barwidth`` controls box width; ``colors`` fills the boxes (one colour, or
    one per column).

    References:
        Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_boxplot.R``. Fidelity: approximate.
    """
    frame = _as_frame(data)
    cols = list(frame.columns)
    series = [pd.to_numeric(frame[c], errors="coerce").dropna().to_numpy() for c in cols]
    fills = resolve_colors(colors, len(cols))
    ax = new_axes()
    bp = ax.boxplot(series, patch_artist=True, widths=barwidth,
                    tick_labels=[str(c) for c in cols])
    _style_boxes(bp, fills)
    apply_theme(ax, label_x, label_y)
    return ax


def plot_boxplot_class(data, class_label: str, label_x: str = "", label_y: str = "",
                       colors: ColorsArg = None):
    """Boxplot of a numeric column grouped by a class label.

    Theory:
        One box-and-whisker summary (see :func:`plot_boxplot`) is drawn per level
        of the grouping column, placing the class-conditional distributions side
        by side for comparison.

    ``data`` must carry the grouping column named by ``class_label`` plus one
    numeric column.

    References:
        Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_boxplot_class.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    value_col = next(c for c in frame.columns if c != class_label)
    classes = list(pd.unique(frame[class_label]))
    series = [pd.to_numeric(frame.loc[frame[class_label] == lv, value_col],
                            errors="coerce").dropna().to_numpy() for lv in classes]
    fills = resolve_colors(colors, len(classes))
    ax = new_axes()
    bp = ax.boxplot(series, patch_artist=True, tick_labels=[str(c) for c in classes])
    _style_boxes(bp, fills)
    apply_theme(ax, label_x, label_y)
    return ax
