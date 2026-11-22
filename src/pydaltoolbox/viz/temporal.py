"""Temporal plots: multi-series line, single time series, series with predictions.

Thin adapters over matplotlib that reproduce the default behaviour of the R
``daltoolbox`` ggplot2 graphics. Each function builds its own figure and returns
a ``matplotlib.axes.Axes`` (``ax.figure`` for the figure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import ColorsArg, apply_theme, new_axes, resolve_colors


def plot_series(data, label_x: str = "", label_y: str = "", colors: ColorsArg = None):
    """Line plot of one or more series sharing a common x index.

    Theory:
        Connecting successive observations with a line encodes order and rate of
        change along the x axis; overlaying several series on one scale makes
        their trends and relative levels directly comparable.

    The first column is the x index; the remaining columns are numeric series,
    each drawn as line plus points in its own colour.

    References:
        Cleveland, W. S. (1985). The Elements of Graphing Data. Wadsworth.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_series.R``. Fidelity: approximate.
    """
    frame = pd.DataFrame(data)
    x = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy()
    series_cols = list(frame.columns[1:])
    palette = resolve_colors(colors, len(series_cols))
    ax = new_axes()
    for col, color in zip(series_cols, palette):
        y = pd.to_numeric(frame[col], errors="coerce").to_numpy()
        ax.plot(x, y, color=color, linewidth=1, marker="o", markersize=4, label=str(col))
    apply_theme(ax, label_x, label_y)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(series_cols),
              frameon=False, fontsize=9)
    return ax


def plot_ts(x=None, y=None, label_x: str = "", label_y: str = "", color: str = "black"):
    """Single time-series line (points + line) over a time index.

    Theory:
        A time-series plot renders a single ordered sequence as a connected line,
        the canonical view for inspecting level, trend, seasonality and anomalies
        over time.

    If ``x`` is ``None`` an integer index ``1..n`` is used. ``color`` applies to
    both the points and the line.

    References:
        Cleveland, W. S. (1985). The Elements of Graphing Data. Wadsworth.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_ts.R``. Fidelity: approximate.
    """
    y = np.asarray(y, dtype=float).ravel()
    if x is None:
        x = np.arange(1, len(y) + 1)
    else:
        x = np.asarray(x).ravel()
    ax = new_axes()
    ax.plot(x, y, color=color, linewidth=1, marker="o", markersize=4)
    apply_theme(ax, label_x, label_y)
    return ax


def plot_ts_pred(x=None, y=None, yadj=None, ypred=None, label_x: str = "",
                 label_y: str = "", color: str = "black", color_adjust: str = "#2a78d6",
                 color_prediction: str = "#008300"):
    """Time series with in-sample adjustment and optional out-of-sample prediction.

    Theory:
        The observed series is drawn solid; the model's fitted values over the
        training window and its forecast beyond it are overlaid as dashed lines,
        so fit quality and forecast behaviour are read against the ground truth.

    ``yadj`` length defines the training segment; ``ypred`` (if given) is appended
    after it. ``color`` is the observed series, ``color_adjust`` the fitted,
    ``color_prediction`` the forecast.

    References:
        Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and
            Practice (3rd ed.). OTexts.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_ts_pred.R``. Fidelity: approximate.
    """
    y = np.asarray(y, dtype=float).ravel()
    if x is None:
        x = np.arange(1, len(y) + 1)
    else:
        x = np.asarray(x).ravel()
    yadj = np.asarray(yadj, dtype=float).ravel()
    ntrain = len(yadj)
    yhat = yadj
    if ypred is not None:
        ypred = np.asarray(ypred, dtype=float).ravel()
        yhat = np.concatenate([yhat, ypred])

    ax = new_axes()
    ax.plot(x, y, color=color, linewidth=1, marker="o", markersize=4, label="observed")
    ax.plot(x[:ntrain], yhat[:ntrain], color=color_adjust, linestyle="--",
            linewidth=1.5, label="adjusted")
    if ypred is not None:
        ntest = len(ypred)
        # include the last training point so the dashed forecast joins the fit
        ax.plot(x[ntrain - 1:ntrain + ntest], yhat[ntrain - 1:ntrain + ntest],
                color=color_prediction, linestyle="--", linewidth=1.5, label="prediction")
    apply_theme(ax, label_x, label_y)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False, fontsize=9)
    return ax
