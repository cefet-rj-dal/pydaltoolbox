"""Visualization family: ``plot_*`` functions over matplotlib (optional ``[viz]``).

Each function mirrors a ggplot2 graphic from the R ``daltoolbox`` ``grf_*.R``
files, builds its own figure, and returns a ``matplotlib.axes.Axes`` so results
compose cleanly (``plot_pair`` returns the scatter-matrix array of Axes).
Requires the optional ``[viz]`` extra (``matplotlib``).
"""
from .comparison import (
    plot_bar,
    plot_groupedbar,
    plot_lollipop,
    plot_pieplot,
    plot_radar,
    plot_stackedbar,
)
from .distribution import (
    plot_boxplot,
    plot_boxplot_class,
    plot_density,
    plot_density_class,
    plot_hist,
)
from .relation import (
    plot_correlation,
    plot_pair,
    plot_parallel,
    plot_pixel,
    plot_points,
    plot_scatter,
)
from .structure import plot_dendrogram
from .temporal import plot_series, plot_ts, plot_ts_pred

__all__ = [
    "plot_bar",
    "plot_boxplot",
    "plot_boxplot_class",
    "plot_correlation",
    "plot_dendrogram",
    "plot_density",
    "plot_density_class",
    "plot_groupedbar",
    "plot_hist",
    "plot_lollipop",
    "plot_pair",
    "plot_parallel",
    "plot_pieplot",
    "plot_pixel",
    "plot_points",
    "plot_radar",
    "plot_scatter",
    "plot_series",
    "plot_stackedbar",
    "plot_ts",
    "plot_ts_pred",
]
