"""Tests for the viz plot_* family (step 2.3, optional [viz] extra).

matplotlib is required; the whole module is skipped when it is unavailable. The
non-interactive ``Agg`` backend is forced so the tests never open a window. Each
plot function must return a ``matplotlib.axes.Axes`` carrying the expected number
of drawn artists.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes

import pydaltoolbox as dal


@pytest.fixture
def frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(5.0, 1.0, 60),
        "b": rng.normal(3.0, 0.5, 60),
        "cls": np.repeat(["x", "y", "z"], 20),
    })


@pytest.fixture
def summary():
    return pd.DataFrame({"cat": ["p", "q", "r"], "v1": [3.0, 5.0, 4.0], "v2": [1.0, 2.0, 1.5]})


def test_plot_hist_returns_axes_with_bars(frame):
    ax = dal.plot_hist(frame[["a"]], label_x="a", color="white")
    assert isinstance(ax, Axes)
    assert len(ax.patches) >= 1
    assert ax.get_xlabel() == "a"


def test_plot_density_single_and_multi(frame):
    ax1 = dal.plot_density(frame[["a"]])
    assert isinstance(ax1, Axes)
    assert ax1.get_legend() is None  # single series: no legend box
    ax2 = dal.plot_density(frame[["a", "b"]])
    assert ax2.get_legend() is not None
    assert len(ax2.get_legend().get_texts()) == 2


def test_plot_density_class(frame):
    ax = dal.plot_density_class(frame[["a", "cls"]], "cls")
    assert isinstance(ax, Axes)
    assert len(ax.get_legend().get_texts()) == 3  # one per class


def test_plot_boxplot_one_box_per_column(frame):
    ax = dal.plot_boxplot(frame[["a", "b"]])
    assert isinstance(ax, Axes)
    # patch_artist draws one filled box patch per column
    assert len(ax.patches) == 2


def test_plot_boxplot_class(frame):
    ax = dal.plot_boxplot_class(frame[["a", "cls"]], "cls")
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 3


def test_plot_bar(summary):
    ax = dal.plot_bar(summary[["cat", "v1"]], colors="#2a78d6")
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 3


def test_plot_groupedbar(summary):
    ax = dal.plot_groupedbar(summary)
    assert isinstance(ax, Axes)
    # 3 categories x 2 series
    assert len(ax.patches) == 6
    assert ax.get_legend() is not None


def test_plot_stackedbar(summary):
    ax = dal.plot_stackedbar(summary)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 6


def test_plot_lollipop(summary):
    ax = dal.plot_lollipop(summary[["cat", "v1"]], max_value_gap=0.2)
    assert isinstance(ax, Axes)
    assert ax.collections  # vlines + scatter are collections
    labels = [t.get_text() for t in ax.texts]
    assert "3" in labels and "5" in labels  # rounded values printed


def test_plot_pieplot(summary):
    ax = dal.plot_pieplot(summary[["cat", "v1"]])
    assert isinstance(ax, Axes)
    # 3 wedges
    from matplotlib.patches import Wedge
    assert sum(isinstance(p, Wedge) for p in ax.patches) == 3


def test_plot_radar(summary):
    data = pd.DataFrame({"name": ["A", "B", "C", "D"], "value": [1.0, 2.0, 3.0, 2.5]})
    ax = dal.plot_radar(data, colors="red")
    assert isinstance(ax, Axes)
    assert ax.name == "polar"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C", "D"]


def test_plot_radar_requires_three_vars():
    data = pd.DataFrame({"name": ["A", "B"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError):
        dal.plot_radar(data)


def test_colors_argument_applied(summary):
    ax = dal.plot_bar(summary[["cat", "v1"]], colors="#e34948")
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx(
        matplotlib.colors.to_rgb("#e34948"), abs=1e-6)
