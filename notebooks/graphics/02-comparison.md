---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Comparison Plots (bar, lollipop, pie, radar)

These `plot_*` functions compare magnitudes across categories. They mirror the R
`daltoolbox` ggplot2 graphics on matplotlib and need the optional `[viz]` extra
(`pip install pydaltoolbox[viz]`). Each returns a `matplotlib.axes.Axes`.

**Reference:** Cleveland, W. S., & McGill, R. (1984). *Graphical perception.* JASA, 79(387).

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()

# per-species summaries
one = iris.groupby("Species", as_index=False)["sepal_length"].mean()
two = iris.groupby("Species", as_index=False)[["sepal_length", "sepal_width"]].mean()
```

## Bar

`plot_bar` takes a two-column (category, value) table.

```python
ax = dal.plot_bar(one, colors="#2a78d6", label_y="mean sepal length")
ax.figure
```

## Grouped bar

`plot_groupedbar` places several series side by side per category.

```python
ax = dal.plot_groupedbar(two)
ax.figure
```

## Stacked bar

`plot_stackedbar` stacks the series within each category.

```python
ax = dal.plot_stackedbar(two)
ax.figure
```

## Lollipop

`plot_lollipop` is a sparser bar chart: a stem, a circle, and the rounded value.

```python
ax = dal.plot_lollipop(one, max_value_gap=0.2)
ax.figure
```

## Pie

`plot_pieplot` shows part-to-whole composition.

```python
ax = dal.plot_pieplot(one)
ax.figure
```

## Radar

`plot_radar` draws a multivariate profile over radial axes (at least three variables).

```python
import pandas as pd

profile = pd.DataFrame({
    "name": ["Petal.L", "Petal.W", "Sepal.L", "Sepal.W"],
    "value": [
        iris["petal_length"].mean(),
        iris["petal_width"].mean(),
        iris["sepal_length"].mean(),
        iris["sepal_width"].mean(),
    ],
})
ax = dal.plot_radar(profile, colors="#2a78d6")
ax.figure
```
