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

# Relation Plots (scatter, correlation, pairs, parallel, pixel)

These `plot_*` functions explore relationships among variables. They mirror the R
`daltoolbox` ggplot2 graphics on matplotlib / `pandas.plotting` / scipy and need the
optional `[viz]` extra (`pip install pydaltoolbox[viz]`). Each returns a
`matplotlib.axes.Axes` — except `plot_pair`, which returns the scatter-matrix grid of
Axes (`arr.flat[0].figure` for the figure).

**Reference:** Cleveland, W. S., & McGill, R. (1984). *Graphical perception.* JASA, 79(387).

```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
num = iris.drop(columns="Species")
```

## Scatter

`plot_scatter` takes a long table (x, value, grouping variable) and colours points by group.

```python
long = pd.DataFrame({
    "x": iris["sepal_length"],
    "value": iris["sepal_width"],
    "variable": iris["Species"],
})
ax = dal.plot_scatter(long, label_x="sepal length", label_y="sepal width")
ax.figure
```

## Points

`plot_points` draws several numeric series over a shared x column as points only.

```python
ax = dal.plot_points(iris[["sepal_length", "sepal_width", "petal_width"]])
ax.figure
```

## Correlation

`plot_correlation` is an annotated heatmap on a diverging scale; `reorder="hclust"` groups
co-varying variables near the diagonal.

```python
ax = dal.plot_correlation(num, reorder="hclust")
ax.figure
```

## Pair (scatter matrix)

`plot_pair` shows every pairwise scatter, with densities on the diagonal; `clabel` colours
by class.

```python
arr = dal.plot_pair(iris, cnames=list(num.columns), clabel="Species")
arr.flat[0].figure
```

## Parallel coordinates

`plot_parallel` draws each observation as a polyline across the variable axes, coloured by
the class column.

```python
ax = dal.plot_parallel(iris, columns=[0, 1, 2, 3], group="Species")
ax.figure
```

## Pixel

`plot_pixel` renders the whole numeric table as a sample-by-attribute heatmap.

```python
ax = dal.plot_pixel(num, title="Iris")
ax.figure
```
