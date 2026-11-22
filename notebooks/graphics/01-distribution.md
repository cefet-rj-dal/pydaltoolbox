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

# Distribution Plots (histogram, density, boxplot)

The `plot_*` functions mirror the R `daltoolbox` ggplot2 graphics on top of matplotlib.
They require the optional `[viz]` extra (`pip install pydaltoolbox[viz]`). Each function
builds its own figure and returns a `matplotlib.axes.Axes`, so you can compose, annotate,
or save it (`ax.figure.savefig(...)`). When `colors` is omitted a colour-vision-deficiency
-safe palette is used.

**Reference:** Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.

```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Histogram

`plot_hist` estimates the distribution of a single numeric column with Sturges-rule bins.

```python
ax = dal.plot_hist(iris[["sepal_width"]], label_x="Sepal width", color="#2a78d6")
ax.figure
```

## Density

`plot_density` draws a smooth kernel density estimate. With several columns the densities
are overlaid and filled per column for comparison.

```python
ax = dal.plot_density(iris[["sepal_length", "sepal_width"]], label_x="value")
ax.figure
```

## Density per class

`plot_density_class` overlays one density per level of a class column.

```python
ax = dal.plot_density_class(iris[["sepal_length", "Species"]], class_label="Species")
ax.figure
```

## Boxplot

`plot_boxplot` draws one box-and-whisker summary per numeric column.

```python
ax = dal.plot_boxplot(iris.drop(columns="Species"), label_y="measurement")
ax.figure
```

## Boxplot per class

`plot_boxplot_class` groups a numeric column by a class label.

```python
ax = dal.plot_boxplot_class(iris[["petal_length", "Species"]], class_label="Species")
ax.figure
```
