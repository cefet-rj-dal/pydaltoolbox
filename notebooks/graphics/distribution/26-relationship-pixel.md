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

# Pixel View (plot_pixel)

## About the chart
- `plot_pixel`: pixel-oriented matrix visualization.

## Didactic goal
Show a dense multivariate view read as a heatmap-like structure rather than as geometric points.

## Environment setup
```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Pixel view of the numeric table
```python
ax = dal.plot_pixel(iris.drop(columns="Species"), title="Iris pixel view")
ax.figure
```

## References
- Keim, D. A. (2000). *Designing pixel-oriented visualization techniques.*
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
