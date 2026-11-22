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

# Stacked Bar Chart (plot_stackedbar)

## About the chart
- Stacked bars: shows each category's composition by summing groups; useful for cumulative proportions.

## Environment setup
```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Aggregate two metrics by species
```python
agg = iris.groupby("Species", as_index=False)[["sepal_length", "sepal_width"]].mean()
agg
```

## Build a stacked bar chart
```python
ax = dal.plot_stackedbar(agg)
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Cleveland, W. S. (1994). *The Elements of Graphing Data* (2nd ed.). Hobart Press.
