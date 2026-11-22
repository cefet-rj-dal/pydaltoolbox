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

# Grouped Bar Chart (plot_groupedbar)

## About the chart
- Grouped bars: compares multiple measures per category, displaying side-by-side bars.

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

## Build a grouped bar chart
```python
ax = dal.plot_groupedbar(agg)
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Cleveland, W. S. (1994). *The Elements of Graphing Data* (2nd ed.). Hobart Press.
