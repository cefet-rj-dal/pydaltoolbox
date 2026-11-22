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

# Lollipop Chart (plot_lollipop)

## About the chart
- Lollipop: alternative to bars; emphasizes values with a marker and a stem, a lighter visual for comparisons.

## Environment setup
```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Aggregate by species
```python
agg = iris.groupby("Species", as_index=False)["sepal_length"].mean()
agg
```

## Build a lollipop chart
```python
ax = dal.plot_lollipop(agg, max_value_gap=0.2)
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Cleveland, W. S. (1994). *The Elements of Graphing Data* (2nd ed.). Hobart Press.
