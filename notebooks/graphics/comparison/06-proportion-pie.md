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

# Pie Chart (plot_pieplot)

## About the chart
- Pie: represents proportions of a total. Use sparingly and with few categories when angles are easy to compare.

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

## Build a pie chart
```python
ax = dal.plot_pieplot(agg, colors=["#2a78d6", "#eb6834", "#1baf7a"])
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.
