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

# Bar Chart with Error Bars (plot_bar)

## About the chart
- Bars with error bars: add uncertainty/variability (e.g., standard deviation) on top of a bar chart.

## Environment setup
```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Aggregate mean and standard deviation by species
```python
agg = iris.groupby("Species", as_index=False).agg(
    mean=("sepal_length", "mean"),
    sd=("sepal_length", "std"),
)
agg
```

## Build the bars and overlay error bars
```python
ax = dal.plot_bar(agg[["Species", "mean"]], colors="#2a78d6")
ax.errorbar(range(len(agg)), agg["mean"], yerr=agg["sd"],
            fmt="none", ecolor="darkred", capsize=4, linewidth=1.1)
ax.figure
```

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
