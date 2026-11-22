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

# Boxplot (plot_boxplot)

## About the chart
- Boxplot: summarizes a distribution via quartiles and highlights outliers; comparable across groups.

## Environment setup
```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Boxplot per numeric column (plain, then coloured)
```python
ax = dal.plot_boxplot(iris.drop(columns="Species"))
ax.figure
```

## References
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
