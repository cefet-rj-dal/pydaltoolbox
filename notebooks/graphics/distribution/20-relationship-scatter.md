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

# Scatter Plot (plot_scatter)

## About the chart
- Scatter: assesses the relationship between two numeric variables, with optional colouring by group.

## Didactic goal
Learn to use the chart for reasoning about the data, not only for producing a figure.

## Environment setup
```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

```python
iris = dal.datasets.iris()
```

## Build a scatter plot (x, value, grouping variable)
```python
long = pd.DataFrame({
    "x": iris["sepal_length"],
    "value": iris["sepal_width"],
    "variable": iris["Species"],
})
ax = dal.plot_scatter(long, label_x="sepal_length", label_y="sepal_width")
ax.figure
```

## References
- Cleveland, W. S. (1994). *The Elements of Graphing Data* (2nd ed.). Hobart Press.
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
